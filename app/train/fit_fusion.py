"""Etapa 3 (opcional): calibra a fusão dos sinais com um conjunto SEPARADO.

Uso: docker compose run --rm trainer python -m app.train.fit_fusion
Lê data/validation/<real|fake>/<grupo>/*.jpg — fotos de cartas que NÃO estão
em data/raw (senão o sinal visual fica otimista e a fusão aprende errado).
Roda o pipeline completo (OCR, bases de referência) e grava fusion.json no run
atual. A fusão registra os componentes presentes e é ignorada se mudarem.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from app.config import Settings, get_settings
from app.pipeline.fusion import SIGNAL_NAMES
from app.pipeline.imageio import read_image_file
from app.pipeline.models import FusionModel
from app.pipeline.registry import resolve_latest
from app.train.dataset import iter_labeled_images
from app.train.train import evaluate

log = logging.getLogger("fit_fusion")


def make_analyzer(s: Settings):
    from app.pipeline.service import Analyzer

    return Analyzer(s.model_copy(update={"store_inferences": False}))  # não persiste a validação


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=s.data_dir / "validation")
    args = parser.parse_args(argv)

    run_dir = resolve_latest(s.models_dir)
    if run_dir is None:
        raise SystemExit("nenhum run promovido; rode antes: python -m app.train.train")

    analyzer = make_analyzer(s)
    analyzer.load()
    if analyzer.classifier is None and analyzer.anomaly is None:
        raise SystemExit("o run atual não tem modelos visuais; treine antes")

    rows, labels, groups = [], [], []
    for sample in iter_labeled_images(args.validation):
        try:
            signals, report, _ = analyzer.compute_signals(read_image_file(sample.path))
        except ValueError as exc:
            log.warning("pulando %s: %s", sample.path, exc)
            continue
        if not report["card_detected"]:
            log.warning("pulando %s: carta não detectada", sample.path)
            continue
        rows.append(signals.to_dict())
        labels.append(sample.label)
        groups.append(sample.group)
        log.info("%s -> %s", sample.path.name, {k: None if v is None else round(v, 3) for k, v in rows[-1].items()})

    y = np.array(labels)
    if len(y) < 10 or len(set(y.tolist())) < 2:
        raise SystemExit(f"poucos exemplos de validação ({len(y)}); precisa de reais e falsas, idealmente 30+")

    meta = {"embedder": analyzer.embedder.meta(), "components": analyzer.components()}
    report = {"n": int(len(y)), "components": analyzer.components()}

    groups_arr = np.array(groups)
    groups_per_class = [len(set(groups_arr[y == c])) for c in (0, 1)]
    if min(groups_per_class) >= 2:
        n_splits = min(5, *groups_per_class)
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(rows, y, groups_arr):
            m = FusionModel(SIGNAL_NAMES, meta).fit([rows[i] for i in tr], y[tr])
            oof[te] = [m.predict_proba(rows[i]) for i in te]
        report["cv_metrics"] = evaluate(y, oof, s.threshold_fake)

    model = FusionModel(SIGNAL_NAMES, meta).fit(rows, y)
    model.save(run_dir / "fusion.json")
    report["coefficients"] = model.coefficients()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("fusão salva em %s — a API recarrega sozinha", run_dir / "fusion.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
