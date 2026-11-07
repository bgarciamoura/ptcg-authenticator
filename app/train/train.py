"""Etapa 2: treina classificador, anomalia k-NN e anomalia condicionada à referência.

Uso: docker compose run --rm trainer python -m app.train.train [--no-promote]
Requer data/features.npz (gerado por app.train.prepare).

Cada execução cria models/runs/<timestamp>/ com os modelos, metrics.json,
manifest.json e, quando há dados suficientes, thresholds.json (limiares
sugeridos pela curva PR out-of-fold). Salvo --no-promote, aponta latest.json.

- Classificador: grupos invariantes + textura (sem arte), PCA, regressão
  logística e calibração de Platt em predições out-of-fold. Precisa de reais e
  falsas com >= 2 cartas físicas por classe.
- Anomalia k-NN: só cartas REAIS (>= 5), mesmos grupos.
- Anomalia por referência: >= 5 fotos reais com carta oficial encontrada.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from app.config import get_settings
from app.pipeline.embedder import ANOMALY_GROUPS, CLASSIFIER_GROUPS, select_groups
from app.pipeline.models import KnnAnomaly, ReferenceAnomaly, VisualClassifier
from app.pipeline.registry import new_run_dir, promote

log = logging.getLogger("train")
MIN_SAMPLES_FOR_THRESHOLDS = 40
MIN_PER_CLASS_FOR_THRESHOLDS = 10


def evaluate(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "recall_fake": round(float(recall_score(y, pred, zero_division=0)), 4),
        "precision_fake": round(float(precision_score(y, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def suggest_thresholds(y: np.ndarray, prob: np.ndarray, defaults: tuple[float, float]) -> dict | None:
    """threshold_fake: menor limiar >= 0.5 com precisão >= 0.9 para 'falsa';
    threshold_real: maior limiar <= 0.5 (e >= 0.1 abaixo de threshold_fake)
    abaixo do qual ficam <= 5% das falsas. Só com dados suficientes."""
    if len(y) < MIN_SAMPLES_FOR_THRESHOLDS or min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS_FOR_THRESHOLDS:
        return None
    grid = np.round(np.linspace(0.05, 0.95, 91), 3)
    fake = next(
        (t for t in grid if t >= 0.5 and (prob >= t).sum() > 0 and precision_score(y, prob >= t, zero_division=0) >= 0.9),
        None,
    )
    if fake is None:
        return None
    real = next((t for t in grid[::-1] if t <= min(0.5, fake - 0.1) and (prob[y == 1] < t).mean() <= 0.05), None)
    if real is None:
        return None
    return {"threshold_fake": round(float(fake), 3), "threshold_real": round(float(real), 3), "n": int(len(y))}


def train_classifier(X, y, groups, meta, C, n_components, defaults, out: Path) -> dict | None:
    groups_per_class = [len(set(groups[y == c])) for c in (0, 1)]
    if min(groups_per_class) < 2:
        log.warning("classificador pulado: precisa de >= 2 cartas físicas por classe (tem %s)", groups_per_class)
        return None

    n_splits = min(5, *groups_per_class)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_logits = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        model = VisualClassifier(CLASSIFIER_GROUPS, meta, C=C, n_components=n_components).fit(X[tr], y[tr])
        oof_logits[te] = model.decision(X[te])
        log.info("fold %d/%d ok", fold, n_splits)

    final = VisualClassifier(CLASSIFIER_GROUPS, meta, C=C, n_components=n_components).fit(X, y)
    final.calibrate(oof_logits, y)  # Platt sobre logits out-of-fold
    oof_prob = 1.0 / (1.0 + np.exp(-(final.calib_a * oof_logits + final.calib_b)))
    final.save(out)
    log.info("classificador salvo em %s", out)
    thresholds = suggest_thresholds(y, oof_prob, defaults)
    return {
        "cv_folds": n_splits,
        "groups": list(CLASSIFIER_GROUPS),
        "pca_components": int(final.components.shape[0]),
        "calibration": {"a": round(final.calib_a, 4), "b": round(final.calib_b, 4)},
        "oof_at_0.5": evaluate(y, oof_prob, 0.5),
        "oof_at_default_fake": evaluate(y, oof_prob, defaults[0]),
        "suggested_thresholds": thresholds,
    }


def train_anomaly(X_sel, y, meta, k, out: Path) -> dict | None:
    real = X_sel[y == 0]
    if len(real) < 5:
        log.warning("anomalia pulada: precisa de >= 5 fotos reais (tem %d)", len(real))
        return None
    model = KnnAnomaly(ANOMALY_GROUPS, meta, k=k).fit(real)
    model.save(out)
    info = {"n_real": int(len(real)), "threshold": round(model.threshold, 5), "groups": list(ANOMALY_GROUPS)}
    fakes = X_sel[y == 1]
    if len(fakes):
        scores = np.array([model.normalized(f) for f in fakes])
        # indicativo apenas: o banco não contém essas falsas, mas contém todas as reais
        info["fake_flagged_ratio"] = round(float((scores > 1.0).mean()), 4)
    log.info("anomalia salva em %s", out)
    return info


def train_reference_anomaly(ref_dist, y, ref_groups, meta, out: Path) -> dict | None:
    valid = ~np.isnan(ref_dist).any(axis=1)
    real = ref_dist[valid & (y == 0)]
    if len(real) < 5:
        log.warning("anomalia por referência pulada: precisa de >= 5 reais com carta oficial (tem %d)", len(real))
        return None
    model = ReferenceAnomaly(ref_groups, meta).fit(real)
    model.save(out)
    info = {"n_real": int(len(real)), "thresholds": {g: round(float(t), 4) for g, t in zip(ref_groups, model.thresholds)}}
    fakes = ref_dist[valid & (y == 1)]
    if len(fakes):
        info["fake_flagged_ratio"] = round(float(np.mean([model.normalized(f) > 1.0 for f in fakes])), 4)
    log.info("anomalia por referência salva em %s", out)
    return info


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=s.data_dir / "features.npz")
    parser.add_argument("--models", type=Path, default=s.models_dir)
    parser.add_argument("--C", type=float, default=0.1, help="regularização da regressão logística")
    parser.add_argument("--pca", type=int, default=32, help="componentes do PCA")
    parser.add_argument("--k", type=int, default=5, help="vizinhos do detector de anomalia")
    parser.add_argument("--no-promote", action="store_true", help="não aponta latest.json para este run")
    args = parser.parse_args(argv)

    if not args.features.exists():
        raise SystemExit(f"{args.features} não existe; rode antes: python -m app.train.prepare")
    data = np.load(args.features, allow_pickle=False)
    X, y, groups = data["X"], data["y"], data["groups"]
    emb_meta = json.loads(str(data["meta"]))
    emb_groups: list[str] = emb_meta["groups"]
    meta = {"embedder": emb_meta}
    defaults = (s.threshold_fake, s.threshold_real)
    log.info("X=%s | reais=%d falsas=%d", X.shape, int((y == 0).sum()), int((y == 1).sum()))

    X_clf = select_groups(X, emb_groups, CLASSIFIER_GROUPS)
    X_anom = select_groups(X, emb_groups, ANOMALY_GROUPS)

    run = new_run_dir(args.models)
    report = {
        "run": run.name,
        "classifier": train_classifier(X_clf, y, groups, meta, args.C, args.pca, defaults, run / "classifier.json"),
        "anomaly": train_anomaly(X_anom, y, meta, args.k, run / "anomaly.npz"),
        "ref_anomaly": None,
    }
    if "ref_dist" in data:
        ref_groups = json.loads(str(data["ref_groups"]))
        report["ref_anomaly"] = train_reference_anomaly(data["ref_dist"], y, ref_groups, meta, run / "ref_anomaly.json")

    thresholds = (report["classifier"] or {}).get("suggested_thresholds")
    if thresholds:
        (run / "thresholds.json").write_text(json.dumps(thresholds))
        log.info("limiares sugeridos gravados: %s", thresholds)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "features_file": str(args.features),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "dataset_fingerprint": str(data["fingerprint"]) if "fingerprint" in data else None,
        "pairing": json.loads(str(data["pairing"])) if "pairing" in data else None,
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "n_physical_cards": len(set(groups.tolist())),
        "embedder": emb_meta,
        "params": {"C": args.C, "pca": args.pca, "k": args.k},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (run / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.no_promote:
        log.info("run %s criado sem promover (latest.json inalterado)", run.name)
    else:
        promote(args.models, run)
        log.info("run %s promovido; a API recarrega sozinha na próxima análise", run.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
