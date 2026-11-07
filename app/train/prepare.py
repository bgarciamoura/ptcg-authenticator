"""Etapa 1: retifica as fotos, extrai embeddings e (opcional) distâncias à carta oficial.

Uso: docker compose run --rm trainer python -m app.train.prepare [--no-reference]
Saídas:
  data/processed/<grupo>/<arquivo>.png   cartas retificadas (CONFIRA visualmente)
  data/features.npz                      X, y, groups, paths, meta, fingerprint,
                                         ref_dist (n x grupos; NaN sem referência)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.pipeline.detect import CARD_H, CARD_W, rectify
from app.pipeline.embedder import REFERENCE_GROUPS, group_distances, select_groups
from app.pipeline.imageio import read_image_file
from app.pipeline.ocr import parse_card_text
from app.train.dataset import (
    dataset_fingerprint,
    iter_feedback_samples,
    iter_labeled_images,
    pairing_report,
)

log = logging.getLogger("prepare")


# Fábricas separadas para os testes injetarem dublês.
def make_embedder(s: Settings):
    from app.pipeline.embedder import Embedder

    return Embedder(s.dino_model, s.device, s.embed_size, s.allow_model_download)


def make_ocr(s: Settings):
    from app.pipeline.ocr import CardOCR

    return CardOCR(s.allow_model_download)


def make_reference(s: Settings):
    from app.pipeline.reference import ReferenceClient

    return ReferenceClient(
        s.reference_cache_dir, s.tcg_api_url, s.tcg_api_key,
        s.tcgdex_api_url if s.enable_tcgdex else None, s.ocr_lang,
        s.reference_timeout, s.reference_cache_ttl_hours,
    )


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=s.data_dir / "raw")
    parser.add_argument("--inferences", type=Path, default=s.inferences_dir)
    parser.add_argument("--no-feedback", action="store_true", help="ignora inferências rotuladas via /feedback")
    parser.add_argument("--include-pending", action="store_true", help="inclui rótulos ainda não revisados (cuidado)")
    parser.add_argument("--no-reference", action="store_true", help="não calcula distâncias à carta oficial (sem OCR/rede)")
    parser.add_argument("--out", type=Path, default=s.data_dir / "features.npz")
    parser.add_argument("--processed", type=Path, default=s.data_dir / "processed")
    args = parser.parse_args(argv)

    samples = list(iter_labeled_images(args.raw))
    if not args.no_feedback:
        feedback = list(iter_feedback_samples(args.inferences, args.include_pending))
        log.info("%d amostras de feedback (%s)", len(feedback), "inclui pendentes" if args.include_pending else "só aprovadas")
        samples += feedback
    if not samples:
        raise SystemExit(f"nenhuma imagem em {args.raw}/real, {args.raw}/fake ou {args.inferences}")
    pairing = pairing_report(samples)
    if pairing["pairing_ratio"] is not None and pairing["pairing_ratio"] < 0.5:
        log.warning(
            "dataset pouco pareado (%s): o classificador pode aprender QUAL carta é em vez de SE é falsa. "
            "Para cada pasta fake/<carta> tenha uma real/<carta>.", pairing,
        )

    embedder = make_embedder(s)
    ocr = reference = None
    if not args.no_reference:
        try:
            ocr, reference = make_ocr(s), make_reference(s)
        except Exception:
            log.exception("OCR/referência indisponíveis; seguindo sem distâncias à carta oficial")
    log.info("%d imagens | embedder %s em %s | referência=%s", len(samples), s.dino_model, embedder.device, reference is not None)

    X, y, groups, paths, ref_dist, not_detected = [], [], [], [], [], []
    n_ref = 0
    for i, sample in enumerate(samples, 1):
        try:
            img = read_image_file(sample.path)
        except ValueError as exc:
            log.warning("pulando %s: %s", sample.path, exc)
            continue
        if sample.rectified:
            card = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)
        else:
            rect = rectify(img)
            if not rect.detected:
                not_detected.append(str(sample.path))  # sem carta detectada só traria ruído
                continue
            card = rect.card

        out_img = args.processed / sample.group / f"{sample.path.stem}.png"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), card)

        feats = embedder.card_features(card)
        dist = np.full(len(REFERENCE_GROUPS), np.nan, dtype=np.float32)
        if ocr is not None and reference is not None:
            parsed = parse_card_text(ocr.read_card(card), s.ocr_lang)
            ref = reference.find_card(parsed.name, parsed.number, parsed.total) if (parsed.name or parsed.number) else None
            ref_img = reference.get_image(ref) if ref is not None else None
            if ref_img is not None:
                ref_feats = embedder.card_features(cv2.resize(ref_img, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC))
                a = select_groups(feats, embedder.groups, REFERENCE_GROUPS)
                b = select_groups(ref_feats, embedder.groups, REFERENCE_GROUPS)
                dist = group_distances(a, b, len(REFERENCE_GROUPS)).astype(np.float32)
                n_ref += 1

        X.append(feats)
        y.append(sample.label)
        groups.append(sample.group)
        paths.append(str(sample.path))
        ref_dist.append(dist)
        if i % 25 == 0:
            log.info("%d/%d", i, len(samples))

    if not X:
        raise SystemExit("nenhuma carta detectada em nenhuma foto")

    np.savez_compressed(
        args.out,
        X=np.stack(X).astype(np.float32),
        y=np.array(y, dtype=np.int64),
        groups=np.array(groups),
        paths=np.array(paths),
        ref_dist=np.stack(ref_dist),
        ref_groups=np.array(json.dumps(list(REFERENCE_GROUPS))),
        meta=np.array(json.dumps(embedder.meta())),
        fingerprint=np.array(dataset_fingerprint(samples)),
        pairing=np.array(json.dumps(pairing)),
    )
    y_arr = np.array(y)
    log.info(
        "salvo %s | reais=%d falsas=%d cartas físicas=%d com referência=%d",
        args.out, int((y_arr == 0).sum()), int((y_arr == 1).sum()), len(set(groups)), n_ref,
    )
    if not_detected:
        log.warning("%d fotos DESCARTADAS por carta não detectada: %s", len(not_detected), not_detected[:10])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
