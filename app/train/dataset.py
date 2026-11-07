"""Leitura do dataset em disco.

Fontes:
  1. ``data/raw/<real|fake>/<id_da_carta>/*.jpg`` — uma subpasta por CARTA
     FÍSICA. O id vira o "grupo": fotos da mesma carta ficam sempre no mesmo
     fold. Fotos soltas direto em real/ ou fake/ viram um grupo cada.
  2. ``data/inferences/<id>/`` com rótulo APROVADO (python -m app.admin review)
     — a carta já está retificada (``rectified=True``).

Dataset pareado: para o classificador não aprender "Charizard = falsa", cada
carta falsa deve ter a mesma carta real no conjunto. ``pairing_report`` mede isso.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.storage import InferenceStore

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = {"real": 0, "fake": 1}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 = real, 1 = falsa
    group: str  # carta física
    rectified: bool = False  # True = imagem já é a carta recortada


def iter_labeled_images(root: Path) -> Iterator[Sample]:
    for label_name, label in LABELS.items():
        base = root / label_name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            rel = path.relative_to(base)
            group = rel.parts[0] if len(rel.parts) > 1 else path.stem
            yield Sample(path, label, f"{label_name}/{group}")


def iter_feedback_samples(inferences_dir: Path, include_pending: bool = False) -> Iterator[Sample]:
    if not inferences_dir.is_dir():
        return
    for image, label, inference_id in InferenceStore(inferences_dir).iter_labeled(include_pending):
        yield Sample(image, label, f"feedback/{inference_id}", rectified=True)


def dataset_fingerprint(samples: Iterable[Sample]) -> str:
    """Hash estável da lista de arquivos (caminho, rótulo, tamanho) — vai no manifesto."""
    digest = hashlib.sha256()
    for s in sorted(samples, key=lambda s: str(s.path)):
        digest.update(f"{s.group}|{s.path.name}|{s.label}|{s.path.stat().st_size}\n".encode())
    return digest.hexdigest()


def pairing_report(samples: Iterable[Sample]) -> dict:
    """Quantas cartas falsas têm uma real 'homônima' (mesmo nome de pasta sem o
    prefixo real/ fake/). Baixo pareamento = risco de aprender identidade."""
    names = {0: set(), 1: set()}
    for s in samples:
        if s.group.startswith(("real/", "fake/")):
            names[s.label].add(s.group.split("/", 1)[1].lower())
    paired = names[1] & names[0]
    return {
        "fake_cards": len(names[1]),
        "paired_fake_cards": len(paired),
        "pairing_ratio": round(len(paired) / len(names[1]), 3) if names[1] else None,
    }
