"""Persistência das inferências e dos rótulos humanos.

    data/inferences/<id>/original.<jpg|png|webp>  upload como chegou
                          card.jpg                carta retificada
                          result.json             sinais, veredito, OCR, quad
                          label.json              {"label", "status": "pending"|"approved"|"rejected", ...}
    data/inferences/index/<sha256>                id (dedup de uploads idênticos)

Rótulos nascem ``pending``: um operador aprova com ``python -m app.admin review``
antes de entrarem no treino. Isso fecha a porta a envenenamento via /feedback.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

LABELS = {"real": 0, "fake": 1}
STATUSES = ("pending", "approved", "rejected")
ID_RE = re.compile(r"^[0-9a-f]{32}$")
_DROP_KEYS = {"preview"}
_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


class InferenceStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(exist_ok=True)

    # ----------------------------------------------------------------- save
    def find_duplicate(self, sha256: str) -> str | None:
        pointer = self.root / "index" / sha256
        if pointer.exists():
            inference_id = pointer.read_text().strip()
            if ID_RE.match(inference_id) and (self.root / inference_id / "result.json").exists():
                return inference_id
        return None

    def save(
        self,
        card_bgr: np.ndarray,
        result: dict,
        original: bytes | None = None,
        original_format: str | None = None,
        sha256: str | None = None,
    ) -> str:
        inference_id = uuid.uuid4().hex
        folder = self.root / inference_id
        folder.mkdir()
        cv2.imwrite(str(folder / "card.jpg"), card_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if original is not None:
            (folder / f"original.{_EXT.get(original_format or '', 'bin')}").write_bytes(original)
        payload = {k: v for k, v in result.items() if k not in _DROP_KEYS}
        payload["saved_at"] = datetime.now(UTC).isoformat()
        payload["sha256"] = sha256
        (folder / "result.json").write_text(json.dumps(payload, ensure_ascii=False))
        if sha256:
            (self.root / "index" / sha256).write_text(inference_id)
        return inference_id

    # --------------------------------------------------------------- labels
    def set_label(self, inference_id: str, label: str, note: str | None = None, source: str | None = None) -> dict:
        if not ID_RE.match(inference_id):
            raise ValueError("id inválido")
        if label not in LABELS:
            raise ValueError("rótulo deve ser 'real' ou 'fake'")
        folder = self.root / inference_id
        if not (folder / "result.json").exists():
            raise KeyError(inference_id)
        payload = {
            "label": label,
            "status": "pending",
            "note": note,
            "source": source,
            "labeled_at": datetime.now(UTC).isoformat(),
        }
        (folder / "label.json").write_text(json.dumps(payload, ensure_ascii=False))
        return payload

    def review(self, inference_id: str, status: str, reviewer: str | None = None) -> dict:
        if status not in STATUSES:
            raise ValueError(f"status deve ser um de {STATUSES}")
        label_file = self.root / inference_id / "label.json"
        if not (ID_RE.match(inference_id) and label_file.exists()):
            raise KeyError(inference_id)
        payload = json.loads(label_file.read_text())
        payload.update({"status": status, "reviewed_by": reviewer, "reviewed_at": datetime.now(UTC).isoformat()})
        label_file.write_text(json.dumps(payload, ensure_ascii=False))
        return payload

    def _folders(self) -> Iterator[Path]:
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir() and ID_RE.match(folder.name):
                yield folder

    def iter_labels(self, statuses=("approved",)) -> Iterator[tuple[Path, dict]]:
        for folder in self._folders():
            label_file = folder / "label.json"
            if not label_file.exists():
                continue
            try:
                payload = json.loads(label_file.read_text())
            except ValueError:
                continue
            if payload.get("label") in LABELS and payload.get("status", "pending") in statuses:
                yield folder, payload

    def iter_labeled(self, include_pending: bool = False) -> Iterator[tuple[Path, int, str]]:
        """Gera (card.jpg, rótulo 0/1, id) das inferências aprovadas (ou pendentes)."""
        statuses = ("approved", "pending") if include_pending else ("approved",)
        for folder, payload in self.iter_labels(statuses):
            if (folder / "card.jpg").exists():
                yield folder / "card.jpg", LABELS[payload["label"]], folder.name

    # -------------------------------------------------------------- cleanup
    def cleanup(self, retention_days: int, max_gb: float) -> dict:
        """Apaga inferências SEM rótulo: mais antigas que retention_days e, se
        ainda acima de max_gb, as mais antigas até caber."""
        now = time.time()
        removed = 0
        entries = []
        for folder in self._folders():
            size = sum(p.stat().st_size for p in folder.iterdir())
            labeled = (folder / "label.json").exists()
            entries.append((folder.stat().st_mtime, folder, size, labeled))
        entries.sort()  # mais antigas primeiro

        def remove(folder: Path) -> None:
            nonlocal removed
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1

        total = sum(e[2] for e in entries)
        kept = []
        for mtime, folder, size, labeled in entries:
            if not labeled and now - mtime > retention_days * 86400:
                remove(folder)
                total -= size
            else:
                kept.append((mtime, folder, size, labeled))
        for mtime, folder, size, labeled in kept:
            if total <= max_gb * 1024**3:
                break
            if not labeled:
                remove(folder)
                total -= size
        # ponteiros de dedup órfãos
        for pointer in (self.root / "index").iterdir():
            target = pointer.read_text().strip() if pointer.is_file() else ""
            if not (self.root / target / "result.json").exists():
                pointer.unlink(missing_ok=True)
        return {"removed": removed, "remaining_gb": round(total / 1024**3, 3)}

    def stats(self) -> dict:
        total = labeled = pending = approved = 0
        for folder in self._folders():
            total += 1
            label_file = folder / "label.json"
            if label_file.exists():
                labeled += 1
                try:
                    status = json.loads(label_file.read_text()).get("status", "pending")
                except ValueError:
                    continue
                pending += status == "pending"
                approved += status == "approved"
        return {"total": total, "labeled": labeled, "pending": pending, "approved": approved}
