"""Registro de modelos versionado por run.

    models/
      runs/<AAAAMMDDTHHMMSSZ>/   classifier.json, anomaly.npz, fusion.json,
                                 metrics.json, manifest.json
      latest.json                {"run": "<nome>", "promoted_at": "..."}

A API carrega o run apontado por latest.json. Voltar atrás = editar o ponteiro.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

RUNS_DIR = "runs"
LATEST_FILE = "latest.json"
_RUN_NAME = re.compile(r"^[0-9]{8}T[0-9]{6}Z(-[a-z0-9]+)?$")


def new_run_dir(models_dir: Path, suffix: str | None = None) -> Path:
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if suffix:
        name += f"-{re.sub(r'[^a-z0-9]', '', suffix.lower())[:16]}"
    run = models_dir / RUNS_DIR / name
    run.mkdir(parents=True, exist_ok=False)
    return run


def promote(models_dir: Path, run_dir: Path) -> None:
    (models_dir / LATEST_FILE).write_text(
        json.dumps({"run": run_dir.name, "promoted_at": datetime.now(UTC).isoformat()})
    )


def resolve_latest(models_dir: Path) -> Path | None:
    pointer = models_dir / LATEST_FILE
    if not pointer.exists():
        return None
    try:
        name = json.loads(pointer.read_text())["run"]
    except (ValueError, KeyError, TypeError):
        return None
    if not _RUN_NAME.match(str(name)):
        return None
    run = models_dir / RUNS_DIR / name
    return run if run.is_dir() else None


def list_runs(models_dir: Path) -> list[Path]:
    base = models_dir / RUNS_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and _RUN_NAME.match(p.name))
