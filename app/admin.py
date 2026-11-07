"""Tarefas de operador (rodar no host, não expostas pela API).

    python -m app.admin review                 lista rótulos pendentes
    python -m app.admin review --approve ID    aprova (entra no treino)
    python -m app.admin review --reject ID
    python -m app.admin review --approve-all   aprova todos os pendentes
    python -m app.admin cleanup                aplica retenção/limite de disco
    python -m app.admin runs                   lista runs e o atual
"""
from __future__ import annotations

import argparse
import json

from app.config import get_settings
from app.pipeline.registry import list_runs, resolve_latest
from app.pipeline.storage import InferenceStore


def cmd_review(args, store: InferenceStore) -> None:
    if args.approve or args.reject:
        for inference_id in args.approve:
            print(json.dumps({"id": inference_id, **store.review(inference_id, "approved", args.reviewer)}))
        for inference_id in args.reject:
            print(json.dumps({"id": inference_id, **store.review(inference_id, "rejected", args.reviewer)}))
        return
    pending = list(store.iter_labels(("pending",)))
    if args.approve_all:
        for folder, _ in pending:
            store.review(folder.name, "approved", args.reviewer)
        print(f"{len(pending)} rótulos aprovados")
        return
    if not pending:
        print("nenhum rótulo pendente")
    for folder, payload in pending:
        result = json.loads((folder / "result.json").read_text())
        print(
            f"{folder.name}  rótulo={payload['label']:<4}  veredito={result.get('verdict'):<12} "
            f"p={result.get('fake_probability')}  ocr={result.get('ocr', {}).get('name')}  nota={payload.get('note') or '-'}"
        )
    print(f"\n{len(pending)} pendentes — aprove com: python -m app.admin review --approve <id>")


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("--approve", nargs="*", default=[], metavar="ID")
    rv.add_argument("--reject", nargs="*", default=[], metavar="ID")
    rv.add_argument("--approve-all", action="store_true")
    rv.add_argument("--reviewer", default="operator")
    sub.add_parser("cleanup")
    sub.add_parser("runs")
    args = parser.parse_args()

    if args.cmd == "runs":
        latest = resolve_latest(s.models_dir)
        for run in list_runs(s.models_dir):
            print(("* " if run == latest else "  ") + run.name)
        if latest is None:
            print("(nenhum run promovido)")
        return

    store = InferenceStore(s.inferences_dir)
    if args.cmd == "review":
        cmd_review(args, store)
    elif args.cmd == "cleanup":
        print(json.dumps(store.cleanup(s.retention_days, s.max_inferences_gb)))
        print(json.dumps(store.stats()))


if __name__ == "__main__":
    main()
