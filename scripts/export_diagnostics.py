#!/usr/bin/env python3
"""Export a small, Git-trackable report from a local/cloud training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--log-tail-lines", type=int, default=500)
    args = parser.parse_args()
    if args.log_tail_lines <= 0:
        parser.error("--log-tail-lines must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    target = args.reports_root.expanduser().resolve() / run_dir.name
    target.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    for relative in (
        "config.yaml",
        "manifest.json",
        "metrics.json",
        "logs/metrics.jsonl",
        "logs/events.jsonl",
        "logs/failure.json",
    ):
        source = run_dir / relative
        if source.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative)

    train_log = run_dir / "logs" / "train.log"
    if train_log.is_file():
        lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()
        destination = target / "logs" / "train.tail.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(lines[-args.log_tail_lines :]) + "\n", encoding="utf-8"
        )
        copied.append("logs/train.tail.log")

    inventory: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_run_name": run_dir.name,
        "files": {},
        "excluded": ["model.npz", "history.json", "per_frame.csv", "figures", "videos"],
    }
    for relative in copied:
        path = target / relative
        inventory["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    with (target / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(target)


if __name__ == "__main__":
    main()
