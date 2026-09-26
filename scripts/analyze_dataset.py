#!/usr/bin/env python3
"""Audit natural attributes in a classic BOP dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoposetrack.analysis import AuditConfig, audit_bop_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--visual-samples", type=int, default=24)
    parser.add_argument("--reference-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_bop_dataset(
        AuditConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output,
            split=args.split,
            seed=args.seed,
            visual_samples=args.visual_samples,
            reference_manifest=args.reference_manifest,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
