#!/usr/bin/env python3
"""Convert rollout CSV into feature NPZ and explicit scene-level split JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from autoposetrack.reliability import (
    build_recoverability_features,
    build_temporal_recoverability_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diameter-m", type=float, required=True)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--success-fraction", type=float, default=0.10)
    parser.add_argument("--train-sequence", action="append")
    parser.add_argument("--validation-sequence", action="append")
    parser.add_argument(
        "--summary-features",
        action="store_true",
        help="use the legacy summary baseline instead of a causal temporal window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.rollout.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    builder = (
        build_recoverability_features
        if args.summary_features
        else build_temporal_recoverability_features
    )
    dataset = builder(
        rows, args.diameter_m, args.history_frames, args.future_frames,
        args.success_fraction,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset.save(output / "recoverability_features.npz")
    train_ids = set(args.train_sequence or ["000050"])
    validation_ids = set(args.validation_sequence or ["000052"])
    if train_ids & validation_ids:
        raise ValueError("train and validation sequence IDs overlap")
    known = set(dataset.sequence_ids.tolist())
    if train_ids | validation_ids != known:
        raise ValueError(
            f"split must cover all sequences exactly: known={sorted(known)}"
        )
    split = {
        "unit": "complete_ycbv_scene",
        "train_sequences": sorted(train_ids),
        "validation_sequences": sorted(validation_ids),
        "test_sequences": [],
        "train_indices": np.flatnonzero(
            np.isin(dataset.sequence_ids, list(train_ids))
        ).tolist(),
        "validation_indices": np.flatnonzero(
            np.isin(dataset.sequence_ids, list(validation_ids))
        ).tolist(),
        "note": "pilot split only; no independent test set",
    }
    (output / "sequence_split.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
