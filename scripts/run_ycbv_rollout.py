#!/usr/bin/env python3
"""Generate continuous MegaPose RGB tracking rollouts on one YCB-V object."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

from autoposetrack.datasets import YCBVSingleObjectDataset
from autoposetrack.evaluation import load_ascii_ply_vertices_m
from autoposetrack.pose.global_estimator import MegaPoseRGBAdapter
from autoposetrack.rollout import RolloutGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=5)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--model-name", default="megapose-1.0-RGB")
    parser.add_argument("--renderer-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--refiner-iterations", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = YCBVSingleObjectDataset(args.dataset_root, args.object_id)
    model = dataset.model_reference(args.object_id)
    adapter = MegaPoseRGBAdapter(
        model, args.model_name, args.renderer_workers, args.batch_size,
        args.refiner_iterations,
    )
    generator = RolloutGenerator(
        dataset, adapter, adapter, model,
        load_ascii_ply_vertices_m(model.mesh_path), dataset.is_symmetric(),
    )
    sequences = args.sequence or ["50", "52"]
    targets = dataset.target_frames(sequences)
    if args.max_frames_per_sequence is not None:
        if args.max_frames_per_sequence <= 0:
            raise ValueError("max_frames_per_sequence must be positive")
        targets = tuple(
            target
            for sequence in sequences
            for target in dataset.target_frames([sequence])[: args.max_frames_per_sequence]
        )
    try:
        records = generator.run(targets)
    finally:
        adapter.close()
    del generator
    del adapter
    gc.collect()
    rows = [record.to_dict() for record in records]
    with (output / "rollout.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "object_id": args.object_id,
        "sequences": sorted({record.sequence_id for record in records}),
        "frames": len(records),
        "successful_predictions": sum(record.status == "ok" for record in records),
        "model_name": args.model_name,
        "refiner_iterations": args.refiner_iterations,
        "detection_source": "oracle_bbox_obj",
        "modality": "rgb",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
