#!/usr/bin/env python3
"""Run controlled SE(3) trials to measure the local tracker's recovery basin."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

from autoposetrack.datasets import YCBVSingleObjectDataset
from autoposetrack.evaluation import load_ascii_ply_vertices_m
from autoposetrack.pose.global_estimator import MegaPoseRGBAdapter
from autoposetrack.rollout import RecoveryBasinGenerator, make_perturbation_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=5)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--anchor-stride", type=int, default=10)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--rotation-deg", type=float, action="append")
    parser.add_argument("--translation-m", type=float, action="append")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--refiner-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anchor_stride <= 0:
        raise ValueError("anchor_stride must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = YCBVSingleObjectDataset(args.dataset_root, args.object_id)
    model = dataset.model_reference(args.object_id)
    adapter = MegaPoseRGBAdapter(
        model, "megapose-1.0-RGB", 1, 64, args.refiner_iterations
    )
    generator = RecoveryBasinGenerator(
        dataset,
        adapter,
        model,
        load_ascii_ply_vertices_m(model.mesh_path),
        dataset.is_symmetric(),
        args.future_frames,
        0.10,
    )
    perturbations = make_perturbation_grid(
        args.rotation_deg or [0, 10, 20, 40],
        args.translation_m or [0, 0.01, 0.03, 0.06],
    )
    trials = []
    try:
        for sequence in args.sequence or ["50", "52"]:
            targets = dataset.target_frames([sequence])
            for anchor_index in range(0, len(targets), args.anchor_stride):
                for perturbation in perturbations:
                    trials.append(generator.run(targets, anchor_index, perturbation))
                    if args.max_trials is not None and len(trials) >= args.max_trials:
                        break
                if args.max_trials is not None and len(trials) >= args.max_trials:
                    break
            if args.max_trials is not None and len(trials) >= args.max_trials:
                break
    finally:
        adapter.close()
    del generator
    del adapter
    gc.collect()
    rows = [trial.to_dict() for trial in trials]
    with (output / "recovery_trials.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "trials": len(trials),
        "recovered": sum(trial.recovered for trial in trials),
        "future_frames": args.future_frames,
        "success_fraction_of_diameter": 0.10,
        "perturbation_frame": "object_local_right_composition",
        "modality": "rgb",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
