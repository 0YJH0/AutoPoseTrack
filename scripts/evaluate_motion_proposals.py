#!/usr/bin/env python3
"""Evaluate saved motion proposals against GT boxes without feeding GT upstream."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from autoposetrack.evaluation import evaluate_proposals, summarize_proposal_metrics
from autoposetrack.motion.types import MotionProposal


def _proposal(row: dict[str, str]) -> MotionProposal:
    bbox_fields = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    return MotionProposal(
        bbox_xyxy=tuple(float(row[name]) for name in bbox_fields),
        motion_score=float(row["motion_score"]),
        direction_score=float(row["direction_score"]),
        compactness_score=float(row["compactness_score"]),
        temporal_score=float(row["temporal_score"]),
        confidence=float(row["final_score"]),
        area=int(row["area"]),
        centroid=(float(row["centroid_x"]), float(row["centroid_y"])),
        mean_residual_flow=(float(row["mean_flow_x"]), float(row["mean_flow_y"])),
        max_residual_flow=float(row["max_residual_flow"]),
        tracklet_id=int(row["tracklet_id"]) if row["tracklet_id"] else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--gt-bboxes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    proposals = defaultdict(list)
    with args.proposals.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            proposals[int(row["frame_id"])].append(_proposal(row))
    with args.frames.open(newline="", encoding="utf-8") as file:
        frame_ids = {int(row["frame_id"]) for row in csv.DictReader(file)}
    gt = {}
    with args.gt_bboxes.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame_id = int(row["frame_id"])
            bbox_fields = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            gt[frame_id] = tuple(float(row[name]) for name in bbox_fields)
    evaluated = []
    rows = []
    for frame_id in sorted(frame_ids & set(gt)):
        metrics = evaluate_proposals(
            proposals[frame_id], gt[frame_id], args.iou_threshold
        )
        evaluated.append(metrics)
        rows.append(
            {
                "frame_id": frame_id,
                "max_iou": metrics.max_iou,
                "center_recalled": metrics.center_recalled,
                "num_proposals": metrics.num_proposals,
                **{
                    f"recall_at_{key}": value
                    for key, value in metrics.recall_at_k.items()
                },
            }
        )
    if not evaluated:
        raise ValueError("no common evaluated frame IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_frame_proposal_metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_proposal_metrics(evaluated)
    summary["iou_threshold"] = args.iou_threshold
    summary["gt_usage"] = "evaluator_only"
    (args.output / "proposal_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
