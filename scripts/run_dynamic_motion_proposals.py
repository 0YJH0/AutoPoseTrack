#!/usr/bin/env python3
"""Evaluate GT-free motion proposals on YCBInEOAT or an extracted HOT3D clip."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from autoposetrack.datasets import load_hot3d_clip, load_ycbineoat
from autoposetrack.evaluation import evaluate_proposals, summarize_proposal_metrics
from autoposetrack.motion import FullFrameMotionProposer, MotionProposalConfig
from autoposetrack.visualization import debug_panel, proposal_overlay


def _config(path: Path) -> MotionProposalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("motion config must be a mapping")
    return MotionProposalConfig.from_mapping(payload)


def _extract_hot3d(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.name}")
        tar.extractall(destination)
    return destination


def _proposal_row(dataset: str, frame_id: int, index: int, proposal) -> dict:
    return {
        "dataset": dataset,
        "frame_id": frame_id,
        "proposal_id": index,
        "tracklet_id": proposal.tracklet_id,
        "bbox_x1": proposal.bbox_xyxy[0], "bbox_y1": proposal.bbox_xyxy[1],
        "bbox_x2": proposal.bbox_xyxy[2], "bbox_y2": proposal.bbox_xyxy[3],
        "area": proposal.area,
        "centroid_x": proposal.centroid[0], "centroid_y": proposal.centroid[1],
        "mean_flow_x": proposal.mean_residual_flow[0],
        "mean_flow_y": proposal.mean_residual_flow[1],
        "max_residual_flow": proposal.max_residual_flow,
        "motion_score": proposal.motion_score,
        "direction_score": proposal.direction_score,
        "compactness_score": proposal.compactness_score,
        "temporal_score": proposal.temporal_score,
        "final_score": proposal.confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ycbineoat", "hot3d"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="sequence directory or HOT3D clip tar")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--start-frame-index", type=int, default=0,
        help="zero-based sequence index; useful for a fixed contiguous motion window",
    )
    parser.add_argument("--debug-every", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    debug_dir = args.output / "debug"
    debug_dir.mkdir()
    source = args.input
    if args.dataset == "hot3d" and source.is_file():
        source = _extract_hot3d(source, args.output / "extracted_clip")
    frames = (
        load_ycbineoat(source)
        if args.dataset == "ycbineoat"
        else load_hot3d_clip(source, args.stream_id)
    )
    if args.start_frame_index < 0 or args.start_frame_index >= len(frames) - 1:
        raise ValueError("start-frame-index must leave at least one adjacent pair")
    frames = frames[args.start_frame_index :]
    proposer = FullFrameMotionProposer(_config(args.config))
    frame_rows: list[dict] = []
    proposal_rows: list[dict] = []
    instance_rows: list[dict] = []
    metrics_05 = []
    writer = None
    processed = 0
    try:
        import cv2
        for previous, current in zip(frames[:-1], frames[1:]):
            if args.max_pairs is not None and processed >= args.max_pairs:
                break
            if current.frame_id != previous.frame_id + 1:
                proposer.reset()
            previous_rgb = np.asarray(Image.open(previous.rgb_path).convert("RGB"))
            current_rgb = np.asarray(Image.open(current.rgb_path).convert("RGB"))
            result = proposer.propose(previous_rgb, current_rgb, current.frame_id)
            frame_rows.append({
                "dataset": args.dataset, "frame_id": current.frame_id,
                "previous_frame_id": previous.frame_id,
                "frame_delta": current.frame_id - previous.frame_id,
                "num_gt_instances": len(current.gt_boxes), **result.debug_info,
            })
            proposal_rows.extend(
                _proposal_row(args.dataset, current.frame_id, index, proposal)
                for index, proposal in enumerate(result.proposals)
            )
            for object_id, gt_box in zip(current.object_ids, current.gt_boxes):
                at03 = evaluate_proposals(result.proposals, gt_box, 0.3)
                at05 = evaluate_proposals(result.proposals, gt_box, 0.5)
                metrics_05.append(at05)
                instance_rows.append({
                    "frame_id": current.frame_id, "object_id": object_id,
                    "gt_bbox_x1": gt_box[0], "gt_bbox_y1": gt_box[1],
                    "gt_bbox_x2": gt_box[2], "gt_bbox_y2": gt_box[3],
                    "max_iou": at05.max_iou, "center_recalled": at05.center_recalled,
                    **{f"recall_at_{k}_iou_03": at03.recall_at_k[k] for k in (1, 3, 5, 10)},
                    **{f"recall_at_{k}_iou_05": at05.recall_at_k[k] for k in (1, 3, 5, 10)},
                })
            overlay = proposal_overlay(current_rgb, result)
            if writer is None:
                height, width = overlay.shape[:2]
                writer = cv2.VideoWriter(str(args.output / "motion_proposals.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if args.debug_every > 0 and processed % args.debug_every == 0:
                gt_box = current.gt_boxes[0] if current.gt_boxes else None
                Image.fromarray(debug_panel(current_rgb, result, gt_box)).save(debug_dir / f"frame_{current.frame_id:06d}.jpg")
            processed += 1
    finally:
        if writer is not None:
            writer.release()
    if not frame_rows:
        raise ValueError("sequence has fewer than two usable frames")
    for filename, rows in (("frames.csv", frame_rows), ("proposals.csv", proposal_rows), ("per_instance_metrics.csv", instance_rows)):
        if not rows:
            continue
        with (args.output / filename).open("x", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            csv_writer.writeheader(); csv_writer.writerows(rows)
    summary = summarize_proposal_metrics(metrics_05) if metrics_05 else {"num_frames": 0}
    summary.update({
        "dataset": args.dataset, "sequence": args.input.name,
        "start_frame_index": args.start_frame_index,
        "num_frame_pairs": len(frame_rows), "num_instances": len(instance_rows),
        "dense_backend": frame_rows[0]["dense_backend"],
        "iou_03": {f"recall_at_{k}": float(np.mean([row[f"recall_at_{k}_iou_03"] for row in instance_rows])) for k in (1, 3, 5, 10)} if instance_rows else {},
        "gt_usage": "evaluator_and_debug_visualization_only_after_proposal_generation",
    })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
