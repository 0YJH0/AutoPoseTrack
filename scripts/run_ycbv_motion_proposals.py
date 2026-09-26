#!/usr/bin/env python3
"""Run motion proposals on strictly adjacent BOP-YCB-V public dataset frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from autoposetrack.evaluation import evaluate_proposals, summarize_proposal_metrics
from autoposetrack.motion import FullFrameMotionProposer, MotionProposalConfig
from autoposetrack.visualization import debug_panel, proposal_overlay


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb(scene_dir: Path, frame_id: int) -> np.ndarray:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = scene_dir / "rgb" / f"{frame_id:06d}{suffix}"
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    raise FileNotFoundError(f"RGB frame {frame_id} missing in {scene_dir}")


def _bbox_xyxy(info):
    x, y, width, height = map(float, info["bbox_visib"])
    return (x, y, x + width, y + height)


def _load_config(path: Path) -> MotionProposalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("motion config must be a mapping")
    return MotionProposalConfig.from_mapping(payload)


def _proposal_row(scene_id, frame_id, index, proposal):
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "proposal_id": index,
        "tracklet_id": proposal.tracklet_id,
        "bbox_x1": proposal.bbox_xyxy[0],
        "bbox_y1": proposal.bbox_xyxy[1],
        "bbox_x2": proposal.bbox_xyxy[2],
        "bbox_y2": proposal.bbox_xyxy[3],
        "area": proposal.area,
        "motion_score": proposal.motion_score,
        "direction_score": proposal.direction_score,
        "temporal_score": proposal.temporal_score,
        "final_score": proposal.confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--debug-every", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    debug_dir = args.output / "debug"
    debug_dir.mkdir()
    config = _load_config(args.config)
    proposer = FullFrameMotionProposer(config)
    split_dir = args.dataset_root / "test"
    scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if args.scene:
        wanted = set(args.scene)
        scene_dirs = [path for path in scene_dirs if int(path.name) in wanted]
    frame_rows = []
    proposal_rows = []
    instance_rows = []
    frame_metrics = []
    writer = None
    processed = 0
    last_scene = None
    last_current = None
    try:
        import cv2

        for scene_dir in scene_dirs:
            scene_id = int(scene_dir.name)
            scene_gt = _json(scene_dir / "scene_gt.json")
            scene_info = _json(scene_dir / "scene_gt_info.json")
            frame_ids = sorted(map(int, scene_gt))
            pairs = [
                (previous, current)
                for previous, current in zip(frame_ids[:-1], frame_ids[1:])
                if current - previous == 1
            ]
            for previous_id, current_id in pairs:
                if args.max_pairs is not None and processed >= args.max_pairs:
                    break
                if scene_id != last_scene or previous_id != last_current:
                    proposer.reset()
                previous = _rgb(scene_dir, previous_id)
                current = _rgb(scene_dir, current_id)
                result = proposer.propose(previous, current, current_id)
                frame_rows.append(
                    {
                        "scene_id": scene_id,
                        "previous_frame_id": previous_id,
                        "frame_id": current_id,
                        "frame_delta": current_id - previous_id,
                        **result.debug_info,
                    }
                )
                proposal_rows.extend(
                    _proposal_row(scene_id, current_id, index, proposal)
                    for index, proposal in enumerate(result.proposals)
                )
                gt_items = scene_gt[str(current_id)]
                info_items = scene_info[str(current_id)]
                for gt_id, (gt, info) in enumerate(zip(gt_items, info_items)):
                    bbox = _bbox_xyxy(info)
                    metrics_03 = evaluate_proposals(
                        result.proposals, bbox, iou_threshold=0.3
                    )
                    metrics_05 = evaluate_proposals(
                        result.proposals, bbox, iou_threshold=0.5
                    )
                    frame_metrics.append(metrics_05)
                    instance_rows.append(
                        {
                            "scene_id": scene_id,
                            "frame_id": current_id,
                            "gt_id": gt_id,
                            "object_id": int(gt["obj_id"]),
                            "visibility_fraction": float(info["visib_fract"]),
                            "bbox_area": float(info["bbox_visib"][2])
                            * float(info["bbox_visib"][3]),
                            "max_iou": metrics_05.max_iou,
                            "center_recalled": metrics_05.center_recalled,
                            "recall_at_1_iou_03": metrics_03.recall_at_k[1],
                            "recall_at_3_iou_03": metrics_03.recall_at_k[3],
                            "recall_at_5_iou_03": metrics_03.recall_at_k[5],
                            "recall_at_10_iou_03": metrics_03.recall_at_k[10],
                            "recall_at_1_iou_05": metrics_05.recall_at_k[1],
                            "recall_at_3_iou_05": metrics_05.recall_at_k[3],
                            "recall_at_5_iou_05": metrics_05.recall_at_k[5],
                            "recall_at_10_iou_05": metrics_05.recall_at_k[10],
                        }
                    )
                overlay = proposal_overlay(current, result)
                if writer is None:
                    height, width = overlay.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output / "motion_proposals.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10.0,
                        (width, height),
                    )
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if args.debug_every > 0 and processed % args.debug_every == 0:
                    evaluator_bbox = _bbox_xyxy(info_items[0]) if info_items else None
                    Image.fromarray(debug_panel(current, result, evaluator_bbox)).save(
                        debug_dir
                        / f"scene_{scene_id:06d}_frame_{current_id:06d}.jpg"
                    )
                last_scene, last_current = scene_id, current_id
                processed += 1
            if args.max_pairs is not None and processed >= args.max_pairs:
                break
    finally:
        if writer is not None:
            writer.release()
    if not frame_rows:
        raise ValueError("no strictly adjacent frame pairs found")
    for name, rows in (
        ("frames.csv", frame_rows),
        ("proposals.csv", proposal_rows),
        ("per_instance_metrics.csv", instance_rows),
    ):
        with (args.output / name).open("x", newline="", encoding="utf-8") as file:
            writer_csv = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(rows)
    summary = summarize_proposal_metrics(frame_metrics)
    summary.pop("num_frames", None)
    summary.update(
        {
            "dataset": "BOP-YCB-V",
            "protocol": "strictly_adjacent_frame_pairs_only",
            "num_frame_pairs": len(frame_rows),
            "num_instances": len(instance_rows),
            "scenes": sorted({row["scene_id"] for row in frame_rows}),
            "frame_delta_distribution": dict(
                Counter(row["frame_delta"] for row in frame_rows)
            ),
            "iou_03": {
                f"recall_at_{key}": float(
                    np.mean([row[f"recall_at_{key}_iou_03"] for row in instance_rows])
                )
                for key in (1, 3, 5, 10)
            },
            "dense_backend": frame_rows[0]["dense_backend"],
            "gt_usage": "evaluator_and_visualization_only_after_proposal_generation",
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
