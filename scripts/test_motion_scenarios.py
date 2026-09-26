#!/usr/bin/env python3
"""Controlled camera-only and camera-plus-object motion scenario test."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from autoposetrack.evaluation import evaluate_proposals
from autoposetrack.motion import FullFrameMotionProposer, MotionProposalConfig
from autoposetrack.motion.types import DenseProcessingConfig
from autoposetrack.visualization import debug_panel


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("scenario test requires the motion dependency") from error
    return cv2


def _load_config(path: Path, backend: str) -> MotionProposalConfig:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = MotionProposalConfig.from_mapping(payload)
    return replace(
        config,
        dense_processing=DenseProcessingConfig(backend=backend, device="cuda:0"),
    )


def _scene(seed: int, object_extra_x: int):
    cv2 = _cv2()
    height, width = 192, 256
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 180, (height, width, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (3, 3), 0)
    x1, y1, x2, y2 = 82, 68, 126, 116
    texture = rng.integers(40, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
    texture[..., 0] = np.maximum(texture[..., 0], 210)
    previous = background.copy()
    previous[y1:y2, x1:x2] = texture
    camera_dx, camera_dy = 6, 3
    affine = np.asarray([[1.0, 0.0, camera_dx], [0.0, 1.0, camera_dy]])
    current = cv2.warpAffine(
        background,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    current_x1 = x1 + camera_dx + object_extra_x
    current_y1 = y1 + camera_dy
    current_x2 = current_x1 + (x2 - x1)
    current_y2 = current_y1 + (y2 - y1)
    current[current_y1:current_y2, current_x1:current_x2] = texture
    gt_bbox = (
        float(current_x1),
        float(current_y1),
        float(current_x2),
        float(current_y2),
    )
    return previous, current, gt_bbox


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/motion_proposal/default.yaml"),
    )
    parser.add_argument(
        "--dense-backend",
        choices=("numpy_cpu", "torch_cuda"),
        default="numpy_cpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = _load_config(args.config, args.dense_backend)
    scenarios = {
        "camera_moves_target_static": 0,
        "camera_and_target_move": 12,
    }
    summary = {}
    for name, extra_motion in scenarios.items():
        previous, current, gt_bbox = _scene(args.seed, extra_motion)
        proposer = FullFrameMotionProposer(config)
        result = proposer.propose(previous, current, frame_id=1)
        metrics = evaluate_proposals(result.proposals, gt_bbox, iou_threshold=0.3)
        scenario_dir = args.output / name
        scenario_dir.mkdir()
        Image.fromarray(previous).save(scenario_dir / "previous.png")
        Image.fromarray(current).save(scenario_dir / "current.png")
        Image.fromarray(debug_panel(current, result, gt_bbox)).save(
            scenario_dir / "debug_panel.jpg"
        )
        x1, y1, x2, y2 = map(int, gt_bbox)
        target_residual = result.motion_score_map[y1:y2, x1:x2]
        summary[name] = {
            "dense_backend": result.debug_info["dense_backend"],
            "num_proposals": len(result.proposals),
            "max_iou": metrics.max_iou,
            "center_recalled": metrics.center_recalled,
            "recall_at_1_iou_0.3": metrics.recall_at_k[1],
            "target_residual_mean": float(np.mean(target_residual)),
            "global_residual_mean": result.debug_info["residual_flow_mean"],
            "threshold": result.debug_info["threshold"],
            "background_valid": result.debug_info["background_model_valid"],
            "runtime_total_s": result.debug_info["runtime_total_s"],
        }
    (args.output / "scenario_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
