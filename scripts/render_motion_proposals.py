#!/usr/bin/env python3
"""Render full-frame motion proposals for a video or ordered image directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import yaml
from PIL import Image

from autoposetrack.motion import FullFrameMotionProposer, MotionProposalConfig
from autoposetrack.visualization import debug_panel, proposal_overlay


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install AutoPoseTrack with the 'motion' optional dependency"
        ) from error
    return cv2


def _frames(path: Path) -> Tuple[Iterator[Tuple[int, np.ndarray]], float]:
    cv2 = _cv2()
    if path.is_dir():
        files = sorted(
            item
            for item in path.iterdir()
            if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        def image_iterator() -> Iterator[Tuple[int, np.ndarray]]:
            for index, image_path in enumerate(files):
                yield index, np.asarray(Image.open(image_path).convert("RGB"))

        return image_iterator(), 30.0
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

    def video_iterator() -> Iterator[Tuple[int, np.ndarray]]:
        index = 0
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            capture.release()

    return video_iterator(), fps


def _load_config(path: Path) -> MotionProposalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("motion proposal config must be a mapping")
    return MotionProposalConfig.from_mapping(payload)


def _proposal_row(frame_id: int, index: int, proposal) -> dict:
    return {
        "frame_id": frame_id,
        "proposal_id": index,
        "tracklet_id": proposal.tracklet_id,
        "bbox_x1": proposal.bbox_xyxy[0],
        "bbox_y1": proposal.bbox_xyxy[1],
        "bbox_x2": proposal.bbox_xyxy[2],
        "bbox_y2": proposal.bbox_xyxy[3],
        "area": proposal.area,
        "centroid_x": proposal.centroid[0],
        "centroid_y": proposal.centroid[1],
        "mean_flow_x": proposal.mean_residual_flow[0],
        "mean_flow_y": proposal.mean_residual_flow[1],
        "max_residual_flow": proposal.max_residual_flow,
        "motion_score": proposal.motion_score,
        "direction_score": proposal.direction_score,
        "temporal_score": proposal.temporal_score,
        "compactness_score": proposal.compactness_score,
        "final_score": proposal.confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--debug-every", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    debug_dir = args.output / "debug"
    debug_dir.mkdir()
    proposer = FullFrameMotionProposer(_load_config(args.config))
    iterator, fps = _frames(args.input)
    previous = None
    writer = None
    frame_rows = []
    proposal_rows = []
    processed = 0
    cv2 = _cv2()
    try:
        for frame_id, rgb in iterator:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            if previous is None:
                previous = rgb
                continue
            result = proposer.propose(previous, rgb, frame_id)
            overlay = proposal_overlay(rgb, result)
            mask_rgb = np.repeat(
                (result.motion_mask[..., None] * 255).astype(np.uint8), 3, axis=2
            )
            video_frame = np.concatenate((overlay, mask_rgb), axis=1)
            if writer is None:
                height, width = video_frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output / "motion_proposals.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("failed to open output video writer")
            writer.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
            if args.debug_every > 0 and processed % args.debug_every == 0:
                Image.fromarray(debug_panel(rgb, result)).save(
                    debug_dir / f"frame_{frame_id:06d}.jpg"
                )
            frame_rows.append({"frame_id": frame_id, **result.debug_info})
            proposal_rows.extend(
                _proposal_row(frame_id, index, proposal)
                for index, proposal in enumerate(result.proposals)
            )
            previous = rgb
            processed += 1
    finally:
        if writer is not None:
            writer.release()
    if not frame_rows:
        raise ValueError("input must contain at least two readable RGB frames")
    with (args.output / "frames.csv").open("x", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)
    proposal_fields = [
        "frame_id", "proposal_id", "tracklet_id", "bbox_x1", "bbox_y1",
        "bbox_x2", "bbox_y2", "area", "centroid_x", "centroid_y",
        "mean_flow_x", "mean_flow_y", "max_residual_flow", "motion_score",
        "direction_score", "temporal_score", "compactness_score", "final_score",
    ]
    with (args.output / "proposals.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer_csv = csv.DictWriter(file, fieldnames=proposal_fields)
        writer_csv.writeheader()
        writer_csv.writerows(proposal_rows)
    summary = {
        "num_frame_pairs": len(frame_rows),
        "num_proposals": len(proposal_rows),
        "average_proposals_per_frame": len(proposal_rows) / len(frame_rows),
        "input": str(args.input),
        "config": str(args.config),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
