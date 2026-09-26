"""Debug panels and video frames for full-frame motion proposals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from autoposetrack.motion.residual import flow_magnitude
from autoposetrack.motion.types import BBox, MotionProposalResult


def _heatmap(value: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    finite = value[np.isfinite(value)]
    upper = float(np.percentile(finite, 99.0)) if len(finite) else 1.0
    normalized = np.clip(value / max(upper, 1e-6), 0.0, 1.0)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    return (np.stack((red, green, blue), axis=2) * 255).astype(np.uint8)


def proposal_overlay(
    rgb: npt.NDArray[np.uint8],
    result: MotionProposalResult,
    gt_bbox: Optional[BBox] = None,
) -> npt.NDArray[np.uint8]:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for index, proposal in enumerate(result.proposals):
        draw.rectangle(proposal.bbox_xyxy, outline=(255, 50, 50), width=2)
        label = (
            f"P{index} s={proposal.confidence:.2f} "
            f"m={proposal.motion_score:.2f} t={proposal.temporal_score:.2f}"
        )
        x, y = proposal.bbox_xyxy[:2]
        draw.text((x + 2, y + 2), label, fill=(255, 255, 0))
    if gt_bbox is not None:
        draw.rectangle(gt_bbox, outline=(50, 255, 50), width=2)
        draw.text(
            (gt_bbox[0] + 2, gt_bbox[1] + 2),
            "GT evaluator only",
            fill=(50, 255, 50),
        )
    return np.asarray(image)


def debug_panel(
    rgb: npt.NDArray[np.uint8],
    result: MotionProposalResult,
    gt_bbox: Optional[BBox] = None,
) -> npt.NDArray[np.uint8]:
    raw = _heatmap(flow_magnitude(result.raw_flow))
    background = _heatmap(flow_magnitude(result.background_flow))
    residual = _heatmap(result.motion_score_map)
    mask = np.repeat((result.motion_mask[..., None] * 255).astype(np.uint8), 3, axis=2)
    overlay = proposal_overlay(rgb, result, gt_bbox)
    labels = (
        "RGB",
        "Raw flow magnitude",
        "Background flow",
        "Residual magnitude",
        "Motion mask",
        "Final proposals",
    )
    panels = []
    for panel, label in zip((rgb, raw, background, residual, mask, overlay), labels):
        image = Image.fromarray(panel)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width, 150), 16), fill=(0, 0, 0))
        draw.text((3, 2), label, fill=(255, 255, 255))
        panels.append(np.asarray(image))
    return np.concatenate(
        (np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)),
        axis=0,
    )
