"""Connected components, interpretable scoring, expansion, and merging."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .types import BBox, MotionProposal, ProposalConfig


def bbox_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def expand_bbox(bbox: BBox, scale: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = (x2 - x1) * scale / 2.0
    half_height = (y2 - y1) * scale / 2.0
    return (
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(float(width), center_x + half_width),
        min(float(height), center_y + half_height),
    )


def direction_consistency(vectors: npt.NDArray[np.floating]) -> float:
    magnitude = np.linalg.norm(vectors, axis=1)
    valid = magnitude > 1e-6
    if not np.any(valid):
        return 0.0
    normalized = vectors[valid] / magnitude[valid, None]
    return float(np.clip(np.linalg.norm(np.mean(normalized, axis=0)), 0.0, 1.0))


def _confidence(proposal: MotionProposal, config: ProposalConfig) -> float:
    weights = np.asarray(
        [
            config.weight_motion,
            config.weight_direction,
            config.weight_temporal,
            config.weight_compactness,
        ],
        dtype=float,
    )
    values = np.asarray(
        [
            proposal.motion_score,
            proposal.direction_score,
            proposal.temporal_score,
            proposal.compactness_score,
        ],
        dtype=float,
    )
    return float(np.dot(weights, values) / np.sum(weights))


def extract_components(
    mask: npt.NDArray[np.bool_],
    residual: npt.NDArray[np.floating],
    threshold: float,
    config: ProposalConfig,
) -> List[MotionProposal]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "component extraction requires the motion dependency"
        ) from error
    height, width = mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    magnitude = np.linalg.norm(residual, axis=2)
    min_area = max(
        config.min_area_pixels, int(config.min_area_ratio * height * width)
    )
    proposals: List[MotionProposal] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component = labels == component_id
        component_magnitude = magnitude[component]
        vectors = residual[component]
        mean_magnitude = float(np.mean(component_magnitude))
        keep_small = (
            config.preserve_small_strong
            and area > 0
            and mean_magnitude >= threshold * config.small_strong_factor
        )
        if area < min_area and not keep_small:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        raw_bbox = (float(x), float(y), float(x + box_width), float(y + box_height))
        bbox = expand_bbox(raw_bbox, config.bbox_expand_scale, width, height)
        compactness = area / float(max(box_width * box_height, 1))
        motion_score = 1.0 - math.exp(
            -mean_magnitude / max(float(threshold), 1e-6)
        )
        proposal = MotionProposal(
            bbox_xyxy=bbox,
            motion_score=float(np.clip(motion_score, 0.0, 1.0)),
            direction_score=direction_consistency(vectors),
            compactness_score=float(np.clip(compactness, 0.0, 1.0)),
            temporal_score=0.0,
            confidence=0.0,
            area=area,
            centroid=(
                float(centroids[component_id, 0]),
                float(centroids[component_id, 1]),
            ),
            mean_residual_flow=(
                float(np.mean(vectors[:, 0])),
                float(np.mean(vectors[:, 1])),
            ),
            max_residual_flow=float(np.max(component_magnitude)),
        )
        proposals.append(replace(proposal, confidence=_confidence(proposal, config)))
    return proposals


def _direction_cosine(first: MotionProposal, second: MotionProposal) -> float:
    a = np.asarray(first.mean_residual_flow, dtype=float)
    b = np.asarray(second.mean_residual_flow, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-8 else 0.0


def _should_merge(
    first: MotionProposal,
    second: MotionProposal,
    config: ProposalConfig,
    diagonal: float,
) -> bool:
    distance = float(
        np.linalg.norm(np.asarray(first.centroid) - np.asarray(second.centroid))
    )
    spatial = bbox_iou(first.bbox_xyxy, second.bbox_xyxy) >= config.merge_iou
    spatial |= distance <= config.merge_centroid_distance_ratio * diagonal
    return spatial and _direction_cosine(first, second) >= config.merge_direction_cosine


def _merge_group(
    group: Sequence[MotionProposal], config: ProposalConfig
) -> MotionProposal:
    areas = np.asarray([proposal.area for proposal in group], dtype=float)
    weights = areas / max(float(np.sum(areas)), 1.0)
    boxes = np.asarray([proposal.bbox_xyxy for proposal in group], dtype=float)
    centroid = np.sum(
        np.asarray([proposal.centroid for proposal in group]) * weights[:, None], axis=0
    )
    flow = np.sum(
        np.asarray([proposal.mean_residual_flow for proposal in group])
        * weights[:, None],
        axis=0,
    )
    proposal = MotionProposal(
        bbox_xyxy=(
            float(np.min(boxes[:, 0])),
            float(np.min(boxes[:, 1])),
            float(np.max(boxes[:, 2])),
            float(np.max(boxes[:, 3])),
        ),
        motion_score=float(
            np.sum(np.asarray([p.motion_score for p in group]) * weights)
        ),
        direction_score=float(
            np.sum(np.asarray([p.direction_score for p in group]) * weights)
        ),
        compactness_score=float(
            np.sum(np.asarray([p.compactness_score for p in group]) * weights)
        ),
        temporal_score=max(proposal.temporal_score for proposal in group),
        confidence=0.0,
        area=int(np.sum(areas)),
        centroid=(float(centroid[0]), float(centroid[1])),
        mean_residual_flow=(float(flow[0]), float(flow[1])),
        max_residual_flow=max(proposal.max_residual_flow for proposal in group),
    )
    return replace(proposal, confidence=_confidence(proposal, config))


def merge_proposals(
    proposals: Sequence[MotionProposal],
    config: ProposalConfig,
    image_shape: Tuple[int, int],
) -> List[MotionProposal]:
    if not proposals:
        return []
    diagonal = math.hypot(*image_shape)
    groups: List[List[MotionProposal]] = []
    for proposal in proposals:
        for group in groups:
            if any(_should_merge(proposal, item, config, diagonal) for item in group):
                group.append(proposal)
                break
        else:
            groups.append([proposal])
    return [_merge_group(group, config) for group in groups]


def rerank(
    proposals: Sequence[MotionProposal], config: ProposalConfig
) -> List[MotionProposal]:
    scored = [
        replace(item, confidence=_confidence(item, config)) for item in proposals
    ]
    ordered = sorted(scored, key=lambda item: item.confidence, reverse=True)
    return ordered[: config.top_k]
