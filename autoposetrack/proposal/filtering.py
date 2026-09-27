"""High-recall filtering and duplicate suppression for segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import SegmentationProposal


@dataclass(frozen=True)
class ProposalFilterConfig:
    min_area: int = 100
    max_area_ratio: float = 0.8
    min_stability: float = 0.0
    duplicate_iou: float = 0.9
    max_proposals: int = 100


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def filter_proposals(proposals: list[SegmentationProposal], shape: tuple[int, int], config: ProposalFilterConfig) -> list[SegmentationProposal]:
    max_area = shape[0] * shape[1] * config.max_area_ratio
    candidates = [item for item in proposals if config.min_area <= item.area <= max_area and item.stability_score >= config.min_stability]
    candidates.sort(key=lambda item: (item.stability_score, item.segmentation_score, item.area), reverse=True)
    kept: list[SegmentationProposal] = []
    for proposal in candidates:
        if any(mask_iou(proposal.mask, other.mask) >= config.duplicate_iou for other in kept):
            continue
        kept.append(proposal)
        if len(kept) >= config.max_proposals:
            break
    return kept
