"""Metrics that separate proposal coverage from reference matching errors."""

from __future__ import annotations

from dataclasses import dataclass

from autoposetrack.proposal import SegmentationProposal, mask_iou
from autoposetrack.reference.types import InitializationResult


@dataclass(frozen=True)
class DiscoveryMetrics:
    oracle_proposal_recall: bool
    matching_eligible: bool
    matching_correct: bool
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    selected_mask_iou: float
    best_proposal_iou: float
    failure_type: str


def evaluate_discovery(proposals: list[SegmentationProposal], result: InitializationResult, gt_mask, iou_threshold: float = 0.5) -> DiscoveryMetrics:
    ious = [mask_iou(item.mask, gt_mask) for item in proposals]
    best_iou = max(ious, default=0.0)
    correct_ids = {index for index, value in enumerate(ious) if value >= iou_threshold}
    ranked = [item.proposal_id for item in result.matches]
    recalls = {k: bool(correct_ids & set(ranked[:k])) for k in (1, 3, 5)}
    selected_id = ranked[0] if ranked else None
    selected_iou = ious[selected_id] if selected_id is not None else 0.0
    oracle = bool(correct_ids)
    matching_correct = oracle and selected_id in correct_ids
    if not oracle:
        failure = "proposal_failure"
    elif not matching_correct:
        failure = "matching_failure"
    elif not result.success:
        failure = "acceptance_rejection"
    else:
        failure = "success"
    return DiscoveryMetrics(oracle, oracle, matching_correct, recalls[1], recalls[3], recalls[5], selected_iou, best_iou, failure)
