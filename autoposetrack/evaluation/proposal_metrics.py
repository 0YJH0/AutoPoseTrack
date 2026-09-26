"""GT-isolated proposal-level metrics; never imported by the proposer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from autoposetrack.motion.components import bbox_iou
from autoposetrack.motion.types import BBox, MotionProposal


@dataclass(frozen=True)
class ProposalFrameMetrics:
    max_iou: float
    center_recalled: bool
    recall_at_k: Dict[int, bool]
    num_proposals: int


def evaluate_proposals(
    proposals: Sequence[MotionProposal],
    gt_bbox_xyxy: BBox,
    iou_threshold: float = 0.5,
    top_k: Iterable[int] = (1, 3, 5, 10),
) -> ProposalFrameMetrics:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(proposals, key=lambda item: item.confidence, reverse=True)
    overlaps = [bbox_iou(item.bbox_xyxy, gt_bbox_xyxy) for item in ordered]
    gt_center = (
        (gt_bbox_xyxy[0] + gt_bbox_xyxy[2]) / 2.0,
        (gt_bbox_xyxy[1] + gt_bbox_xyxy[3]) / 2.0,
    )
    center_recalled = any(
        item.bbox_xyxy[0] <= gt_center[0] <= item.bbox_xyxy[2]
        and item.bbox_xyxy[1] <= gt_center[1] <= item.bbox_xyxy[3]
        for item in ordered
    )
    recall = {
        int(k): any(value >= iou_threshold for value in overlaps[: int(k)])
        for k in top_k
    }
    return ProposalFrameMetrics(
        max_iou=max(overlaps, default=0.0),
        center_recalled=center_recalled,
        recall_at_k=recall,
        num_proposals=len(ordered),
    )


def summarize_proposal_metrics(
    frames: Sequence[ProposalFrameMetrics],
) -> Dict[str, float]:
    if not frames:
        raise ValueError("at least one evaluated frame is required")
    keys = sorted({key for frame in frames for key in frame.recall_at_k})
    summary = {
        f"recall_at_{key}": sum(frame.recall_at_k[key] for frame in frames)
        / len(frames)
        for key in keys
    }
    summary.update(
        {
            "mean_max_iou": sum(frame.max_iou for frame in frames) / len(frames),
            "center_recall": sum(frame.center_recalled for frame in frames)
            / len(frames),
            "average_num_proposals": sum(frame.num_proposals for frame in frames)
            / len(frames),
            "num_frames": float(len(frames)),
        }
    )
    return summary
