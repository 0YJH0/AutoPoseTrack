"""End-to-end full-frame motion proposal orchestration."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from .background import BackgroundMotionEstimator, background_flow_from_model
from .components import merge_proposals, rerank
from .components import extract_components as extract_motion_components
from .flow import FarnebackFlowEstimator, OpticalFlowEstimator
from .residual import flow_magnitude, residual_flow
from .saliency import motion_mask
from .temporal import MotionTrackletMemory
from .types import MotionProposalConfig, MotionProposalResult


class FullFrameMotionProposer:
    """Generate identity-agnostic full-frame proposals from two RGB frames."""

    def __init__(
        self,
        config: MotionProposalConfig,
        flow_estimator: Optional[OpticalFlowEstimator] = None,
    ) -> None:
        config.validate()
        self.config = config
        if flow_estimator is None and config.optical_flow.backend != "farneback":
            raise ValueError(
                f"backend '{config.optical_flow.backend}' requires an injected "
                "OpticalFlowEstimator adapter"
            )
        self.flow_estimator = flow_estimator or FarnebackFlowEstimator(
            config.optical_flow
        )
        self.background_estimator = BackgroundMotionEstimator(config.background)
        self.temporal_memory = MotionTrackletMemory(config.temporal)

    def reset(self) -> None:
        self.background_estimator.reset()
        self.temporal_memory.reset()

    def propose(
        self,
        previous_rgb: npt.NDArray[np.uint8],
        current_rgb: npt.NDArray[np.uint8],
        frame_id: int,
    ) -> MotionProposalResult:
        if previous_rgb.shape != current_rgb.shape:
            raise ValueError("previous and current RGB frames must have equal shape")
        started = time.perf_counter()
        flow_started = time.perf_counter()
        observed_flow = self.flow_estimator.estimate(previous_rgb, current_rgb)
        flow_time = time.perf_counter() - flow_started

        compensation_started = time.perf_counter()
        background_model = self.background_estimator.estimate(observed_flow)
        background_flow = background_flow_from_model(
            background_model, current_rgb.shape[:2]
        )
        compensated_flow = residual_flow(observed_flow, background_flow)
        compensation_time = time.perf_counter() - compensation_started

        proposal_started = time.perf_counter()
        score_map = flow_magnitude(compensated_flow)
        binary_mask, threshold = motion_mask(
            score_map, self.config.saliency, self.config.morphology
        )
        proposals = extract_motion_components(
            binary_mask,
            compensated_flow,
            threshold,
            self.config.proposal,
        )
        proposals = merge_proposals(
            proposals, self.config.proposal, current_rgb.shape[:2]
        )
        proposals = self.temporal_memory.update(proposals, current_rgb.shape[:2])
        proposals = rerank(proposals, self.config.proposal)
        proposal_time = time.perf_counter() - proposal_started
        raw_magnitude = flow_magnitude(observed_flow)
        background_magnitude = flow_magnitude(background_flow)
        quality = background_model.quality
        debug_info = {
            "threshold": threshold,
            "num_proposals": len(proposals),
            "background_model_type": background_model.used_type,
            "background_model_valid": background_model.valid,
            "background_fallback": background_model.fallback_used,
            "background_num_matches": quality.num_matches,
            "background_inlier_ratio": quality.inlier_ratio,
            "background_reprojection_error": quality.median_reprojection_error,
            "background_coverage_ratio": quality.coverage_ratio,
            "raw_flow_mean": float(np.mean(raw_magnitude)),
            "background_flow_mean": float(np.mean(background_magnitude)),
            "residual_flow_mean": float(np.mean(score_map)),
            "runtime_flow_s": flow_time,
            "runtime_compensation_s": compensation_time,
            "runtime_proposal_s": proposal_time,
            "runtime_total_s": time.perf_counter() - started,
        }
        return MotionProposalResult(
            frame_id=frame_id,
            proposals=tuple(proposals),
            raw_flow=observed_flow,
            background_flow=background_flow,
            residual_flow=compensated_flow,
            motion_score_map=score_map,
            motion_mask=binary_mask,
            background_motion_model=background_model,
            debug_info=debug_info,
        )


def motion_alignment_score(
    tracked_bbox: tuple[float, float, float, float],
    result: MotionProposalResult,
) -> float:
    """Record-only reliability cue: maximum tracker/proposal IoU."""
    from .components import bbox_iou

    return max(
        (bbox_iou(tracked_bbox, item.bbox_xyxy) for item in result.proposals),
        default=0.0,
    )
