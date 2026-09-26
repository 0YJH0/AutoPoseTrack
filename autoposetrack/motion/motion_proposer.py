"""End-to-end full-frame motion proposal orchestration."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from .background import BackgroundMotionEstimator
from .components import merge_proposals, rerank
from .components import extract_components as extract_motion_components
from .flow import (
    FarnebackFlowEstimator,
    OpticalFlowEstimator,
    TorchvisionRaftFlowEstimator,
)
from .dense import (
    DenseMotionBackend,
    NumpyDenseMotionBackend,
    TorchCudaMotionBackend,
)
from .saliency import postprocess_binary_mask
from .temporal import MotionTrackletMemory
from .types import MotionProposalConfig, MotionProposalResult


class FullFrameMotionProposer:
    """Generate identity-agnostic full-frame proposals from two RGB frames."""

    def __init__(
        self,
        config: MotionProposalConfig,
        flow_estimator: Optional[OpticalFlowEstimator] = None,
        dense_backend: Optional[DenseMotionBackend] = None,
    ) -> None:
        config.validate()
        self.config = config
        if flow_estimator is None and config.optical_flow.backend == "gmflow":
            raise ValueError(
                f"backend '{config.optical_flow.backend}' requires an injected "
                "OpticalFlowEstimator adapter"
            )
        if flow_estimator is not None:
            self.flow_estimator = flow_estimator
        elif config.optical_flow.backend == "raft":
            self.flow_estimator = TorchvisionRaftFlowEstimator(config.optical_flow)
        else:
            self.flow_estimator = FarnebackFlowEstimator(config.optical_flow)
        self.background_estimator = BackgroundMotionEstimator(config.background)
        if dense_backend is not None:
            self.dense_backend = dense_backend
        elif config.dense_processing.backend == "torch_cuda":
            self.dense_backend = TorchCudaMotionBackend(
                config.dense_processing.device
            )
        else:
            self.dense_backend = NumpyDenseMotionBackend()
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
        dense = self.dense_backend.process(
            observed_flow,
            background_model.matrix,
            self.config.saliency,
        )
        compensation_time = time.perf_counter() - compensation_started

        proposal_started = time.perf_counter()
        binary_mask = postprocess_binary_mask(
            dense.threshold_mask, self.config.morphology
        )
        proposals = extract_motion_components(
            binary_mask,
            dense.residual_flow,
            dense.threshold,
            self.config.proposal,
        )
        proposals = merge_proposals(
            proposals, self.config.proposal, current_rgb.shape[:2]
        )
        proposals = self.temporal_memory.update(proposals, current_rgb.shape[:2])
        proposals = rerank(proposals, self.config.proposal)
        proposal_time = time.perf_counter() - proposal_started
        quality = background_model.quality
        debug_info = {
            "threshold": dense.threshold,
            "num_proposals": len(proposals),
            "background_model_type": background_model.used_type,
            "background_model_valid": background_model.valid,
            "background_fallback": background_model.fallback_used,
            "background_num_matches": quality.num_matches,
            "background_inlier_ratio": quality.inlier_ratio,
            "background_reprojection_error": quality.median_reprojection_error,
            "background_coverage_ratio": quality.coverage_ratio,
            "raw_flow_mean": float(np.mean(dense.raw_magnitude)),
            "background_flow_mean": float(np.mean(dense.background_magnitude)),
            "residual_flow_mean": float(np.mean(dense.residual_magnitude)),
            "dense_backend": dense.backend,
            "dense_device": dense.device,
            "runtime_dense_backend_s": dense.runtime_s,
            "runtime_flow_s": flow_time,
            "runtime_compensation_s": compensation_time,
            "runtime_proposal_s": proposal_time,
            "runtime_total_s": time.perf_counter() - started,
        }
        return MotionProposalResult(
            frame_id=frame_id,
            proposals=tuple(proposals),
            raw_flow=observed_flow,
            background_flow=dense.background_flow,
            residual_flow=dense.residual_flow,
            motion_score_map=dense.residual_magnitude,
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
