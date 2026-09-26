"""Contracts and validated configuration for motion proposals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BackgroundQuality:
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    median_reprojection_error: float
    coverage_ratio: float


@dataclass(frozen=True)
class BackgroundMotionModel:
    requested_type: str
    used_type: str
    matrix: FloatArray
    valid: bool
    fallback_used: Optional[str]
    quality: BackgroundQuality


@dataclass(frozen=True)
class MotionProposal:
    bbox_xyxy: BBox
    motion_score: float
    direction_score: float
    compactness_score: float
    temporal_score: float
    confidence: float
    area: int
    centroid: Tuple[float, float]
    mean_residual_flow: Tuple[float, float]
    max_residual_flow: float
    tracklet_id: Optional[int] = None
    age: int = 1

    def with_temporal(
        self, tracklet_id: int, age: int, temporal_score: float, confidence: float
    ) -> "MotionProposal":
        return replace(
            self,
            tracklet_id=tracklet_id,
            age=age,
            temporal_score=temporal_score,
            confidence=confidence,
        )


@dataclass
class MotionTracklet:
    tracklet_id: int
    age: int
    hits: int
    missed: int
    bbox_history: List[BBox]
    centroid_history: List[Tuple[float, float]]
    motion_history: List[Tuple[float, float]]
    confidence: float


@dataclass(frozen=True)
class MotionProposalResult:
    frame_id: int
    proposals: Tuple[MotionProposal, ...]
    raw_flow: FloatArray
    background_flow: FloatArray
    residual_flow: FloatArray
    motion_score_map: FloatArray
    motion_mask: BoolArray
    background_motion_model: BackgroundMotionModel
    debug_info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpticalFlowConfig:
    backend: str = "farneback"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2


@dataclass(frozen=True)
class BackgroundConfig:
    model: str = "homography"
    sample_step: int = 8
    ransac_threshold: float = 3.0
    min_matches: int = 24
    min_inlier_ratio: float = 0.3
    min_coverage_ratio: float = 0.15
    max_median_reprojection_error: float = 3.0
    fallback: str = "affine"


@dataclass(frozen=True)
class SaliencyConfig:
    threshold_type: str = "mad"
    mad_factor: float = 3.0
    percentile: float = 95.0
    fixed_threshold: float = 2.0
    minimum_threshold: float = 0.5


@dataclass(frozen=True)
class MorphologyConfig:
    enabled: bool = True
    median_kernel: int = 3
    open_kernel: int = 3
    close_kernel: int = 5
    fill_holes: bool = True


@dataclass(frozen=True)
class ProposalConfig:
    min_area_pixels: int = 8
    min_area_ratio: float = 0.0
    preserve_small_strong: bool = True
    small_strong_factor: float = 1.5
    bbox_expand_scale: float = 1.3
    merge_iou: float = 0.1
    merge_centroid_distance_ratio: float = 0.05
    merge_direction_cosine: float = 0.7
    top_k: int = 10
    weight_motion: float = 0.4
    weight_direction: float = 0.25
    weight_temporal: float = 0.25
    weight_compactness: float = 0.1


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool = True
    history_length: int = 5
    association_iou: float = 0.1
    association_distance_ratio: float = 0.08
    max_missed: int = 2


@dataclass(frozen=True)
class MotionProposalConfig:
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MotionProposalConfig":
        known = {
            "optical_flow": OpticalFlowConfig,
            "background": BackgroundConfig,
            "saliency": SaliencyConfig,
            "morphology": MorphologyConfig,
            "proposal": ProposalConfig,
            "temporal": TemporalConfig,
        }
        unknown = set(value) - set(known)
        if unknown:
            raise ValueError(f"unknown motion config sections: {sorted(unknown)}")
        sections: Dict[str, Any] = {}
        for name, section_type in known.items():
            payload = value.get(name, {})
            if not isinstance(payload, Mapping):
                raise ValueError(f"motion config section {name} must be a mapping")
            allowed = set(section_type.__dataclass_fields__)
            extra = set(payload) - allowed
            if extra:
                raise ValueError(f"unknown {name} options: {sorted(extra)}")
            sections[name] = section_type(**payload)
        config = cls(**sections)
        config.validate()
        return config

    def validate(self) -> None:
        if self.optical_flow.backend not in {"farneback", "raft", "gmflow"}:
            raise ValueError("optical_flow.backend must be farneback, raft, or gmflow")
        if self.background.model not in {"none", "affine", "homography"}:
            raise ValueError("background.model must be none, affine, or homography")
        if self.background.fallback not in {"none", "affine", "previous"}:
            raise ValueError("background.fallback must be none, affine, or previous")
        if self.saliency.threshold_type not in {"mad", "percentile", "fixed"}:
            raise ValueError("unknown saliency threshold_type")
        for name in ("median_kernel", "open_kernel", "close_kernel"):
            kernel = getattr(self.morphology, name)
            if kernel < 0 or (kernel > 1 and kernel % 2 == 0):
                raise ValueError(f"{name} must be zero/one or a positive odd number")
        if self.proposal.top_k <= 0 or self.proposal.bbox_expand_scale < 1.0:
            raise ValueError("top_k must be positive and bbox expansion at least one")
        weights = (
            self.proposal.weight_motion,
            self.proposal.weight_direction,
            self.proposal.weight_temporal,
            self.proposal.weight_compactness,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("proposal score weights must be non-negative and non-zero")
