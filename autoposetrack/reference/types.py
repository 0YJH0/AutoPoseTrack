"""Contracts for reference-guided target discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ReferenceFrame:
    reference_id: str
    image: npt.NDArray[np.uint8]
    mask: npt.NDArray[np.bool_]
    pose: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class EncodedReference:
    frame: ReferenceFrame
    crop: npt.NDArray[np.uint8]
    global_descriptor: npt.NDArray[np.float32]
    patch_features: Optional[npt.NDArray[np.float32]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalMatch:
    proposal_id: int
    score: float
    global_score: float
    local_score: float
    matched_reference_id: str
    local_match_count: int = 0


@dataclass(frozen=True)
class InitializationResult:
    success: bool
    target_mask: Optional[npt.NDArray[np.bool_]]
    bbox_xyxy: Optional[BBox]
    reference_score: float
    confidence: float
    matched_reference_id: Optional[str]
    best_score: float
    second_best_score: float
    score_margin: float
    matches: tuple[ProposalMatch, ...]
