"""Shared, dependency-light contracts at project boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
UInt8Array = npt.NDArray[np.uint8]


class PoseMode(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    NONE = "none"


@dataclass(frozen=True)
class FrameObservation:
    """One frame/object observation; GT is intentionally absent."""

    sequence_id: str
    frame_id: int
    object_id: int
    rgb: UInt8Array
    depth_m: FloatArray
    camera_matrix: FloatArray
    mask: npt.NDArray[np.bool_]
    timestamp_s: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        height, width = self.rgb.shape[:2]
        if self.depth_m.shape != (height, width):
            raise ValueError("depth_m must match rgb spatial dimensions")
        if self.mask.shape != (height, width):
            raise ValueError("mask must match rgb spatial dimensions")
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        if not np.isfinite(self.camera_matrix).all():
            raise ValueError("camera_matrix contains non-finite values")


@dataclass(frozen=True)
class PoseEstimate:
    """Object-to-camera pose estimate with translation in metres."""

    object_to_camera: FloatArray
    score: Optional[float]
    mode: PoseMode
    runtime_s: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        from autoposetrack.geometry.se3 import validate_transform

        validate_transform(self.object_to_camera)
        if self.runtime_s < 0:
            raise ValueError("runtime_s must be non-negative")
        if self.score is not None and not np.isfinite(self.score):
            raise ValueError("score must be finite when present")


@dataclass(frozen=True)
class ModelReference:
    object_id: int
    mesh_path: Path
    diameter_m: Optional[float] = None

