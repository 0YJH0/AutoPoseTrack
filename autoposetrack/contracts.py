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
    """One monocular-RGB frame/object observation; GT and depth are absent."""

    sequence_id: str
    frame_id: int
    object_id: int
    rgb: UInt8Array
    camera_matrix: FloatArray
    bbox_xyxy: FloatArray
    mask: Optional[npt.NDArray[np.bool_]] = None
    timestamp_s: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        height, width = self.rgb.shape[:2]
        if self.rgb.dtype != np.uint8:
            raise ValueError("rgb must use uint8 values")
        bbox = np.asarray(self.bbox_xyxy, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("bbox_xyxy must contain four finite values")
        x_min, y_min, x_max, y_max = bbox
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ValueError("bbox_xyxy must be non-empty and inside the image")
        if self.mask is not None and self.mask.shape != (height, width):
            raise ValueError("mask must match rgb spatial dimensions when present")
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        if not np.isfinite(self.camera_matrix).all():
            raise ValueError("camera_matrix contains non-finite values")
        if "depth" in self.metadata or "depth_m" in self.metadata:
            raise ValueError("RGB-only observations must not contain depth metadata")


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
    """CAD-enabled reference retained for auxiliary model-based baselines."""

    object_id: int
    mesh_path: Path
    diameter_m: Optional[float] = None


@dataclass(frozen=True)
class PosedReferenceImage:
    """One RGB reference view with a known object-to-camera pose."""

    image_path: Path
    object_to_camera: FloatArray
    camera_matrix: FloatArray
    mask_path: Optional[Path] = None

    def validate(self) -> None:
        from autoposetrack.geometry.se3 import validate_transform

        if not self.image_path.is_file():
            raise FileNotFoundError(f"reference image not found: {self.image_path}")
        validate_transform(self.object_to_camera)
        if self.camera_matrix.shape != (3, 3) or not np.isfinite(
            self.camera_matrix
        ).all():
            raise ValueError("reference camera_matrix must be finite 3x3")
        if self.mask_path is not None and not self.mask_path.is_file():
            raise FileNotFoundError(f"reference mask not found: {self.mask_path}")


@dataclass(frozen=True)
class ReferenceObject:
    """CAD-free object onboarding data shared by reference-based estimators.

    ``backend_metadata`` may point an adapter to a preprocessed SfM database,
    but the tracking pipeline only depends on this neutral contract.
    """

    object_id: int
    views: tuple[PosedReferenceImage, ...]
    diameter_m: Optional[float] = None
    sparse_point_cloud_path: Optional[Path] = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.views:
            raise ValueError("reference-based estimators require at least one view")
        for view in self.views:
            view.validate()
        if self.diameter_m is not None and self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive when provided")
        if self.sparse_point_cloud_path is not None and not self.sparse_point_cloud_path.is_file():
            raise FileNotFoundError(
                f"reference point cloud not found: {self.sparse_point_cloud_path}"
            )
