"""Robust affine/homography estimation from sampled dense-flow correspondences."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.types import (
    BackgroundConfig,
    BackgroundMotionModel,
    BackgroundQuality,
)


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "background compensation requires the 'motion' optional dependency"
        ) from error
    return cv2


def _correspondences(
    flow: npt.NDArray[np.floating], step: int
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    height, width = flow.shape[:2]
    offset = max(step // 2, 0)
    yy, xx = np.mgrid[offset:height:step, offset:width:step]
    source = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
    sampled = flow[yy, xx].reshape(-1, 2).astype(np.float32)
    destination = source + sampled
    valid = np.isfinite(destination).all(axis=1)
    valid &= (
        (destination[:, 0] >= 0)
        & (destination[:, 0] < width)
        & (destination[:, 1] >= 0)
        & (destination[:, 1] < height)
    )
    return source[valid], destination[valid]


def _coverage(points: npt.NDArray[np.floating], width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    cv2 = _cv2()
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull) / float(width * height))


def _quality(
    source: npt.NDArray[np.float32],
    destination: npt.NDArray[np.float32],
    matrix: npt.NDArray[np.floating],
    inliers: npt.NDArray[np.bool_],
    width: int,
    height: int,
) -> BackgroundQuality:
    homogeneous = np.column_stack((source, np.ones(len(source))))
    predicted_h = (matrix @ homogeneous.T).T
    predicted = predicted_h[:, :2] / predicted_h[:, 2:3]
    error = np.linalg.norm(predicted - destination, axis=1)
    selected_error = error[inliers] if np.any(inliers) else error
    return BackgroundQuality(
        num_matches=len(source),
        num_inliers=int(np.sum(inliers)),
        inlier_ratio=float(np.mean(inliers)) if len(inliers) else 0.0,
        median_reprojection_error=(
            float(np.median(selected_error)) if len(selected_error) else float("inf")
        ),
        coverage_ratio=_coverage(source[inliers], width, height),
    )


def _identity_model(
    requested: str, used: str, fallback: Optional[str], num_matches: int = 0
) -> BackgroundMotionModel:
    quality = BackgroundQuality(num_matches, 0, 0.0, float("inf"), 0.0)
    return BackgroundMotionModel(
        requested_type=requested,
        used_type=used,
        matrix=np.eye(3, dtype=np.float64),
        valid=False,
        fallback_used=fallback,
        quality=quality,
    )


class BackgroundMotionEstimator:
    def __init__(self, config: BackgroundConfig) -> None:
        self.config = config
        self._previous_valid: Optional[BackgroundMotionModel] = None

    def reset(self) -> None:
        self._previous_valid = None

    def _fit(
        self,
        model_type: str,
        source: npt.NDArray[np.float32],
        destination: npt.NDArray[np.float32],
        width: int,
        height: int,
        requested: str,
        fallback: Optional[str],
    ) -> BackgroundMotionModel:
        cv2 = _cv2()
        matrix: Optional[npt.NDArray[np.floating]]
        mask: Optional[npt.NDArray[np.uint8]]
        if model_type == "homography":
            matrix, mask = cv2.findHomography(
                source,
                destination,
                cv2.RANSAC,
                self.config.ransac_threshold,
            )
        elif model_type == "affine":
            affine, mask = cv2.estimateAffine2D(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
            )
            matrix = None
            if affine is not None:
                matrix = np.vstack((affine, np.asarray([0.0, 0.0, 1.0])))
        else:
            raise ValueError(f"unsupported background model: {model_type}")
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return _identity_model(requested, model_type, fallback, len(source))
        inliers = mask.reshape(-1).astype(bool)
        quality = _quality(source, destination, matrix, inliers, width, height)
        valid = (
            quality.num_matches >= self.config.min_matches
            and quality.inlier_ratio >= self.config.min_inlier_ratio
            and quality.coverage_ratio >= self.config.min_coverage_ratio
            and quality.median_reprojection_error
            <= self.config.max_median_reprojection_error
        )
        return BackgroundMotionModel(
            requested_type=requested,
            used_type=model_type,
            matrix=np.asarray(matrix, dtype=np.float64),
            valid=valid,
            fallback_used=fallback,
            quality=quality,
        )

    def estimate(
        self, flow: npt.NDArray[np.floating]
    ) -> BackgroundMotionModel:
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError("flow must have shape (H, W, 2)")
        height, width = flow.shape[:2]
        requested = self.config.model
        if requested == "none":
            return BackgroundMotionModel(
                requested_type="none",
                used_type="none",
                matrix=np.eye(3, dtype=np.float64),
                valid=True,
                fallback_used=None,
                quality=BackgroundQuality(0, 0, 1.0, 0.0, 1.0),
            )
        source, destination = _correspondences(flow, self.config.sample_step)
        if len(source) < self.config.min_matches:
            primary = _identity_model(requested, requested, None, len(source))
        else:
            primary = self._fit(
                requested, source, destination, width, height, requested, None
            )
        if primary.valid:
            self._previous_valid = primary
            return primary
        fallback = self.config.fallback
        if fallback == "affine" and requested != "affine" and len(source) >= 3:
            secondary = self._fit(
                "affine", source, destination, width, height, requested, "affine"
            )
            if secondary.valid:
                self._previous_valid = secondary
                return secondary
            return _identity_model(requested, "none", "none", len(source))
        if fallback == "previous" and self._previous_valid is not None:
            previous = self._previous_valid
            return BackgroundMotionModel(
                requested_type=requested,
                used_type=previous.used_type,
                matrix=previous.matrix.copy(),
                valid=False,
                fallback_used="previous",
                quality=primary.quality,
            )
        return _identity_model(requested, "none", "none", len(source))


def background_flow_from_model(
    model: BackgroundMotionModel, shape: Tuple[int, int]
) -> npt.NDArray[np.float32]:
    return background_flow_from_matrix(model.matrix, shape)


def background_flow_from_matrix(
    matrix: npt.NDArray[np.floating], shape: Tuple[int, int]
) -> npt.NDArray[np.float32]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)
    transformed_h = (matrix @ points.T).T
    denominator = transformed_h[:, 2]
    valid = np.abs(denominator) > 1e-8
    transformed = points[:, :2].astype(np.float64)
    transformed[valid] = transformed_h[valid, :2] / denominator[valid, None]
    displacement = transformed - points[:, :2]
    return displacement.reshape(height, width, 2).astype(np.float32)
