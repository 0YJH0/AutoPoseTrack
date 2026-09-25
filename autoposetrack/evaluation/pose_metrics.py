"""Standard pose metrics. Inputs use metres consistently."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autoposetrack.geometry.se3 import transform_points


def add_m(
    prediction: npt.ArrayLike, target: npt.ArrayLike, model_points_m: npt.ArrayLike
) -> float:
    pred_points = transform_points(prediction, model_points_m)
    true_points = transform_points(target, model_points_m)
    return float(np.linalg.norm(pred_points - true_points, axis=1).mean())


def adds_m(
    prediction: npt.ArrayLike, target: npt.ArrayLike, model_points_m: npt.ArrayLike
) -> float:
    """ADD-S via exact nearest neighbours; intended as a reference implementation."""

    pred_points = transform_points(prediction, model_points_m)
    true_points = transform_points(target, model_points_m)
    distances = np.linalg.norm(
        pred_points[:, None, :] - true_points[None, :, :], axis=2
    )
    return float(distances.min(axis=1).mean())


def threshold_auc(errors_m: npt.ArrayLike, max_threshold_m: float) -> float:
    """Normalized area under the empirical accuracy-threshold curve."""

    if max_threshold_m <= 0:
        raise ValueError("max_threshold_m must be positive")
    errors = np.asarray(errors_m, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("errors_m must be a non-empty 1D array")
    clipped = np.minimum(errors, max_threshold_m)
    return float(np.mean(1.0 - clipped / max_threshold_m))

