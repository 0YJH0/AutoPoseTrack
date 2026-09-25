"""SE(3) helpers using column vectors and object-to-camera transforms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def validate_transform(transform: npt.ArrayLike, atol: float = 1e-5) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if not np.isfinite(matrix).all():
        raise ValueError("transform contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("last transform row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("rotation determinant must be +1")
    return matrix


def invert(transform: npt.ArrayLike) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ matrix[:3, 3])
    return result


def compose(left: npt.ArrayLike, right: npt.ArrayLike) -> np.ndarray:
    """Compose transforms as ``left @ right`` for column vectors."""

    return validate_transform(validate_transform(left) @ validate_transform(right))


def transform_points(
    object_to_camera: npt.ArrayLike, points_object_m: npt.ArrayLike
) -> np.ndarray:
    transform = validate_transform(object_to_camera)
    points = np.asarray(points_object_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_error_deg(prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
    pred = validate_transform(prediction)
    true = validate_transform(target)
    relative = pred[:3, :3] @ true[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error_m(prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
    pred = validate_transform(prediction)
    true = validate_transform(target)
    return float(np.linalg.norm(pred[:3, 3] - true[:3, 3]))

