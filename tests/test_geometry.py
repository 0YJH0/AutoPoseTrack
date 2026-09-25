import numpy as np
import pytest

from autoposetrack.geometry import (
    compose,
    invert,
    rotation_error_deg,
    transform_points,
    translation_error_m,
)


def transform_z(degrees: float, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    result = np.eye(4)
    result[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def test_inverse_and_compose_are_identity():
    transform = transform_z(30, (0.1, -0.2, 0.8))
    np.testing.assert_allclose(compose(transform, invert(transform)), np.eye(4), atol=1e-7)


def test_column_vector_point_convention():
    transform = transform_z(90, (1.0, 2.0, 3.0))
    point = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(transform_points(transform, point), [[1.0, 3.0, 3.0]], atol=1e-7)


def test_pose_errors_have_explicit_units():
    target = np.eye(4)
    prediction = transform_z(60, (0.03, 0.04, 0.0))
    assert rotation_error_deg(prediction, target) == pytest.approx(60.0)
    assert translation_error_m(prediction, target) == pytest.approx(0.05)


def test_invalid_reflection_is_rejected():
    transform = np.eye(4)
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="determinant"):
        invert(transform)

