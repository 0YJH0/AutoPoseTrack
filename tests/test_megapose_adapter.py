import numpy as np
import pytest

from autoposetrack.pose.global_estimator.megapose import _project_to_se3


def test_project_to_se3_repairs_small_rotation_drift():
    pose = np.eye(4)
    pose[0, 0] = 1.001
    projected, residual = _project_to_se3(pose)
    assert residual > 0
    assert projected[:3, :3].T @ projected[:3, :3] == pytest.approx(np.eye(3))
    assert np.linalg.det(projected[:3, :3]) == pytest.approx(1.0)


def test_project_to_se3_rejects_large_drift():
    pose = np.eye(4)
    pose[:3, :3] *= 2
    with pytest.raises(ValueError, match="residual is too large"):
        _project_to_se3(pose)
