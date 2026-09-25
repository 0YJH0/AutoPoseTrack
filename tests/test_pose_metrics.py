import numpy as np
import pytest

from autoposetrack.evaluation import add_m, adds_m, threshold_auc


def test_add_for_pure_translation():
    model = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    target = np.eye(4)
    prediction = np.eye(4)
    prediction[0, 3] = 0.02
    assert add_m(prediction, target, model) == pytest.approx(0.02)


def test_adds_handles_symmetric_point_set():
    model = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.eye(4)
    prediction = np.eye(4)
    prediction[0, 0] = -1.0
    prediction[1, 1] = -1.0
    assert adds_m(prediction, target, model) == pytest.approx(0.0)


def test_threshold_auc_reference_values():
    assert threshold_auc([0.0, 0.05, 0.2], 0.1) == pytest.approx(0.5)

