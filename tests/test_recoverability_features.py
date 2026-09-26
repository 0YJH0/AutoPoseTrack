import numpy as np
import pytest

from autoposetrack.reliability import FEATURE_NAMES, build_recoverability_features


def _row(step, error, score=0.5, sequence="000050"):
    return {
        "rollout_id": f"{sequence}:natural",
        "sequence_id": sequence,
        "frame_id": 100 + step,
        "object_id": 5,
        "step": step,
        "status": "failed" if error is None else "ok",
        "score": "" if error is None else score,
        "runtime_s": 0.1,
        "bbox_area_fraction": 0.2,
        "bbox_center_x_fraction": 0.5,
        "bbox_center_y_fraction": 0.5,
        "translation_jump_m": 0.01,
        "rotation_jump_deg": 2.0,
        "pose_error_m": "" if error is None else error,
    }


def test_features_use_history_and_future_recovery_label():
    rows = [_row(0, 0.04), _row(1, None), _row(2, 0.009), _row(3, 0.04)]
    dataset = build_recoverability_features(
        rows, diameter_m=0.1, history_frames=2, future_frames=1
    )
    assert dataset.feature_names == FEATURE_NAMES
    np.testing.assert_array_equal(dataset.labels, [0, 1, 1, 0])
    assert dataset.features[1, -1] == pytest.approx(0.5)
    assert dataset.metadata["success_threshold_m"] == pytest.approx(0.01)


def test_features_do_not_mix_rollout_horizons():
    rows = [_row(0, None, sequence="000050"), _row(0, 0.001, sequence="000052")]
    dataset = build_recoverability_features(rows, diameter_m=0.1, future_frames=10)
    np.testing.assert_array_equal(dataset.labels, [0, 1])


def test_feature_configuration_is_validated():
    with pytest.raises(ValueError):
        build_recoverability_features([_row(0, 0.01)], 0.1, history_frames=0)
