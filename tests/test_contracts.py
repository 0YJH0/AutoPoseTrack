from pathlib import Path

import numpy as np
import pytest

from autoposetrack.contracts import (
    FrameObservation,
    ModelReference,
    PoseEstimate,
    PoseMode,
)


def test_observation_contract_has_no_ground_truth():
    frame = FrameObservation(
        sequence_id="000048",
        frame_id=1,
        object_id=2,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        camera_matrix=np.eye(3),
        bbox_xyxy=np.array([0.0, 0.0, 3.0, 2.0]),
        mask=np.ones((2, 3), dtype=bool),
    )
    frame.validate()
    assert not hasattr(frame, "ground_truth_pose")
    assert not hasattr(frame, "depth_m")


def test_rgb_observation_rejects_depth_metadata():
    frame = FrameObservation(
        sequence_id="000048",
        frame_id=1,
        object_id=2,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        camera_matrix=np.eye(3),
        bbox_xyxy=np.array([0.0, 0.0, 3.0, 2.0]),
        metadata={"depth_m": np.ones((2, 3))},
    )
    with pytest.raises(ValueError, match="RGB-only"):
        frame.validate()


def test_rgb_observation_rejects_invalid_bbox():
    frame = FrameObservation(
        sequence_id="000048",
        frame_id=1,
        object_id=2,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        camera_matrix=np.eye(3),
        bbox_xyxy=np.array([2.0, 0.0, 1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="bbox"):
        frame.validate()


def test_pose_contract_rejects_negative_runtime():
    pose = PoseEstimate(np.eye(4), 0.5, PoseMode.LOCAL, -1.0)
    with pytest.raises(ValueError, match="runtime"):
        pose.validate()


def test_model_reference_is_path_only():
    model = ModelReference(2, Path("models/obj_000002.ply"), 0.172)
    assert model.object_id == 2
