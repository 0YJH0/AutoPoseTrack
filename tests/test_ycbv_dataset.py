import json

import numpy as np
import pytest
from PIL import Image

from autoposetrack.datasets import YCBVSingleObjectDataset
from autoposetrack.evaluation import load_ascii_ply_vertices_m


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_ycbv_fixture(tmp_path):
    root = tmp_path / "ycbv"
    scene = root / "test" / "000050"
    (scene / "rgb").mkdir(parents=True)
    models = root / "models"
    models.mkdir()
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(
        scene / "rgb" / "000001.png"
    )
    _write_json(
        scene / "scene_gt.json",
        {
            "1": [
                {
                    "obj_id": 5,
                    "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "cam_t_m2c": [10, 20, 1000],
                }
            ]
        },
    )
    _write_json(
        scene / "scene_gt_info.json",
        {"1": [{"bbox_obj": [1, 1, 3, 2], "bbox_visib": [1, 1, 2, 2], "visib_fract": 0.8}]},
    )
    _write_json(
        scene / "scene_camera.json",
        {"1": {"cam_K": [100, 0, 3, 0, 100, 2, 0, 0, 1]}},
    )
    _write_json(models / "models_info.json", {"5": {"diameter": 200.0}})
    (models / "obj_000005.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n100 0 0\n",
        encoding="ascii",
    )
    return root


def test_ycbv_reader_separates_rgb_observation_and_metric_annotation(tmp_path):
    dataset = YCBVSingleObjectDataset(_make_ycbv_fixture(tmp_path), 5)
    assert dataset.sequence_ids() == ("000050",)
    target = dataset.target_frames()[0]
    observation = dataset.observation(target)
    annotation = dataset.annotation("50", 1, 5)
    assert observation.rgb.shape == (4, 6, 3)
    assert observation.bbox_xyxy.tolist() == [1.0, 1.0, 4.0, 3.0]
    assert "depth" not in observation.metadata
    assert annotation.object_to_camera[:3, 3].tolist() == pytest.approx(
        [0.01, 0.02, 1.0]
    )
    assert dataset.model_reference(5).diameter_m == pytest.approx(0.2)


def test_ascii_ply_vertices_are_converted_from_mm_to_m(tmp_path):
    root = _make_ycbv_fixture(tmp_path)
    points = load_ascii_ply_vertices_m(root / "models" / "obj_000005.ply")
    assert points.tolist() == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
