import json
from pathlib import Path

import numpy as np
from PIL import Image

from autoposetrack.analysis import AuditConfig, audit_bop_dataset


def _make_dataset(root: Path) -> None:
    scene = root / "test" / "000001"
    (scene / "rgb").mkdir(parents=True)
    for frame_id in (0, 2):
        array = np.zeros((20, 30, 3), dtype=np.uint8)
        array[5:15, 8:20] = 100 + frame_id
        Image.fromarray(array).save(scene / "rgb" / f"{frame_id:06d}.png")
    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    gt = {
        "0": [{"cam_R_m2c": identity, "cam_t_m2c": [0, 0, 1000], "obj_id": 5}],
        "2": [{"cam_R_m2c": identity, "cam_t_m2c": [10, 0, 1000], "obj_id": 5}],
    }
    info = {
        key: [
            {
                "bbox_obj": [8, 5, 12, 10],
                "bbox_visib": [8, 5, 12, 10],
                "px_count_all": 120,
                "px_count_valid": 120,
                "px_count_visib": 90,
                "visib_fract": 0.75,
            }
        ]
        for key in gt
    }
    (scene / "scene_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    (scene / "scene_gt_info.json").write_text(json.dumps(info), encoding="utf-8")


def test_audit_writes_required_outputs_and_preserves_unknowns(tmp_path: Path) -> None:
    dataset = tmp_path / "ycbv"
    output = tmp_path / "audit"
    _make_dataset(dataset)
    summary = audit_bop_dataset(
        AuditConfig(dataset_root=dataset, output_dir=output, visual_samples=1)
    )
    assert summary["num_frames"] == 2
    assert summary["num_annotations"] == 2
    assert summary["is_continuous_tracking_split"] is False
    for name in (
        "dataset_summary.json",
        "per_frame_attributes.csv",
        "per_object_statistics.csv",
        "per_sequence_statistics.csv",
        "attribute_distribution.csv",
    ):
        assert (output / name).exists()
    text = (output / "per_frame_attributes.csv").read_text(encoding="utf-8")
    assert ",nan,nan," in text
    assert "0.01" in text
    assert (output / "figures" / "visual_audit_contact_sheet.jpg").exists()
