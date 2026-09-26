import json

import numpy as np
from PIL import Image

from autoposetrack.datasets import load_hot3d_clip, load_ycbineoat


def test_ycbineoat_reader_uses_mask_only_as_gt(tmp_path):
    sequence = tmp_path / "cracker"
    (sequence / "rgb").mkdir(parents=True)
    (sequence / "gt_mask").mkdir()
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(sequence / "rgb/000001.png")
    mask = np.zeros((10, 12), dtype=np.uint8); mask[2:7, 3:9] = 255
    Image.fromarray(mask).save(sequence / "gt_mask/000001.png")
    frames = load_ycbineoat(sequence)
    assert frames[0].gt_boxes == ((3.0, 2.0, 9.0, 7.0),)


def test_hot3d_reader_selects_rgb_stream_and_amodal_box(tmp_path):
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(tmp_path / "000001.image_214-1.jpg")
    payload = {"obj": [{"object_bop_id": 7, "boxes_amodal": {"214-1": [1, 2, 8, 9]}}]}
    (tmp_path / "000001.objects.json").write_text(json.dumps(payload), encoding="utf-8")
    frames = load_hot3d_clip(tmp_path)
    assert frames[0].gt_boxes == ((1.0, 2.0, 8.0, 9.0),)
    assert frames[0].object_ids == ("7",)
