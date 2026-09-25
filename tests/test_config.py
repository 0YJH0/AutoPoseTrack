from pathlib import Path

import pytest
import yaml

from autoposetrack.utils.config import load_experiment_config


def test_checked_in_smoke_config_is_strict_rgb():
    config = load_experiment_config(
        Path("configs/experiments/megapose_rgb_ycbv_smoke.yaml")
    )
    assert config.raw["dataset"]["modality"] == "rgb"
    assert config.raw["model"]["modality"] == "rgb"
    assert config.raw["features"]["depth"] is False


def test_rgb_experiment_rejects_depth(tmp_path):
    source = Path("configs/experiments/megapose_rgb_ycbv_smoke.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["features"]["depth"] = True
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="features.depth"):
        load_experiment_config(target)

