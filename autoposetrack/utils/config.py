"""Strict YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: Mapping[str, Any]
    source: Path

    @property
    def name(self) -> str:
        return str(self.raw["experiment"]["name"])

    @property
    def seed(self) -> int:
        return int(self.raw["experiment"]["seed"])


def load_experiment_config(path: Path) -> ExperimentConfig:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    required_sections = {
        "experiment",
        "dataset",
        "model",
        "state_manager",
        "features",
        "evaluation",
        "output",
    }
    missing = required_sections - payload.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    experiment = payload["experiment"]
    if not isinstance(experiment, dict) or not {"name", "seed"} <= experiment.keys():
        raise ValueError("experiment requires name and seed")
    dataset_root = payload["dataset"].get("root")
    if not dataset_root:
        raise ValueError("dataset.root must be set explicitly")
    for section_name in ("dataset", "model"):
        section = payload[section_name]
        if section.get("modality") != "rgb":
            raise ValueError(f"{section_name}.modality must be 'rgb'")
        if section.get("allow_depth_at_inference") is not False:
            raise ValueError(
                f"{section_name}.allow_depth_at_inference must be false"
            )
    if payload.get("features", {}).get("depth") is not False:
        raise ValueError("features.depth must be false for RGB-only experiments")
    return ExperimentConfig(raw=payload, source=source)
