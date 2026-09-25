"""Atomic creation of the standard experiment artifact tree."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class RunArtifacts:
    def __init__(self, root: Path, experiment_name: str):
        if not experiment_name or Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must be a safe single path component")
        self.path = root.expanduser().resolve() / experiment_name

    def create(self, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "figures").mkdir()
        (self.path / "logs").mkdir()
        self._write_yaml("config.yaml", config)
        self._write_json("manifest.json", manifest)

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        self._write_json("metrics.json", metrics)

    def write_rows(
        self, filename: str, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]
    ) -> None:
        target = self.path / filename
        with target.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        with (self.path / filename).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _write_yaml(self, filename: str, payload: Mapping[str, Any]) -> None:
        with (self.path / filename).open("x", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False)

