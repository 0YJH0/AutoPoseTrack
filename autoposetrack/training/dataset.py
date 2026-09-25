"""Portable feature dataset used after tracker rollouts are generated."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FeatureDataset:
    """Observable features and offline binary supervision.

    A row identifies one frame/object hypothesis. Ground-truth poses are not
    stored here; upstream dataset generation converts them into an offline label.
    Sequence IDs are mandatory to make adjacent-frame leakage detectable.
    """

    features: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64]
    sequence_ids: npt.NDArray[np.str_]
    frame_ids: npt.NDArray[np.int64]
    object_ids: npt.NDArray[np.int64]
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape (N, D)")
        count, dimension = self.features.shape
        if count == 0 or dimension == 0:
            raise ValueError("features must be non-empty")
        if len(self.feature_names) != dimension:
            raise ValueError("feature_names length must match feature dimension")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        for name, values in (
            ("labels", self.labels),
            ("sequence_ids", self.sequence_ids),
            ("frame_ids", self.frame_ids),
            ("object_ids", self.object_ids),
        ):
            if values.ndim != 1 or len(values) != count:
                raise ValueError(f"{name} must have shape (N,)")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain NaN or infinite values")
        if not set(np.unique(self.labels)).issubset({0, 1}):
            raise ValueError("labels must be binary values 0 or 1")
        if any(not value for value in self.sequence_ids.tolist()):
            raise ValueError("sequence_ids must be non-empty")

    def subset(self, indices: npt.ArrayLike) -> FeatureDataset:
        selected = np.asarray(indices)
        return FeatureDataset(
            features=self.features[selected],
            labels=self.labels[selected],
            sequence_ids=self.sequence_ids[selected],
            frame_ids=self.frame_ids[selected],
            object_ids=self.object_ids[selected],
            feature_names=self.feature_names,
            metadata=self.metadata,
        )

    def save(self, path: Path) -> None:
        self.validate()
        target = path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite dataset: {target}")
        np.savez_compressed(
            target,
            features=self.features,
            labels=self.labels,
            sequence_ids=self.sequence_ids,
            frame_ids=self.frame_ids,
            object_ids=self.object_ids,
            feature_names=np.asarray(self.feature_names),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )

    @classmethod
    def load(
        cls, path: Path, required_features: Optional[Sequence[str]] = None
    ) -> FeatureDataset:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"feature dataset not found: {source}")
        with np.load(source, allow_pickle=False) as payload:
            dataset = cls(
                features=np.asarray(payload["features"], dtype=np.float64),
                labels=np.asarray(payload["labels"], dtype=np.int64),
                sequence_ids=np.asarray(payload["sequence_ids"], dtype=np.str_),
                frame_ids=np.asarray(payload["frame_ids"], dtype=np.int64),
                object_ids=np.asarray(payload["object_ids"], dtype=np.int64),
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                metadata=json.loads(str(payload["metadata_json"])),
            )
        dataset.validate()
        if required_features is not None and tuple(required_features) != dataset.feature_names:
            raise ValueError(
                f"feature schema mismatch: expected {tuple(required_features)}, "
                f"found {dataset.feature_names}"
            )
        return dataset
