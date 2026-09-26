"""Causal fixed-window features and portable recoverability inference."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np


class TemporalWindow:
    """Flatten a causal feature history with an explicit validity mask."""

    def __init__(self, feature_names: tuple[str, ...], history_frames: int = 5):
        if not feature_names or history_frames <= 0:
            raise ValueError("temporal feature schema must be non-empty")
        self.feature_names = feature_names
        self.history_frames = history_frames
        self._history: deque[np.ndarray] = deque(maxlen=history_frames)

    @property
    def output_names(self) -> tuple[str, ...]:
        names = [
            f"lag_{lag}_{name}"
            for lag in range(self.history_frames - 1, -1, -1)
            for name in self.feature_names
        ]
        names.extend(
            f"lag_{lag}_valid" for lag in range(self.history_frames - 1, -1, -1)
        )
        return tuple(names)

    def append(self, features: Mapping[str, float]) -> np.ndarray:
        vector = np.asarray(
            [features[name] for name in self.feature_names], dtype=np.float64
        )
        if not np.isfinite(vector).all():
            raise ValueError("temporal features must be finite")
        self._history.append(vector)
        missing = self.history_frames - len(self._history)
        rows = [np.zeros(len(self.feature_names)) for _ in range(missing)]
        rows.extend(self._history)
        mask = np.asarray([0.0] * missing + [1.0] * len(self._history))
        return np.concatenate((np.asarray(rows).reshape(-1), mask))

    def reset(self) -> None:
        self._history.clear()


class CheckpointRecoverabilityPredictor:
    """Load the safe NPZ logistic checkpoint without training dependencies."""

    def __init__(self, checkpoint: Path):
        with np.load(checkpoint.expanduser().resolve(), allow_pickle=False) as payload:
            self.weights = np.asarray(payload["weights"], dtype=np.float64)
            self.bias = float(np.asarray(payload["bias"])[0])
            self.mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            self.scale = np.asarray(payload["feature_scale"], dtype=np.float64)
            self.feature_names = tuple(str(x) for x in payload["feature_names"])

    def predict(self, vector: np.ndarray) -> float:
        x = np.asarray(vector, dtype=np.float64)
        if x.shape != self.weights.shape:
            raise ValueError("recoverability feature dimension mismatch")
        logit = float(((x - self.mean) / self.scale) @ self.weights + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
