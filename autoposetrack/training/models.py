"""Trainable reliability model protocol and an auditable NumPy baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import numpy.typing as npt


class TrainableReliabilityModel(ABC):
    @abstractmethod
    def fit(self, features: npt.ArrayLike, labels: npt.ArrayLike) -> Mapping[str, list[float]]:
        pass

    @abstractmethod
    def predict_probability(self, features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def state_dict(self) -> Mapping[str, npt.NDArray[np.float64]]:
        pass


@dataclass
class LogisticRegressionGD(TrainableReliabilityModel):
    learning_rate: float = 0.05
    epochs: int = 500
    l2: float = 1e-4
    weights: Optional[npt.NDArray[np.float64]] = None
    bias: float = 0.0
    feature_mean: Optional[npt.NDArray[np.float64]] = None
    feature_scale: Optional[npt.NDArray[np.float64]] = None

    @staticmethod
    def _sigmoid(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        clipped = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, features: npt.ArrayLike, labels: npt.ArrayLike) -> Mapping[str, list[float]]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0:
            raise ValueError("features and labels must have shapes (N,D) and (N,)")
        if set(np.unique(y)).difference({0.0, 1.0}):
            raise ValueError("labels must be binary")
        if len(np.unique(y)) < 2:
            raise ValueError("training labels must contain both classes")
        if self.learning_rate <= 0 or self.epochs < 1 or self.l2 < 0:
            raise ValueError("invalid optimizer hyperparameters")

        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        normalized = (x - self.feature_mean) / self.feature_scale
        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = 0.0
        losses: list[float] = []
        for _ in range(self.epochs):
            probabilities = self._sigmoid(normalized @ self.weights + self.bias)
            error = probabilities - y
            self.weights -= self.learning_rate * (
                normalized.T @ error / len(x) + self.l2 * self.weights
            )
            self.bias -= self.learning_rate * float(error.mean())
            epsilon = 1e-12
            loss = -np.mean(
                y * np.log(probabilities + epsilon)
                + (1.0 - y) * np.log(1.0 - probabilities + epsilon)
            ) + 0.5 * self.l2 * float(self.weights @ self.weights)
            losses.append(float(loss))
        return {"train_loss": losses}

    def predict_probability(self, features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.weights):
            raise ValueError("feature dimension does not match fitted model")
        normalized = (x - self.feature_mean) / self.feature_scale
        return self._sigmoid(normalized @ self.weights + self.bias)

    def state_dict(self) -> Mapping[str, npt.NDArray[np.float64]]:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        return {
            "weights": self.weights.copy(),
            "bias": np.asarray([self.bias], dtype=np.float64),
            "feature_mean": self.feature_mean.copy(),
            "feature_scale": self.feature_scale.copy(),
        }
