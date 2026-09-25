"""Model-agnostic reliability training orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import numpy.typing as npt

from .dataset import FeatureDataset
from .models import TrainableReliabilityModel
from .splits import SequenceSplit


def binary_metrics(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike
) -> dict[str, Optional[float]]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    if target.shape != score.shape or target.ndim != 1 or len(target) == 0:
        raise ValueError("labels and probabilities must be non-empty matching vectors")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    prediction = score >= 0.5
    return {
        "count": float(len(target)),
        "positive_rate": float(target.mean()),
        "accuracy_at_0.5": float(np.mean(prediction == target)),
        "brier": float(np.mean((score - target) ** 2)),
        "auroc": _auroc(target, score),
    }


def _auroc(
    labels: npt.NDArray[np.int64], scores: npt.NDArray[np.float64]
) -> Optional[float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


@dataclass(frozen=True)
class TrainingResult:
    history: Mapping[str, list[float]]
    train_metrics: Mapping[str, Optional[float]]
    validation_metrics: Optional[Mapping[str, Optional[float]]]
    test_metrics: Optional[Mapping[str, Optional[float]]]


class ReliabilityTrainer:
    def __init__(self, model: TrainableReliabilityModel):
        self.model = model

    def run(
        self,
        dataset: FeatureDataset,
        split: SequenceSplit,
        callback: Optional[Callable[[int, Mapping[str, float]], None]] = None,
    ) -> TrainingResult:
        dataset.validate()
        split.validate(dataset.sequence_ids)
        train = dataset.subset(split.train)
        history = self.model.fit(train.features, train.labels, callback=callback)
        return TrainingResult(
            history=history,
            train_metrics=binary_metrics(
                train.labels, self.model.predict_probability(train.features)
            ),
            validation_metrics=self._evaluate_subset(dataset, split.validation),
            test_metrics=self._evaluate_subset(dataset, split.test),
        )

    def _evaluate_subset(
        self, dataset: FeatureDataset, indices: npt.NDArray[np.int64]
    ) -> Optional[Mapping[str, Optional[float]]]:
        if not len(indices):
            return None
        subset = dataset.subset(indices)
        return binary_metrics(
            subset.labels, self.model.predict_probability(subset.features)
        )

    def save_checkpoint(
        self,
        path: Path,
        feature_names: tuple[str, ...],
        training_metadata: Mapping[str, Any],
    ) -> None:
        target = path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
        arrays = dict(self.model.state_dict())
        arrays["feature_names"] = np.asarray(feature_names)
        arrays["metadata_json"] = np.asarray(
            json.dumps(dict(training_metadata), sort_keys=True)
        )
        np.savez_compressed(target, **arrays)
