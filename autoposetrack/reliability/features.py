"""Create observable temporal features and offline future-recovery labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Optional, Sequence

import numpy as np

from autoposetrack.training.dataset import FeatureDataset

FEATURE_NAMES = (
    "score", "score_delta", "runtime_s", "bbox_area_fraction",
    "bbox_center_x_fraction", "bbox_center_y_fraction",
    "translation_jump_m", "rotation_jump_deg", "history_valid_fraction",
)


def build_recoverability_features(
    rows: Sequence[Mapping[str, object]],
    diameter_m: float,
    history_frames: int = 5,
    future_frames: int = 10,
    success_fraction: float = 0.10,
) -> FeatureDataset:
    """Use past/current observable fields; GT error is used only for labels."""
    if history_frames <= 0 or future_frames <= 0:
        raise ValueError("history_frames and future_frames must be positive")
    if diameter_m <= 0 or not 0 < success_fraction < 1:
        raise ValueError("diameter and success fraction must be valid")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rollout_id"])].append(row)
    features, labels, sequence_ids, frame_ids, object_ids = [], [], [], [], []
    threshold = diameter_m * success_fraction
    for rollout_rows in grouped.values():
        ordered = sorted(rollout_rows, key=lambda row: int(row["step"]))
        for index, row in enumerate(ordered):
            history = ordered[max(0, index - history_frames + 1): index + 1]
            score = _number(row.get("score"))
            previous_score = _number(ordered[index - 1].get("score")) if index else score
            features.append([
                score, score - previous_score, _number(row.get("runtime_s")),
                _number(row.get("bbox_area_fraction")),
                _number(row.get("bbox_center_x_fraction")),
                _number(row.get("bbox_center_y_fraction")),
                _number(row.get("translation_jump_m")),
                _number(row.get("rotation_jump_deg")),
                sum(item.get("status") == "ok" for item in history) / len(history),
            ])
            future = ordered[index:min(len(ordered), index + future_frames + 1)]
            labels.append(int(any(
                item.get("status") == "ok"
                and _optional_number(item.get("pose_error_m")) is not None
                and float(item["pose_error_m"]) < threshold
                for item in future
            )))
            sequence_ids.append(str(row["sequence_id"]))
            frame_ids.append(int(row["frame_id"]))
            object_ids.append(int(row["object_id"]))
    dataset = FeatureDataset(
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        sequence_ids=np.asarray(sequence_ids, dtype=np.str_),
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        object_ids=np.asarray(object_ids, dtype=np.int64),
        feature_names=FEATURE_NAMES,
        metadata={
            "modality": "rgb", "history_frames": history_frames,
            "future_frames": future_frames, "success_metric": "ADD(-S)",
            "success_fraction_of_diameter": success_fraction,
            "success_threshold_m": threshold,
            "label_definition": "success at current or within next K rollout frames",
        },
    )
    dataset.validate()
    return dataset


def _optional_number(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _number(value: object) -> float:
    number = _optional_number(value)
    return 0.0 if number is None else number
