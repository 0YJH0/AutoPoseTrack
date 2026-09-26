#!/usr/bin/env python3
"""Join tracker errors with audited natural attributes without inventing buckets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ATTRIBUTES = (
    "visibility_fraction",
    "bbox_area_ratio",
    "visible_area_ratio",
    "translation_delta",
    "rotation_delta_deg",
    "nearest_reference_angle",
    "sharpness_score",
    "num_instances",
    "max_bbox_overlap",
)


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return math.nan
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def analyze(
    predictions: Sequence[Mapping[str, str]],
    attributes: Sequence[Mapping[str, str]],
    error_field: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    index: Dict[tuple[str, str, str], Mapping[str, str]] = {}
    duplicates = set()
    for row in attributes:
        key = (row["sequence_id"], row["frame_id"], row["object_id"])
        if key in index:
            duplicates.add(key)
        index[key] = row
    if duplicates:
        raise ValueError(
            "Attribute join is ambiguous for repeated object instances; include "
            "gt_id in both inputs before analysis."
        )
    joined: List[Dict[str, Any]] = []
    unmatched = []
    for prediction in predictions:
        key = (
            prediction["sequence_id"],
            prediction["frame_id"],
            prediction["object_id"],
        )
        attribute = index.get(key)
        if attribute is None:
            unmatched.append(key)
            continue
        row: Dict[str, Any] = dict(prediction)
        for name in DEFAULT_ATTRIBUTES:
            row[name] = attribute.get(name, "nan")
        joined.append(row)
    if unmatched:
        raise ValueError(f"{len(unmatched)} predictions have no matching attribute row")
    errors = np.asarray([float(row[error_field]) for row in joined], dtype=float)
    trends = {}
    for field in DEFAULT_ATTRIBUTES:
        pairs = []
        for row in joined:
            try:
                attribute_value = float(row[field])
                error_value = float(row[error_field])
            except (TypeError, ValueError):
                continue
            if math.isfinite(attribute_value) and math.isfinite(error_value):
                pairs.append((attribute_value, error_value))
        x = np.asarray([pair[0] for pair in pairs], dtype=float)
        y = np.asarray([pair[1] for pair in pairs], dtype=float)
        correlation = _spearman(x, y)
        trends[field] = {
            "num_samples": len(pairs),
            "spearman_error_correlation": (
                correlation if math.isfinite(correlation) else None
            ),
        }
    summary = {
        "schema_version": 1,
        "analysis_type": "continuous_error_attribute_alignment",
        "error_field": error_field,
        "num_predictions": len(predictions),
        "num_joined": len(joined),
        "threshold_free": True,
        "error_summary": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "q75": float(np.quantile(errors, 0.75)),
            "q90": float(np.quantile(errors, 0.90)),
            "max": float(np.max(errors)),
        },
        "attribute_trends": trends,
        "largest_error_frames": [
            {
                "sequence_id": row["sequence_id"],
                "frame_id": int(row["frame_id"]),
                "object_id": int(row["object_id"]),
                error_field: float(row[error_field]),
            }
            for row in sorted(
                joined,
                key=lambda item: float(item[error_field]),
                reverse=True,
            )[:10]
        ],
    }
    return joined, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-field", default="pose_error_m")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    joined, summary = analyze(
        _read(args.predictions), _read(args.attributes), args.error_field
    )
    with (args.output / "per_frame_with_attributes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    (args.output / "failure_alignment.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
