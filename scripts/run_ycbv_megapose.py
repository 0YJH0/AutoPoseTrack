#!/usr/bin/env python3
"""Run and evaluate pinned MegaPose RGB on one YCB-V object."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from autoposetrack.datasets import YCBVSingleObjectDataset
from autoposetrack.evaluation import (
    add_m,
    adds_m,
    load_ascii_ply_vertices_m,
    threshold_auc,
)
from autoposetrack.geometry import rotation_error_deg, translation_error_m
from autoposetrack.pose.global_estimator import MegaPoseRGBEstimator
from autoposetrack.utils.experiment_logging import ExperimentLogger
from autoposetrack.utils.reproducibility import build_manifest

FIELDS = (
    "sequence_id",
    "frame_id",
    "object_id",
    "status",
    "score",
    "runtime_s",
    "upstream_time_s",
    "rotation_orthogonality_residual",
    "visibility_fraction",
    "pose_error_name",
    "pose_error_m",
    "rotation_error_deg",
    "translation_error_m",
    "prediction_object_to_camera",
    "target_object_to_camera",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=5)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="megapose-1.0-RGB-multi-hypothesis"
    )
    parser.add_argument("--renderer-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--auc-threshold-m", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.auc_threshold_m <= 0:
        parser.error("--auc-threshold-m must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = ExperimentLogger(output)
    try:
        dataset = YCBVSingleObjectDataset(args.dataset_root, args.object_id)
        targets = dataset.target_frames(args.sequence)[: args.max_frames]
        model = dataset.model_reference(args.object_id)
        symmetric = dataset.is_symmetric()
        metric_name = "ADD-S" if symmetric else "ADD"
        config = {
            "dataset": {
                "name": "ycbv",
                "root": str(dataset.root),
                "split": "test_bop19",
                "object_id": args.object_id,
                "sequences": list(args.sequence) if args.sequence else None,
                "detection_source": "oracle_bbox_obj",
                "modality": "rgb",
            },
            "model": {
                "name": args.model_name,
                "renderer_workers": args.renderer_workers,
                "batch_size": args.batch_size,
                "allow_depth_at_inference": False,
            },
            "evaluation": {
                "metric": metric_name,
                "auc_threshold_m": args.auc_threshold_m,
                "max_frames": args.max_frames,
            },
            "output": str(output),
        }
        _write_yaml(output / "config.yaml", config)
        _write_json(output / "manifest.json", build_manifest(Path.cwd()))
        _write_json(
            output / "frame_index.json",
            {
                "count": len(targets),
                "frames": [
                    {
                        "sequence_id": item.sequence_id,
                        "frame_id": item.frame_id,
                        "object_id": item.object_id,
                        "instance_index": item.instance_index,
                        "visibility_fraction": item.visibility_fraction,
                    }
                    for item in targets
                ],
            },
        )
        logger.log_event(
            "target_index_created",
            object_id=args.object_id,
            available_frames=len(dataset.target_frames(args.sequence)),
            selected_frames=len(targets),
            sequences=list(dataset.sequence_ids()),
        )
        points = load_ascii_ply_vertices_m(model.mesh_path)
        if symmetric and len(points) > 2000:
            indices = np.linspace(0, len(points) - 1, 2000, dtype=np.int64)
            points = points[indices]
        logger.log_event(
            "model_geometry_loaded",
            mesh=str(model.mesh_path),
            points=len(points),
            diameter_m=model.diameter_m,
            metric=metric_name,
        )
        estimator = MegaPoseRGBEstimator(
            model,
            model_name=args.model_name,
            renderer_workers=args.renderer_workers,
            batch_size=args.batch_size,
        )
        logger.log_event("megapose_loaded", model_name=args.model_name)
        rows: list[dict[str, Any]] = []
        errors: list[float] = []
        runtimes: list[float] = []
        for step, target in enumerate(targets):
            row = _evaluate_target(dataset, estimator, model, target, points, symmetric)
            rows.append(row)
            if row["status"] == "ok":
                errors.append(float(row["pose_error_m"]))
                runtimes.append(float(row["runtime_s"]))
                logger.log_metrics(
                    step,
                    "test_frame",
                    {
                        "pose_error_m": float(row["pose_error_m"]),
                        "rotation_error_deg": float(row["rotation_error_deg"]),
                        "translation_error_m": float(row["translation_error_m"]),
                        "runtime_s": float(row["runtime_s"]),
                    },
                )
            else:
                logger.warning(
                    "frame failed sequence=%s frame=%s error=%s",
                    target.sequence_id,
                    target.frame_id,
                    row["error"],
                )
        _write_rows(output / "per_frame.csv", rows)
        metrics = _aggregate(rows, errors, runtimes, args.auc_threshold_m, metric_name)
        _write_json(output / "metrics.json", metrics)
        _write_events(output / "events.csv", rows)
        logger.log_event("run_completed", **metrics)
    except BaseException as error:
        logger.record_failure(error)
        raise
    finally:
        logger.close()
    print(output)


def _evaluate_target(dataset, estimator, model, target, points, symmetric):
    annotation = dataset.annotation(target.sequence_id, target.frame_id, target.object_id)
    base = {
        "sequence_id": target.sequence_id,
        "frame_id": target.frame_id,
        "object_id": target.object_id,
        "visibility_fraction": annotation.visibility_fraction,
        "pose_error_name": "ADD-S" if symmetric else "ADD",
    }
    try:
        estimate = estimator.estimate(dataset.observation(target), model)
        if estimate is None:
            return {**base, "status": "no_prediction", "error": "no pose candidate"}
        error = (
            adds_m(estimate.object_to_camera, annotation.object_to_camera, points)
            if symmetric
            else add_m(estimate.object_to_camera, annotation.object_to_camera, points)
        )
        return {
            **base,
            "status": "ok",
            "score": estimate.score,
            "runtime_s": estimate.runtime_s,
            "upstream_time_s": estimate.diagnostics.get("upstream_time_s"),
            "rotation_orthogonality_residual": estimate.diagnostics.get(
                "rotation_orthogonality_residual"
            ),
            "pose_error_m": error,
            "rotation_error_deg": rotation_error_deg(
                estimate.object_to_camera, annotation.object_to_camera
            ),
            "translation_error_m": translation_error_m(
                estimate.object_to_camera, annotation.object_to_camera
            ),
            "prediction_object_to_camera": json.dumps(
                estimate.object_to_camera.tolist(), separators=(",", ":")
            ),
            "target_object_to_camera": json.dumps(
                annotation.object_to_camera.tolist(), separators=(",", ":")
            ),
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - failed frames are evaluation data.
        return {**base, "status": "error", "error": f"{type(error).__name__}: {error}"}


def _aggregate(rows, errors, runtimes, threshold, metric_name):
    successful = len(errors)
    total = len(rows)
    return {
        "target_frames": total,
        "successful_frames": successful,
        "failed_frames": total - successful,
        "success_rate": successful / total if total else 0.0,
        "pose_error_name": metric_name,
        "mean_pose_error_m": float(np.mean(errors)) if errors else None,
        "median_pose_error_m": float(np.median(errors)) if errors else None,
        "auc_threshold_m": threshold,
        "pose_auc": threshold_auc(errors, threshold) if errors else None,
        "mean_runtime_s": float(np.mean(runtimes)) if runtimes else None,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ("event", "sequence_id", "frame_id", "details")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["status"] != "ok":
                writer.writerow(
                    {
                        "event": "inference_failure",
                        "sequence_id": row["sequence_id"],
                        "frame_id": row["frame_id"],
                        "details": row["error"],
                    }
                )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


if __name__ == "__main__":
    main()
