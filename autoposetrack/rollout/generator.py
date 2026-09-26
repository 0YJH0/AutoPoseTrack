"""Inference/evaluation-separated continuous tracking rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

from autoposetrack.contracts import ModelReference, PoseEstimate
from autoposetrack.datasets.interfaces import PoseAnnotation
from autoposetrack.datasets.ycbv import YCBVSingleObjectDataset, YCBVTargetFrame
from autoposetrack.evaluation import add_m, adds_m
from autoposetrack.geometry import rotation_error_deg, translation_error_m
from autoposetrack.pose.interfaces import GlobalPoseEstimator, LocalPoseTracker


@dataclass(frozen=True)
class RolloutRecord:
    rollout_id: str
    sequence_id: str
    frame_id: int
    object_id: int
    step: int
    status: str
    mode: str
    score: Optional[float]
    runtime_s: Optional[float]
    bbox_area_fraction: float
    bbox_center_x_fraction: float
    bbox_center_y_fraction: float
    visibility_fraction: Optional[float]
    translation_jump_m: Optional[float]
    rotation_jump_deg: Optional[float]
    pose_error_m: Optional[float]
    rotation_error_deg: Optional[float]
    translation_error_m: Optional[float]
    prediction_object_to_camera: Optional[list[list[float]]]
    target_object_to_camera: list[list[float]]
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RolloutGenerator:
    """Run global initialization once, then causal local RGB refinement."""

    def __init__(
        self,
        dataset: YCBVSingleObjectDataset,
        global_estimator: GlobalPoseEstimator,
        local_tracker: LocalPoseTracker,
        model: ModelReference,
        model_points_m: np.ndarray,
        symmetric: bool,
    ):
        self.dataset = dataset
        self.global_estimator = global_estimator
        self.local_tracker = local_tracker
        self.model = model
        self.model_points_m = model_points_m
        self.symmetric = symmetric

    def run(self, targets: Iterable[YCBVTargetFrame]) -> list[RolloutRecord]:
        grouped: dict[str, list[YCBVTargetFrame]] = {}
        for target in targets:
            grouped.setdefault(target.sequence_id, []).append(target)
        records: list[RolloutRecord] = []
        for sequence_id, sequence_targets in grouped.items():
            self.local_tracker.reset()
            previous: Optional[PoseEstimate] = None
            for step, target in enumerate(sorted(sequence_targets, key=lambda item: item.frame_id)):
                observation = self.dataset.observation(target)
                annotation = self.dataset.annotation(
                    target.sequence_id, target.frame_id, target.object_id
                )
                try:
                    estimate = (
                        self.global_estimator.estimate(observation, self.model)
                        if previous is None
                        else self.local_tracker.track(observation, previous, self.model)
                    )
                    record = self._record(
                        f"{sequence_id}:natural", step, observation, annotation, estimate, previous
                    )
                    if estimate is not None:
                        previous = estimate
                except Exception as error:  # noqa: BLE001 - failures are rollout data.
                    record = self._record(
                        f"{sequence_id}:natural", step, observation, annotation, None, previous,
                        f"{type(error).__name__}: {error}",
                    )
                records.append(record)
        return records

    def _record(self, rollout_id, step, observation, annotation: PoseAnnotation,
                estimate, previous, error="") -> RolloutRecord:
        height, width = observation.rgb.shape[:2]
        x1, y1, x2, y2 = observation.bbox_xyxy
        bbox_area = float((x2 - x1) * (y2 - y1) / (width * height))
        center_x = float((x1 + x2) / (2 * width))
        center_y = float((y1 + y2) / (2 * height))
        if estimate is None:
            return RolloutRecord(
                rollout_id, observation.sequence_id, observation.frame_id,
                observation.object_id, step, "failed", "none", None, None,
                bbox_area, center_x, center_y, annotation.visibility_fraction,
                None, None, None, None, None, None,
                annotation.object_to_camera.tolist(), error or "no pose candidate",
            )
        pose_error = (
            adds_m(estimate.object_to_camera, annotation.object_to_camera, self.model_points_m)
            if self.symmetric else
            add_m(estimate.object_to_camera, annotation.object_to_camera, self.model_points_m)
        )
        return RolloutRecord(
            rollout_id, observation.sequence_id, observation.frame_id,
            observation.object_id, step, "ok", estimate.mode.value, estimate.score,
            estimate.runtime_s, bbox_area, center_x, center_y,
            annotation.visibility_fraction,
            None if previous is None else translation_error_m(
                estimate.object_to_camera, previous.object_to_camera
            ),
            None if previous is None else rotation_error_deg(
                estimate.object_to_camera, previous.object_to_camera
            ),
            pose_error,
            rotation_error_deg(estimate.object_to_camera, annotation.object_to_camera),
            translation_error_m(estimate.object_to_camera, annotation.object_to_camera),
            estimate.object_to_camera.tolist(), annotation.object_to_camera.tolist(), error,
        )
