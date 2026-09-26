"""Empirical future-recovery trials from controlled pose perturbations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from autoposetrack.contracts import ModelReference, PoseEstimate, PoseMode
from autoposetrack.datasets.ycbv import YCBVSingleObjectDataset, YCBVTargetFrame
from autoposetrack.evaluation import add_m, adds_m
from autoposetrack.pose.interfaces import LocalPoseTracker
from autoposetrack.rollout.perturbation import (
    SE3Perturbation,
    perturb_pose_object_frame,
)


@dataclass(frozen=True)
class RecoveryTrial:
    trial_id: str
    sequence_id: str
    anchor_frame_id: int
    object_id: int
    rotation_deg_xyz: tuple[float, float, float]
    translation_m_xyz: tuple[float, float, float]
    initial_pose_error_m: float
    recovered: int
    first_recovery_step: Optional[int]
    minimum_future_error_m: Optional[float]
    attempted_steps: int
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RecoveryBasinGenerator:
    """Measure whether the frozen local tracker recovers within a horizon."""

    def __init__(
        self,
        dataset: YCBVSingleObjectDataset,
        tracker: LocalPoseTracker,
        model: ModelReference,
        model_points_m: np.ndarray,
        symmetric: bool,
        future_frames: int = 10,
        success_fraction: float = 0.10,
    ):
        if future_frames <= 0 or not 0 < success_fraction < 1:
            raise ValueError("invalid recovery protocol")
        if model.diameter_m is None:
            raise ValueError("model diameter is required")
        self.dataset, self.tracker, self.model = dataset, tracker, model
        self.points, self.symmetric = model_points_m, symmetric
        self.future_frames = future_frames
        self.threshold_m = model.diameter_m * success_fraction

    def run(
        self,
        sequence_targets: Sequence[YCBVTargetFrame],
        anchor_index: int,
        perturbation: SE3Perturbation,
    ) -> RecoveryTrial:
        ordered = sorted(sequence_targets, key=lambda item: item.frame_id)
        if not 0 <= anchor_index < len(ordered):
            raise IndexError("anchor_index is outside the sequence")
        anchor = ordered[anchor_index]
        annotation = self.dataset.annotation(
            anchor.sequence_id, anchor.frame_id, anchor.object_id
        )
        perturbed = perturb_pose_object_frame(annotation.object_to_camera, perturbation)
        initial_error = self._error(perturbed, annotation.object_to_camera)
        previous = PoseEstimate(
            perturbed, None, PoseMode.LOCAL, 0.0, {"source": "controlled_perturbation"}
        )
        errors: list[float] = []
        failure = ""
        self.tracker.reset()
        future = ordered[anchor_index : anchor_index + self.future_frames + 1]
        for target in future:
            try:
                estimate = self.tracker.track(
                    self.dataset.observation(target), previous, self.model
                )
                if estimate is None:
                    failure = "local tracker returned no pose"
                    break
                target_pose = self.dataset.annotation(
                    target.sequence_id, target.frame_id, target.object_id
                ).object_to_camera
                errors.append(self._error(estimate.object_to_camera, target_pose))
                previous = estimate
            except Exception as error:  # noqa: BLE001 - failure is trial data.
                failure = f"{type(error).__name__}: {error}"
                break
        recovery_steps = [
            index for index, error in enumerate(errors) if error < self.threshold_m
        ]
        return RecoveryTrial(
            f"{anchor.sequence_id}:{anchor.frame_id}:{perturbation.trial_id}",
            anchor.sequence_id,
            anchor.frame_id,
            anchor.object_id,
            perturbation.rotation_deg_xyz,
            perturbation.translation_m_xyz,
            initial_error,
            int(bool(recovery_steps)),
            recovery_steps[0] if recovery_steps else None,
            min(errors) if errors else None,
            len(errors),
            failure,
        )

    def _error(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return (
            adds_m(prediction, target, self.points)
            if self.symmetric
            else add_m(prediction, target, self.points)
        )
