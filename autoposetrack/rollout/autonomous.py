"""Closed-loop recoverability-aware pose routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from autoposetrack.contracts import FrameObservation, ModelReference, PoseEstimate
from autoposetrack.pose.interfaces import GlobalPoseEstimator, LocalPoseTracker
from autoposetrack.reliability.routing import CostSensitiveRouter, RoutingAction

RecoverabilityFunction = Callable[
    [FrameObservation, PoseEstimate, Sequence[PoseEstimate]], float
]


@dataclass(frozen=True)
class RoutedPose:
    estimate: Optional[PoseEstimate]
    action: RoutingAction
    recoverability: Optional[float]
    local_candidate: Optional[PoseEstimate]


class RecoverabilityAwareTracker:
    """Frozen pose experts controlled by a learned recoverability policy.

    A local candidate is assessed before it is committed to history. A low-risk
    candidate is accepted, an uncertain one is withheld, and a persistently
    unrecoverable candidate triggers global RGB relocalization.
    """

    def __init__(
        self,
        global_estimator: GlobalPoseEstimator,
        local_tracker: LocalPoseTracker,
        router: CostSensitiveRouter,
        recoverability_fn: RecoverabilityFunction,
        model: ModelReference,
        max_history: int = 5,
    ):
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.global_estimator, self.local_tracker = global_estimator, local_tracker
        self.router, self.recoverability_fn, self.model = (
            router,
            recoverability_fn,
            model,
        )
        self.max_history = max_history
        self._current: Optional[PoseEstimate] = None
        self._history: list[PoseEstimate] = []

    def step(self, observation: FrameObservation) -> RoutedPose:
        if self._current is None:
            estimate = self.global_estimator.estimate(observation, self.model)
            if estimate is not None:
                self._commit(estimate)
            self.router.choose(0.0, has_pose=False)
            return RoutedPose(estimate, RoutingAction.GLOBAL, None, None)
        local = self.local_tracker.track(observation, self._current, self.model)
        if local is None:
            action = self.router.choose(0.0, has_pose=True)
            return self._apply(action, observation, None, 0.0)
        probability = float(
            self.recoverability_fn(observation, local, tuple(self._history))
        )
        action = self.router.choose(probability, has_pose=True)
        return self._apply(action, observation, local, probability)

    def _apply(self, action, observation, local, probability) -> RoutedPose:
        if action is RoutingAction.LOCAL:
            self._commit(local)
            return RoutedPose(local, action, probability, local)
        if action is RoutingAction.GLOBAL:
            estimate = self.global_estimator.estimate(observation, self.model)
            if estimate is not None:
                self._commit(estimate)
            return RoutedPose(estimate, action, probability, local)
        return RoutedPose(self._current, action, probability, local)

    def _commit(self, estimate: PoseEstimate) -> None:
        estimate.validate()
        self._current = estimate
        self._history.append(estimate)
        self._history = self._history[-self.max_history :]

    def reset(self) -> None:
        self._current = None
        self._history.clear()
        self.local_tracker.reset()
        self.router.reset()
