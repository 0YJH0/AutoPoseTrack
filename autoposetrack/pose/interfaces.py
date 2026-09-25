"""Narrow interfaces isolating third-party pose implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from autoposetrack.contracts import FrameObservation, ModelReference, PoseEstimate


class GlobalPoseEstimator(ABC):
    @abstractmethod
    def estimate(
        self, observation: FrameObservation, model: ModelReference
    ) -> Optional[PoseEstimate]:
        """Return a verified candidate or ``None`` when no candidate exists."""


class LocalPoseTracker(ABC):
    @abstractmethod
    def track(
        self,
        observation: FrameObservation,
        previous: PoseEstimate,
        model: ModelReference,
    ) -> Optional[PoseEstimate]:
        """Update a prior pose, returning ``None`` on explicit tracker failure."""

    def reset(self) -> None:
        """Clear implementation-specific temporal state."""

