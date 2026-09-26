"""Narrow RGB pose interfaces isolating third-party implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

from autoposetrack.contracts import (
    FrameObservation,
    ModelReference,
    PoseEstimate,
    ReferenceObject,
)

ObjectReference = Union[ReferenceObject, ModelReference]


class GlobalPoseEstimator(ABC):
    @abstractmethod
    def estimate(
        self, observation: FrameObservation, model: ObjectReference
    ) -> Optional[PoseEstimate]:
        """Return a verified candidate or ``None`` when no candidate exists."""


class LocalPoseTracker(ABC):
    @abstractmethod
    def track(
        self,
        observation: FrameObservation,
        previous: PoseEstimate,
        model: ObjectReference,
    ) -> Optional[PoseEstimate]:
        """Update a prior pose, returning ``None`` on explicit tracker failure."""

    def reset(self) -> None:
        """Clear implementation-specific temporal state."""
