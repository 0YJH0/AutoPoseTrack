"""RGB-only reliability interfaces with no evaluation GT or depth access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from autoposetrack.contracts import FrameObservation, PoseEstimate


@dataclass(frozen=True)
class ReliabilityResult:
    probability: float
    features: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")


class ReliabilityEstimator(ABC):
    @abstractmethod
    def assess(
        self,
        observation: FrameObservation,
        estimate: PoseEstimate,
        history: Sequence[PoseEstimate],
    ) -> ReliabilityResult:
        """Assess recoverability using inference-available information only."""
