"""Configurable state transitions for autonomous tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackingState(str, Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    UNCERTAIN = "uncertain"
    LOST = "lost"
    RELOCALIZING = "relocalizing"


@dataclass(frozen=True)
class StateManagerConfig:
    track_threshold: float
    lost_threshold: float
    consecutive_lost_frames: int

    def validate(self) -> None:
        if not 0.0 <= self.lost_threshold < self.track_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= lost < track <= 1"
            )
        if self.consecutive_lost_frames < 1:
            raise ValueError("consecutive_lost_frames must be >= 1")


class StateManager:
    def __init__(self, config: StateManagerConfig):
        config.validate()
        self.config = config
        self.state = TrackingState.UNINITIALIZED
        self._low_score_count = 0

    def request_global(self) -> TrackingState:
        if self.state not in {TrackingState.UNINITIALIZED, TrackingState.LOST}:
            raise RuntimeError(f"cannot request global pose from {self.state.value}")
        self.state = TrackingState.RELOCALIZING
        return self.state

    def global_result(self, accepted: bool) -> TrackingState:
        if self.state != TrackingState.RELOCALIZING:
            raise RuntimeError("global_result is valid only while relocalizing")
        self._low_score_count = 0
        self.state = TrackingState.TRACKING if accepted else TrackingState.LOST
        return self.state

    def local_score(self, probability: float) -> TrackingState:
        if self.state not in {TrackingState.TRACKING, TrackingState.UNCERTAIN}:
            raise RuntimeError(f"cannot consume local score from {self.state.value}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

        if probability > self.config.track_threshold:
            self._low_score_count = 0
            self.state = TrackingState.TRACKING
        elif probability > self.config.lost_threshold:
            self._low_score_count = 0
            self.state = TrackingState.UNCERTAIN
        else:
            self._low_score_count += 1
            self.state = (
                TrackingState.LOST
                if self._low_score_count >= self.config.consecutive_lost_frames
                else TrackingState.UNCERTAIN
            )
        return self.state

    def reset(self) -> None:
        self.state = TrackingState.UNINITIALIZED
        self._low_score_count = 0

