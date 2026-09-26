"""Compute-aware routing from predicted future recoverability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoutingAction(str, Enum):
    LOCAL = "local"
    WAIT = "wait"
    GLOBAL = "global"


@dataclass(frozen=True)
class CostSensitiveRoutingConfig:
    local_cost: float
    global_cost: float
    failure_cost: float
    track_probability: float = 0.7
    lost_probability: float = 0.3
    consecutive_lost_frames: int = 2
    global_cooldown_frames: int = 10

    def validate(self) -> None:
        if min(self.local_cost, self.global_cost, self.failure_cost) < 0:
            raise ValueError("routing costs must be non-negative")
        if not 0 <= self.lost_probability < self.track_probability <= 1:
            raise ValueError("probabilities must satisfy 0 <= lost < track <= 1")
        if self.consecutive_lost_frames < 1 or self.global_cooldown_frames < 0:
            raise ValueError("invalid routing frame counts")


class CostSensitiveRouter:
    """Choose local refinement or global recovery with hysteresis and cooldown.

    The expected local risk is ``local_cost + (1-p_recover)*failure_cost``.
    Global inference is selected only when that risk exceeds ``global_cost`` and
    low recoverability persists, preventing one-frame route oscillations.
    """

    def __init__(self, config: CostSensitiveRoutingConfig):
        config.validate()
        self.config = config
        self.low_count = 0
        self.frames_since_global = config.global_cooldown_frames

    def choose(self, recoverability: float, has_pose: bool = True) -> RoutingAction:
        if not 0 <= recoverability <= 1:
            raise ValueError("recoverability must be in [0, 1]")
        if not has_pose:
            return self._global()
        self.frames_since_global += 1
        expected_local_cost = (
            self.config.local_cost + (1.0 - recoverability) * self.config.failure_cost
        )
        global_is_cheaper = expected_local_cost > self.config.global_cost
        if recoverability >= self.config.track_probability:
            self.low_count = 0
            return RoutingAction.LOCAL
        if recoverability > self.config.lost_probability or not global_is_cheaper:
            self.low_count = 0
            return RoutingAction.WAIT
        self.low_count += 1
        if self.low_count < self.config.consecutive_lost_frames:
            return RoutingAction.WAIT
        if self.frames_since_global < self.config.global_cooldown_frames:
            return RoutingAction.WAIT
        return self._global()

    def _global(self) -> RoutingAction:
        self.low_count = 0
        self.frames_since_global = 0
        return RoutingAction.GLOBAL

    def reset(self) -> None:
        self.low_count = 0
        self.frames_since_global = self.config.global_cooldown_frames
