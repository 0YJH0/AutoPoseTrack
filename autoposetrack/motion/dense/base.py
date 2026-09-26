"""Backend contract for the dense, data-parallel motion stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.types import SaliencyConfig


@dataclass(frozen=True)
class DenseMotionOutput:
    background_flow: npt.NDArray[np.float32]
    residual_flow: npt.NDArray[np.float32]
    raw_magnitude: npt.NDArray[np.float32]
    background_magnitude: npt.NDArray[np.float32]
    residual_magnitude: npt.NDArray[np.float32]
    threshold: float
    threshold_mask: npt.NDArray[np.bool_]
    backend: str
    device: str
    runtime_s: float


class DenseMotionBackend(ABC):
    @abstractmethod
    def process(
        self,
        observed_flow: npt.NDArray[np.floating],
        background_matrix: npt.NDArray[np.floating],
        saliency: SaliencyConfig,
    ) -> DenseMotionOutput:
        """Run the dense stage, including host/device transfers in runtime."""
