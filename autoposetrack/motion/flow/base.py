"""Replaceable optical-flow interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class OpticalFlowEstimator(ABC):
    @abstractmethod
    def estimate(
        self, prev_rgb: npt.NDArray[np.uint8], curr_rgb: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """Return dense previous-to-current flow with shape ``(H, W, 2)``."""
