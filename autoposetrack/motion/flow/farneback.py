"""CPU Farneback optical flow baseline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.types import OpticalFlowConfig

from .base import OpticalFlowEstimator


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Farneback motion proposals require the 'motion' optional dependency"
        ) from error
    return cv2


class FarnebackFlowEstimator(OpticalFlowEstimator):
    def __init__(self, config: OpticalFlowConfig) -> None:
        self.config = config

    def estimate(
        self, prev_rgb: npt.NDArray[np.uint8], curr_rgb: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        if prev_rgb.shape != curr_rgb.shape or prev_rgb.ndim != 3:
            raise ValueError("RGB frames must have identical (H, W, 3) shapes")
        if prev_rgb.dtype != np.uint8 or curr_rgb.dtype != np.uint8:
            raise ValueError("RGB frames must use uint8 values")
        cv2 = _cv2()
        previous = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        current = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            self.config.pyr_scale,
            self.config.levels,
            self.config.winsize,
            self.config.iterations,
            self.config.poly_n,
            self.config.poly_sigma,
            0,
        )
        return np.asarray(flow, dtype=np.float32)
