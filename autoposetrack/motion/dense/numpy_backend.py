"""NumPy reference implementation for dense motion processing."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.background import background_flow_from_matrix
from autoposetrack.motion.residual import flow_magnitude, residual_flow
from autoposetrack.motion.saliency import adaptive_threshold
from autoposetrack.motion.types import SaliencyConfig

from .base import DenseMotionBackend, DenseMotionOutput


class NumpyDenseMotionBackend(DenseMotionBackend):
    def process(
        self,
        observed_flow: npt.NDArray[np.floating],
        background_matrix: npt.NDArray[np.floating],
        saliency: SaliencyConfig,
    ) -> DenseMotionOutput:
        started = time.perf_counter()
        height, width = observed_flow.shape[:2]
        background = background_flow_from_matrix(
            background_matrix, (height, width)
        )
        residual = residual_flow(observed_flow, background)
        raw_magnitude = flow_magnitude(observed_flow)
        background_magnitude = flow_magnitude(background)
        residual_magnitude = flow_magnitude(residual)
        threshold = adaptive_threshold(residual_magnitude, saliency)
        mask = residual_magnitude > threshold
        return DenseMotionOutput(
            background_flow=background,
            residual_flow=residual,
            raw_magnitude=raw_magnitude,
            background_magnitude=background_magnitude,
            residual_magnitude=residual_magnitude,
            threshold=threshold,
            threshold_mask=mask,
            backend="numpy_cpu",
            device="cpu",
            runtime_s=time.perf_counter() - started,
        )
