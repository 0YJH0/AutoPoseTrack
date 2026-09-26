"""Dependency-injected adapter boundary for future RAFT/GMFlow inference."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from .base import OpticalFlowEstimator


class DenseNetworkFlowAdapter(OpticalFlowEstimator):
    """Wrap an external dense-flow callable without coupling proposal logic.

    The callable owns device placement, normalization, padding, and checkpoint
    loading. It must return previous-to-current pixel displacement in RGB image
    coordinates with shape ``(H, W, 2)``.
    """

    def __init__(
        self,
        infer: Callable[
            [npt.NDArray[np.uint8], npt.NDArray[np.uint8]],
            npt.NDArray[np.floating],
        ],
        backend_name: str = "external_dense_flow",
    ) -> None:
        self.infer = infer
        self.backend_name = backend_name

    def estimate(
        self, prev_rgb: npt.NDArray[np.uint8], curr_rgb: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        flow = np.asarray(self.infer(prev_rgb, curr_rgb), dtype=np.float32)
        expected = (*prev_rgb.shape[:2], 2)
        if flow.shape != expected or not np.isfinite(flow).all():
            raise ValueError(
                f"{self.backend_name} must return finite flow of shape {expected}"
            )
        return flow
