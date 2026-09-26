"""Residual motion after dominant-background compensation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def residual_flow(
    observed: npt.NDArray[np.floating], predicted_background: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    if observed.shape != predicted_background.shape:
        raise ValueError("observed and background flow shapes must match")
    return np.asarray(observed - predicted_background, dtype=np.float32)


def flow_magnitude(flow: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape (H, W, 2)")
    return np.linalg.norm(flow, axis=2).astype(np.float32)
