"""Adaptive residual-flow thresholding and spatial filtering."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from .types import MorphologyConfig, SaliencyConfig


def adaptive_threshold(
    score: npt.NDArray[np.floating], config: SaliencyConfig
) -> float:
    finite = np.asarray(score[np.isfinite(score)], dtype=float)
    if not len(finite):
        return float("inf")
    if config.threshold_type == "mad":
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        threshold = median + config.mad_factor * 1.4826 * mad
    elif config.threshold_type == "percentile":
        threshold = float(np.percentile(finite, config.percentile))
    elif config.threshold_type == "fixed":
        threshold = config.fixed_threshold
    else:
        raise ValueError(f"unknown threshold type: {config.threshold_type}")
    return max(float(threshold), config.minimum_threshold)


def _fill_holes(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    import cv2

    height, width = mask.shape
    flood = np.pad(mask, 1, mode="constant", constant_values=0)
    flood_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    return mask | holes


def motion_mask(
    score: npt.NDArray[np.floating],
    saliency: SaliencyConfig,
    morphology: MorphologyConfig,
) -> Tuple[npt.NDArray[np.bool_], float]:
    threshold = adaptive_threshold(score, saliency)
    mask = np.asarray(score > threshold, dtype=np.uint8) * 255
    if morphology.enabled:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("morphology requires the 'motion' dependency") from error
        if morphology.median_kernel > 1:
            mask = cv2.medianBlur(mask, morphology.median_kernel)
        if morphology.open_kernel > 1:
            kernel = np.ones(
                (morphology.open_kernel, morphology.open_kernel), dtype=np.uint8
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if morphology.close_kernel > 1:
            kernel = np.ones(
                (morphology.close_kernel, morphology.close_kernel), dtype=np.uint8
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if morphology.fill_holes and np.any(mask):
            mask = _fill_holes(mask)
    return mask.astype(bool), threshold
