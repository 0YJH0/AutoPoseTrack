"""Shared masked-crop preprocessing for references and proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image


@dataclass(frozen=True)
class CropConfig:
    expansion: float = 1.2
    size: int = 224
    background: str = "mean"


def mask_bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def preprocess_masked_crop(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    config: CropConfig,
) -> npt.NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise ValueError("image must be RGB and mask must share its spatial shape")
    x1, y1, x2, y2 = mask_bbox(mask)
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(width, height) * config.expansion
    left = max(0, int(np.floor(cx - side / 2)))
    top = max(0, int(np.floor(cy - side / 2)))
    right = min(image.shape[1], int(np.ceil(cx + side / 2)))
    bottom = min(image.shape[0], int(np.ceil(cy + side / 2)))
    crop = image[top:bottom, left:right].copy()
    crop_mask = mask[top:bottom, left:right]
    if config.background == "zero":
        fill = np.zeros(3, dtype=np.uint8)
    elif config.background == "mean":
        pixels = crop[crop_mask]
        fill = np.asarray(pixels.mean(axis=0) if len(pixels) else 0, dtype=np.uint8)
    else:
        raise ValueError("background must be zero or mean")
    crop[~crop_mask] = fill
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(
        Image.fromarray(crop).resize((config.size, config.size), resampling.BILINEAR)
    )
