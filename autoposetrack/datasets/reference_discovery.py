"""Evaluator-only mask loaders for reference-guided discovery benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image


@dataclass(frozen=True)
class DiscoveryFrame:
    frame_id: int
    rgb_path: Path
    object_id: str
    load_gt_mask: Callable[[], npt.NDArray[np.bool_]]

    def rgb(self) -> npt.NDArray[np.uint8]:
        return np.asarray(Image.open(self.rgb_path).convert("RGB"))


def load_ycbineoat_discovery(sequence_dir: Path, object_id: Optional[str] = None) -> list[DiscoveryFrame]:
    rgb_paths = sorted((sequence_dir / "rgb").glob("*"))
    rgb_paths = [path for path in rgb_paths if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    mask_dir = sequence_dir / "gt_mask"
    frames = []
    for index, rgb_path in enumerate(rgb_paths):
        mask_path = next((mask_dir / f"{rgb_path.stem}{suffix}" for suffix in (".png", ".jpg") if (mask_dir / f"{rgb_path.stem}{suffix}").exists()), None)
        if mask_path is None:
            continue
        def load(path=mask_path):
            value = np.asarray(Image.open(path))
            return np.any(value != 0, axis=2) if value.ndim == 3 else value != 0
        frames.append(DiscoveryFrame(index, rgb_path, object_id or sequence_dir.name, load))
    return frames


def decode_hot3d_rle(payload: dict) -> npt.NDArray[np.bool_]:
    """Decode HOT3D's row-major ``[start, length, ...]`` representation."""
    flat = np.zeros(int(payload["height"]) * int(payload["width"]), dtype=bool)
    runs = payload["rle"]
    if len(runs) % 2:
        raise ValueError("HOT3D RLE must contain start/length pairs")
    for start, length in zip(runs[0::2], runs[1::2]):
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > flat.size:
            raise ValueError("HOT3D RLE run is outside the image")
        flat[start : start + length] = True
    return flat.reshape(int(payload["height"]), int(payload["width"]))


def load_hot3d_discovery(clip_dir: Path, object_id: str, stream_id: str = "214-1") -> list[DiscoveryFrame]:
    frames = []
    for rgb_path in sorted(clip_dir.glob(f"*.image_{stream_id}.jpg")):
        prefix = rgb_path.name.split(".image_", 1)[0]
        objects = json.loads((clip_dir / f"{prefix}.objects.json").read_text(encoding="utf-8"))
        target = None
        for values in objects.values():
            for item in values:
                if str(item.get("object_bop_id")) == str(object_id) and stream_id in item.get("masks_modal", {}):
                    target = item["masks_modal"][stream_id]
                    break
            if target is not None:
                break
        if target is None:
            continue
        frames.append(DiscoveryFrame(int(prefix), rgb_path, str(object_id), lambda value=target: decode_hot3d_rle(value)))
    return frames
