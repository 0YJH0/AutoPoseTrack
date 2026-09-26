"""Readers for continuous dynamic-object motion-proposal benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from autoposetrack.motion.types import BBox


@dataclass(frozen=True)
class DynamicFrame:
    """One RGB frame and evaluator-only object boxes."""

    frame_id: int
    rgb_path: Path
    gt_boxes: tuple[BBox, ...]
    object_ids: tuple[str, ...]


def _image_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(directory.glob(suffix))
    return sorted(paths)


def _foreground_bbox(path: Path) -> Optional[BBox]:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        foreground = np.any(mask != 0, axis=2)
    else:
        foreground = mask != 0
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def load_ycbineoat(sequence_dir: Path) -> list[DynamicFrame]:
    """Load a YCBInEOAT sequence; masks are never exposed to the proposer."""

    rgb_paths = _image_paths(sequence_dir / "rgb")
    if not rgb_paths:
        raise FileNotFoundError(f"no RGB images in {sequence_dir / 'rgb'}")
    mask_dirs = [sequence_dir / "gt_mask", sequence_dir / "masks"]
    mask_dir = next((path for path in mask_dirs if path.is_dir()), None)
    frames: list[DynamicFrame] = []
    for ordinal, rgb_path in enumerate(rgb_paths):
        box = None
        if mask_dir is not None:
            candidates = [mask_dir / f"{rgb_path.stem}{suffix}" for suffix in (".png", ".jpg")]
            mask_path = next((path for path in candidates if path.exists()), None)
            if mask_path is not None:
                box = _foreground_bbox(mask_path)
        frames.append(
            DynamicFrame(
                frame_id=ordinal,
                rgb_path=rgb_path,
                gt_boxes=() if box is None else (box,),
                object_ids=() if box is None else (sequence_dir.name,),
            )
        )
    return frames


def _box_xyxy(value: object) -> Optional[BBox]:
    if isinstance(value, dict):
        if {"x", "y", "w", "h"} <= set(value):
            x, y, w, h = (float(value[key]) for key in ("x", "y", "w", "h"))
            return (x, y, x + w, y + h)
        for keys in (("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax")):
            if set(keys) <= set(value):
                return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x, y, third, fourth = map(float, value)
        # HOT3D boxes use [xmin, ymin, xmax, ymax].
        return (x, y, third, fourth)
    return None


def _hot3d_objects(payload: object, stream_id: str) -> tuple[tuple[BBox, ...], tuple[str, ...]]:
    if isinstance(payload, dict):
        objects = (
            (fallback_id, item)
            for fallback_id, value in payload.items()
            for item in (value if isinstance(value, list) else [value])
        )
    elif isinstance(payload, list):
        objects = ((str(index), value) for index, value in enumerate(payload))
    else:
        return (), ()
    boxes: list[BBox] = []
    object_ids: list[str] = []
    for fallback_id, raw in objects:
        if not isinstance(raw, dict):
            continue
        object_id = str(raw.get("object_bop_id", raw.get("object_uid", fallback_id)))
        by_stream = raw.get("boxes_amodal", {})
        value = by_stream.get(stream_id) if isinstance(by_stream, dict) else None
        box = _box_xyxy(value)
        if box is not None:
            boxes.append(box)
            object_ids.append(object_id)
    return tuple(boxes), tuple(object_ids)


def load_hot3d_clip(clip_dir: Path, stream_id: str = "214-1") -> list[DynamicFrame]:
    """Load an extracted HOT3D Aria clip and its evaluator-only amodal boxes."""

    image_paths = sorted(clip_dir.glob(f"*.image_{stream_id}.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"no HOT3D stream {stream_id} images in {clip_dir}")
    frames: list[DynamicFrame] = []
    for ordinal, rgb_path in enumerate(image_paths):
        prefix = rgb_path.name.split(".image_", 1)[0]
        annotation_path = clip_dir / f"{prefix}.objects.json"
        boxes: tuple[BBox, ...] = ()
        object_ids: tuple[str, ...] = ()
        if annotation_path.exists():
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            boxes, object_ids = _hot3d_objects(payload, stream_id)
        try:
            frame_id = int(prefix)
        except ValueError:
            frame_id = ordinal
        frames.append(DynamicFrame(frame_id, rgb_path, boxes, object_ids))
    return frames
