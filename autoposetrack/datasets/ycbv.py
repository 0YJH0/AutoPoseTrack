"""BOP-format YCB-Video reader with inference/annotation separation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
from PIL import Image

from autoposetrack.contracts import FrameObservation, ModelReference
from autoposetrack.datasets.interfaces import (
    EvaluationAnnotations,
    PoseAnnotation,
    PoseSequenceDataset,
)


@dataclass(frozen=True)
class YCBVTargetFrame:
    sequence_id: str
    frame_id: int
    object_id: int
    instance_index: int
    visibility_fraction: Optional[float]


class YCBVSingleObjectDataset(PoseSequenceDataset, EvaluationAnnotations):
    """Read one object from the public BOP YCB-V test split.

    The observation path exposes RGB, intrinsics, and an explicitly declared
    oracle annotation box. Ground-truth pose remains available only through the
    separate ``EvaluationAnnotations`` method.
    """

    def __init__(self, root: Path, object_id: int, bbox_source: str = "bbox_obj"):
        self.root = root.expanduser().resolve()
        self.object_id = int(object_id)
        self.bbox_source = bbox_source
        if bbox_source not in {"bbox_obj", "bbox_visib"}:
            raise ValueError("bbox_source must be 'bbox_obj' or 'bbox_visib'")
        if not (self.root / "test").is_dir():
            raise FileNotFoundError(f"YCB-V test directory not found: {self.root / 'test'}")
        if not (self.root / "models" / "models_info.json").is_file():
            raise FileNotFoundError("YCB-V models/models_info.json is missing")
        self._scene_cache: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        self._targets = self._build_target_index()
        if not self._targets:
            raise ValueError(f"object_id={self.object_id} is absent from the test split")

    def target_frames(
        self, sequence_ids: Optional[Sequence[str]] = None
    ) -> tuple[YCBVTargetFrame, ...]:
        if sequence_ids is None:
            return self._targets
        selected = {self._normalize_sequence_id(value) for value in sequence_ids}
        unknown = selected.difference(self.sequence_ids())
        if unknown:
            raise ValueError(f"unknown or target-free sequence IDs: {sorted(unknown)}")
        return tuple(item for item in self._targets if item.sequence_id in selected)

    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.sequence_id for item in self._targets))

    def observations(self, sequence_id: str) -> Iterator[FrameObservation]:
        normalized = self._normalize_sequence_id(sequence_id)
        for target in self.target_frames([normalized]):
            yield self.observation(target)

    def observation(self, target: YCBVTargetFrame) -> FrameObservation:
        _, scene_info, scene_camera = self._load_scene(target.sequence_id)
        key = str(target.frame_id)
        info = scene_info[key][target.instance_index]
        x, y, width, height = (float(value) for value in info[self.bbox_source])
        rgb_path = self.root / "test" / target.sequence_id / "rgb" / f"{target.frame_id:06d}.png"
        with Image.open(rgb_path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        image_height, image_width = rgb.shape[:2]
        bbox = np.asarray(
            [
                max(0.0, x),
                max(0.0, y),
                min(float(image_width), x + width),
                min(float(image_height), y + height),
            ],
            dtype=np.float64,
        )
        camera = scene_camera[key]
        observation = FrameObservation(
            sequence_id=target.sequence_id,
            frame_id=target.frame_id,
            object_id=target.object_id,
            rgb=rgb,
            camera_matrix=np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3),
            bbox_xyxy=bbox,
            metadata={
                "dataset": "ycbv",
                "split": "test",
                "instance_index": target.instance_index,
                "detection_source": f"oracle_{self.bbox_source}",
            },
        )
        observation.validate()
        return observation

    def annotation(
        self, sequence_id: str, frame_id: int, object_id: int
    ) -> PoseAnnotation:
        normalized = self._normalize_sequence_id(sequence_id)
        matches = [
            item
            for item in self._targets
            if item.sequence_id == normalized
            and item.frame_id == int(frame_id)
            and item.object_id == int(object_id)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one target instance, found {len(matches)}")
        target = matches[0]
        scene_gt, _, _ = self._load_scene(normalized)
        payload = scene_gt[str(frame_id)][target.instance_index]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(payload["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        transform[:3, 3] = np.asarray(payload["cam_t_m2c"], dtype=np.float64) / 1000.0
        return PoseAnnotation(
            sequence_id=normalized,
            frame_id=int(frame_id),
            object_id=int(object_id),
            object_to_camera=transform,
            visibility_fraction=target.visibility_fraction,
        )

    def model_reference(self, object_id: int) -> ModelReference:
        if int(object_id) != self.object_id:
            raise ValueError(f"dataset is restricted to object_id={self.object_id}")
        info = self._load_json(self.root / "models" / "models_info.json")
        diameter_m = float(info[str(self.object_id)]["diameter"]) / 1000.0
        mesh_path = self.root / "models" / f"obj_{self.object_id:06d}.ply"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"object mesh not found: {mesh_path}")
        return ModelReference(self.object_id, mesh_path, diameter_m)

    def is_symmetric(self) -> bool:
        info = self._load_json(self.root / "models" / "models_info.json")[str(self.object_id)]
        return bool(info.get("symmetries_discrete") or info.get("symmetries_continuous"))

    def _build_target_index(self) -> tuple[YCBVTargetFrame, ...]:
        targets: list[YCBVTargetFrame] = []
        for scene_dir in sorted((self.root / "test").iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.isdigit():
                continue
            scene_gt, scene_info, _ = self._load_scene(scene_dir.name)
            for frame_key in sorted(scene_gt, key=int):
                for instance_index, pose in enumerate(scene_gt[frame_key]):
                    if int(pose["obj_id"]) != self.object_id:
                        continue
                    visibility = scene_info[frame_key][instance_index].get("visib_fract")
                    targets.append(
                        YCBVTargetFrame(
                            sequence_id=scene_dir.name,
                            frame_id=int(frame_key),
                            object_id=self.object_id,
                            instance_index=instance_index,
                            visibility_fraction=None if visibility is None else float(visibility),
                        )
                    )
        return tuple(targets)

    def _load_scene(self, sequence_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if sequence_id not in self._scene_cache:
            scene_dir = self.root / "test" / sequence_id
            self._scene_cache[sequence_id] = (
                self._load_json(scene_dir / "scene_gt.json"),
                self._load_json(scene_dir / "scene_gt_info.json"),
                self._load_json(scene_dir / "scene_camera.json"),
            )
        return self._scene_cache[sequence_id]

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _normalize_sequence_id(value: str) -> str:
        return f"{int(value):06d}"
