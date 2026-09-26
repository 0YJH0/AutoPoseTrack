"""Deterministic SE(3) perturbations for empirical recoverable-basin trials."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from autoposetrack.geometry.se3 import validate_transform


@dataclass(frozen=True)
class SE3Perturbation:
    trial_id: str
    rotation_deg_xyz: tuple[float, float, float]
    translation_m_xyz: tuple[float, float, float]

    def matrix(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.rotation_deg_xyz)
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = rotation_z @ rotation_y @ rotation_x
        delta[:3, 3] = self.translation_m_xyz
        return validate_transform(delta)


def perturb_pose_object_frame(
    pose: np.ndarray, perturbation: SE3Perturbation
) -> np.ndarray:
    """Right-compose a perturbation expressed in the object's local frame."""
    return validate_transform(validate_transform(pose) @ perturbation.matrix())


def make_perturbation_grid(
    rotation_degrees: Iterable[float],
    translation_metres: Iterable[float],
) -> tuple[SE3Perturbation, ...]:
    """Create axis-aligned and coupled trials, including the identity once."""
    rotations = tuple(float(value) for value in rotation_degrees)
    translations = tuple(float(value) for value in translation_metres)
    if not rotations or not translations:
        raise ValueError("perturbation levels must be non-empty")
    trials: list[SE3Perturbation] = []
    seen: set[tuple[float, ...]] = set()
    for rotation, translation, axis in product(rotations, translations, range(3)):
        r = [0.0, 0.0, 0.0]
        t = [0.0, 0.0, 0.0]
        r[axis] = rotation
        t[axis] = translation
        key = (*r, *t)
        if key in seen:
            continue
        seen.add(key)
        trials.append(SE3Perturbation(f"p{len(trials):04d}", tuple(r), tuple(t)))
    return tuple(trials)
