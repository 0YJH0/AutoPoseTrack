"""Adapter for the pinned Gen6D reference-based RGB development baseline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from autoposetrack.contracts import (
    FrameObservation,
    PoseEstimate,
    PoseMode,
    ReferenceObject,
)
from autoposetrack.geometry import project_to_se3
from autoposetrack.pose.interfaces import GlobalPoseEstimator, ObjectReference


class Gen6DAdapter(GlobalPoseEstimator):
    """Use posed RGB references through a preprocessed Gen6D database.

    Gen6D performs object onboarding (reference selection, SfM point cloud and
    normalization) in its own database abstraction. The neutral project
    contract still carries the original posed references, while
    ``backend_metadata['gen6d_database']`` names the corresponding upstream
    database. No CAD mesh is passed to inference.
    """

    def __init__(
        self,
        reference: ReferenceObject,
        config_path: Path = Path("configs/gen6d_pretrain.yaml"),
        split_type: str = "all",
        translation_scale_to_m: Optional[float] = None,
    ):
        reference.validate()
        database_name = reference.backend_metadata.get("gen6d_database")
        if not isinstance(database_name, str) or not database_name:
            raise ValueError("ReferenceObject requires backend_metadata.gen6d_database")
        scale = (
            translation_scale_to_m
            if translation_scale_to_m is not None
            else reference.backend_metadata.get("translation_scale_to_m")
        )
        if scale is None or float(scale) <= 0:
            raise ValueError("Gen6D metric output requires translation_scale_to_m > 0")
        try:
            from dataset.database import parse_database_name
            from estimator import name2estimator
            from utils.base_utils import load_cfg
        except ImportError as error:
            raise RuntimeError(
                "Gen6D is unavailable; run in the dedicated Gen6D environment "
                "with third_party/Gen6D on PYTHONPATH"
            ) from error
        config = load_cfg(str(config_path))
        self.reference = reference
        self.translation_scale_to_m = float(scale)
        self.database_name = database_name
        self._estimator = name2estimator[config["type"]](config)
        self._estimator.build(parse_database_name(database_name), split_type=split_type)

    def estimate(
        self, observation: FrameObservation, model: ObjectReference
    ) -> Optional[PoseEstimate]:
        observation.validate()
        if not isinstance(model, ReferenceObject):
            raise TypeError("Gen6D requires posed reference images, not a CAD model")
        if model.object_id != observation.object_id or model is not self.reference:
            raise ValueError("observation/reference object mismatch")
        started = time.perf_counter()
        raw_pose, intermediate = self._estimator.predict(
            observation.rgb, observation.camera_matrix
        )
        runtime = time.perf_counter() - started
        raw = np.asarray(raw_pose, dtype=np.float64)
        if raw.shape == (3, 4):
            pose = np.eye(4, dtype=np.float64)
            pose[:3] = raw
        elif raw.shape == (4, 4):
            pose = raw.copy()
        else:
            raise ValueError(f"Gen6D returned pose with shape {raw.shape}")
        pose[:3, 3] *= self.translation_scale_to_m
        pose, residual = project_to_se3(pose)
        diagnostics = self._diagnostics(intermediate)
        diagnostics.update(
            {
                "backend": "gen6d",
                "database": self.database_name,
                "rotation_orthogonality_residual": residual,
                "translation_scale_to_m": self.translation_scale_to_m,
            }
        )
        estimate = PoseEstimate(
            pose,
            diagnostics.get("selector_max_score"),
            PoseMode.GLOBAL,
            runtime,
            diagnostics,
        )
        estimate.validate()
        return estimate

    @staticmethod
    def _diagnostics(intermediate: Any) -> dict[str, Any]:
        if not isinstance(intermediate, dict):
            return {}
        result: dict[str, Any] = {}
        scores = intermediate.get("sel_scores")
        if scores is not None:
            values = np.asarray(scores, dtype=np.float64)
            if values.size and np.isfinite(values).all():
                result["selector_max_score"] = float(values.max())
                result["selector_score_margin"] = float(
                    values.max() - np.partition(values.reshape(-1), -2)[-2]
                ) if values.size > 1 else 0.0
        for source, target in (
            ("det_scale_r2q", "detected_scale"),
            ("sel_angle_r2q", "selected_inplane_angle"),
        ):
            value = intermediate.get(source)
            if value is not None and np.asarray(value).size == 1:
                result[target] = float(np.asarray(value).reshape(-1)[0])
        refine_poses = intermediate.get("refine_poses")
        if refine_poses is not None:
            result["refinement_steps"] = len(refine_poses) - 1
        return result
