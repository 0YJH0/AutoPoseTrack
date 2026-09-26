"""Lazy adapter for the pinned upstream MegaPose RGB estimator."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from autoposetrack.contracts import (
    FrameObservation,
    ModelReference,
    PoseEstimate,
    PoseMode,
)
from autoposetrack.pose.interfaces import GlobalPoseEstimator, LocalPoseTracker


class MegaPoseRGBAdapter(GlobalPoseEstimator, LocalPoseTracker):
    """Pinned RGB-only MegaPose coarse estimator and pose refiner.

    ``estimate`` performs global coarse-to-fine pose estimation from an oracle
    box. ``track`` skips the coarse stage and refines the previous pose.  No
    depth tensor or ground-truth pose crosses this adapter boundary.
    """

    def __init__(
        self,
        model: ModelReference,
        model_name: str = "megapose-1.0-RGB-multi-hypothesis",
        renderer_workers: int = 1,
        batch_size: int = 128,
        tracking_refiner_iterations: int = 2,
    ):
        if "RGBD" in model_name or "icp" in model_name.lower():
            raise ValueError("RGB-only adapter rejects RGB-D/ICP MegaPose models")
        try:
            from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
            from megapose.utils.load_model import NAMED_MODELS, load_named_model
        except ImportError as error:
            raise RuntimeError(
                "MegaPose is unavailable; use the megapose Docker service"
            ) from error
        if model_name not in NAMED_MODELS:
            raise ValueError(f"unknown MegaPose model: {model_name}")
        if NAMED_MODELS[model_name]["requires_depth"]:
            raise ValueError("selected MegaPose model requires depth")
        self.label = f"obj_{model.object_id:06d}"
        object_dataset = RigidObjectDataset(
            [RigidObject(label=self.label, mesh_path=model.mesh_path, mesh_units="mm")]
        )
        self.model_name = model_name
        if tracking_refiner_iterations <= 0:
            raise ValueError("tracking_refiner_iterations must be positive")
        self.tracking_refiner_iterations = tracking_refiner_iterations
        self._parameters = dict(NAMED_MODELS[model_name]["inference_parameters"])
        self._estimator = load_named_model(
            model_name,
            object_dataset,
            n_workers=renderer_workers,
            bsz_images=batch_size,
        ).cuda()

    def estimate(
        self, observation: FrameObservation, model: ModelReference
    ) -> Optional[PoseEstimate]:
        observation.validate()
        if observation.object_id != model.object_id:
            raise ValueError("observation/model object ID mismatch")
        from megapose.datasets.scene_dataset import ObjectData
        from megapose.inference.types import ObservationTensor
        from megapose.inference.utils import make_detections_from_object_data

        tensor = ObservationTensor.from_numpy(
            observation.rgb, depth=None, K=observation.camera_matrix
        ).cuda()
        detection = make_detections_from_object_data(
            [ObjectData(label=self.label, bbox_modal=observation.bbox_xyxy)]
        ).cuda()
        started = time.perf_counter()
        output, extra = self._estimator.run_inference_pipeline(
            tensor, detections=detection, **self._parameters
        )
        runtime = time.perf_counter() - started
        if len(output) == 0:
            return None
        raw_pose = np.asarray(output.poses[0].detach().cpu().numpy(), dtype=np.float64)
        pose, rotation_residual = _project_to_se3(raw_pose)
        score = self._extract_score(output.infos.iloc[0])
        estimate = PoseEstimate(
            object_to_camera=pose,
            score=score,
            mode=PoseMode.GLOBAL,
            runtime_s=runtime,
            diagnostics={
                "model_name": self.model_name,
                "upstream_time_s": float(extra.get("time", runtime)),
                "rotation_orthogonality_residual": rotation_residual,
            },
        )
        estimate.validate()
        return estimate

    def track(
        self,
        observation: FrameObservation,
        previous: PoseEstimate,
        model: ModelReference,
    ) -> Optional[PoseEstimate]:
        observation.validate()
        previous.validate()
        if observation.object_id != model.object_id:
            raise ValueError("observation/model object ID mismatch")
        if previous.mode is PoseMode.NONE:
            raise ValueError("cannot refine a missing pose")

        import pandas as pd
        import torch
        from megapose.inference.types import ObservationTensor
        from megapose.utils.tensor_collection import PandasTensorCollection

        tensor = ObservationTensor.from_numpy(
            observation.rgb, depth=None, K=observation.camera_matrix
        ).cuda()
        infos = pd.DataFrame(
            [{"batch_im_id": 0, "instance_id": 0, "label": self.label}]
        )
        initial = PandasTensorCollection(
            infos=infos,
            poses=torch.as_tensor(
                previous.object_to_camera[None], dtype=torch.float32, device="cuda"
            ),
        )
        started = time.perf_counter()
        output, extra = self._estimator.run_inference_pipeline(
            tensor,
            coarse_estimates=initial,
            n_refiner_iterations=self.tracking_refiner_iterations,
            n_pose_hypotheses=1,
            run_depth_refiner=False,
        )
        runtime = time.perf_counter() - started
        if len(output) == 0:
            return None
        raw_pose = np.asarray(output.poses[0].detach().cpu().numpy(), dtype=np.float64)
        pose, rotation_residual = _project_to_se3(raw_pose)
        estimate = PoseEstimate(
            object_to_camera=pose,
            score=self._extract_score(output.infos.iloc[0]),
            mode=PoseMode.LOCAL,
            runtime_s=runtime,
            diagnostics={
                "model_name": self.model_name,
                "upstream_time_s": float(extra.get("time", runtime)),
                "rotation_orthogonality_residual": rotation_residual,
                "refiner_iterations": self.tracking_refiner_iterations,
            },
        )
        estimate.validate()
        return estimate

    def reset(self) -> None:
        """MegaPose refinement is stateless; kept for the tracker protocol."""

    def close(self) -> None:
        """Stop MegaPose renderer workers so batch jobs terminate cleanly."""
        renderers = {
            id(model.renderer): model.renderer
            for model in (self._estimator.coarse_model, self._estimator.refiner_model)
            if model is not None
        }
        for renderer in renderers.values():
            renderer.stop()

    @staticmethod
    def _extract_score(row: Any) -> Optional[float]:
        for name in ("pose_score", "refiner_logit", "coarse_logit", "score"):
            if name in row and np.isfinite(row[name]):
                return float(row[name])
        return None


def _project_to_se3(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """Project small network-output rotation drift to the closest SO(3)."""

    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("MegaPose returned an invalid 4x4 transform")
    rotation = pose[:3, :3]
    residual = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    if residual > 0.1:
        raise ValueError(f"MegaPose rotation residual is too large: {residual:.6g}")
    left, _, right_t = np.linalg.svd(rotation)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    projected = pose.copy()
    projected[:3, :3] = left @ correction @ right_t
    projected[3] = [0.0, 0.0, 0.0, 1.0]
    return projected, residual


class MegaPoseRGBEstimator(MegaPoseRGBAdapter):
    """Backward-compatible name for the RGB adapter."""
