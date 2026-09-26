"""PyTorch CUDA implementation of the dense motion stage."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.types import SaliencyConfig

from .base import DenseMotionBackend, DenseMotionOutput


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "TorchCudaMotionBackend requires PyTorch with CUDA support"
        ) from error
    return torch


class TorchCudaMotionBackend(DenseMotionBackend):
    """Compute dense geometry and saliency on CUDA, including transfer timing."""

    def __init__(self, device: str = "cuda:0") -> None:
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TorchCudaMotionBackend device must be CUDA")

    @staticmethod
    def _threshold(score: Any, config: SaliencyConfig, torch: Any) -> Any:
        finite = score[torch.isfinite(score)]
        if finite.numel() == 0:
            return torch.tensor(float("inf"), device=score.device)
        if config.threshold_type == "mad":
            median = torch.quantile(finite, 0.5)
            mad = torch.quantile(torch.abs(finite - median), 0.5)
            threshold = median + config.mad_factor * 1.4826 * mad
        elif config.threshold_type == "percentile":
            threshold = torch.quantile(finite, config.percentile / 100.0)
        elif config.threshold_type == "fixed":
            threshold = torch.tensor(config.fixed_threshold, device=score.device)
        else:
            raise ValueError(f"unknown threshold type: {config.threshold_type}")
        minimum = torch.tensor(config.minimum_threshold, device=score.device)
        return torch.maximum(threshold, minimum)

    def process(
        self,
        observed_flow: npt.NDArray[np.floating],
        background_matrix: npt.NDArray[np.floating],
        saliency: SaliencyConfig,
    ) -> DenseMotionOutput:
        torch = _torch()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observed = torch.as_tensor(
            np.asarray(observed_flow, dtype=np.float32), device=self.device
        )
        matrix = torch.as_tensor(
            np.asarray(background_matrix, dtype=np.float32), device=self.device
        )
        height, width = observed.shape[:2]
        yy, xx = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(xx)
        points = torch.stack((xx, yy, ones), dim=-1)
        transformed_h = points @ matrix.T
        denominator = transformed_h[..., 2]
        safe = torch.abs(denominator) > 1e-8
        safe_denominator = torch.where(safe, denominator, torch.ones_like(denominator))
        transformed = transformed_h[..., :2] / safe_denominator[..., None]
        transformed = torch.where(safe[..., None], transformed, points[..., :2])
        background = transformed - points[..., :2]
        residual = observed - background
        raw_magnitude = torch.linalg.vector_norm(observed, dim=2)
        background_magnitude = torch.linalg.vector_norm(background, dim=2)
        residual_magnitude = torch.linalg.vector_norm(residual, dim=2)
        threshold_tensor = self._threshold(residual_magnitude, saliency, torch)
        mask = residual_magnitude > threshold_tensor
        tensors = (
            background,
            residual,
            raw_magnitude,
            background_magnitude,
            residual_magnitude,
            mask,
        )
        arrays = [tensor.detach().cpu().numpy() for tensor in tensors]
        threshold = float(threshold_tensor.detach().cpu())
        torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - started
        return DenseMotionOutput(
            background_flow=arrays[0].astype(np.float32, copy=False),
            residual_flow=arrays[1].astype(np.float32, copy=False),
            raw_magnitude=arrays[2].astype(np.float32, copy=False),
            background_magnitude=arrays[3].astype(np.float32, copy=False),
            residual_magnitude=arrays[4].astype(np.float32, copy=False),
            threshold=threshold,
            threshold_mask=arrays[5].astype(bool, copy=False),
            backend="torch_cuda",
            device=str(self.device),
            runtime_s=runtime,
        )
