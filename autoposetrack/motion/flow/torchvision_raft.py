"""CUDA RAFT optical flow backed by torchvision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from autoposetrack.motion.types import OpticalFlowConfig

from .base import OpticalFlowEstimator


class TorchvisionRaftFlowEstimator(OpticalFlowEstimator):
    """Estimate previous-to-current flow with pretrained RAFT.

    Inputs are resized to a multiple of eight and optionally capped by
    ``raft_max_side``. The returned vectors and grid are restored to the
    original RGB resolution, including the required x/y displacement scaling.
    """

    def __init__(self, config: OpticalFlowConfig) -> None:
        try:
            import torch
            from torchvision.models.optical_flow import raft_large, raft_small
        except ImportError as error:
            raise RuntimeError(
                "RAFT requires torch and torchvision with optical-flow models"
            ) from error
        self.config = config
        self.torch = torch
        self.device = torch.device(config.raft_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"RAFT requested {self.device}, but CUDA is unavailable")
        constructor = raft_small if config.raft_variant == "small" else raft_large
        # torchvision 0.12 uses pretrained=; newer releases retain compatibility.
        model = constructor(pretrained=config.raft_pretrained, progress=True)
        if config.raft_checkpoint:
            checkpoint = torch.load(Path(config.raft_checkpoint), map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

    def _inference_shape(self, height: int, width: int) -> tuple[int, int]:
        scale = min(1.0, self.config.raft_max_side / max(height, width))
        resized_h = max(8, int(round(height * scale / 8.0)) * 8)
        resized_w = max(8, int(round(width * scale / 8.0)) * 8)
        return resized_h, resized_w

    def estimate(
        self, prev_rgb: npt.NDArray[np.uint8], curr_rgb: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        if prev_rgb.shape != curr_rgb.shape or prev_rgb.ndim != 3 or prev_rgb.shape[2] != 3:
            raise ValueError("RGB frames must have identical (H, W, 3) shapes")
        if prev_rgb.dtype != np.uint8 or curr_rgb.dtype != np.uint8:
            raise ValueError("RGB frames must use uint8 values")
        torch = self.torch
        functional = torch.nn.functional
        height, width = prev_rgb.shape[:2]
        resized_h, resized_w = self._inference_shape(height, width)

        def prepare(image):
            # PIL-backed arrays can be read-only; own the buffer before zero-copy conversion.
            tensor = torch.from_numpy(np.array(image, copy=True, order="C")).to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            if (resized_h, resized_w) != (height, width):
                tensor = functional.interpolate(
                    tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False
                )
            return tensor

        with torch.inference_mode():
            predictions = self.model(
                prepare(prev_rgb),
                prepare(curr_rgb),
                num_flow_updates=self.config.raft_num_flow_updates,
            )
            flow = predictions[-1]
            if (resized_h, resized_w) != (height, width):
                flow = functional.interpolate(
                    flow, size=(height, width), mode="bilinear", align_corners=False
                )
                flow[:, 0].mul_(width / resized_w)
                flow[:, 1].mul_(height / resized_h)
        return (
            flow[0].permute(1, 2, 0).contiguous().float().cpu().numpy().astype(np.float32)
        )
