"""Reference feature encoder interfaces and DINOv2 implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt


class ReferenceEncoder(ABC):
    @abstractmethod
    def encode_global(self, images: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Return L2-normalized descriptors with shape (B, D)."""

    def encode_local(self, images: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        raise NotImplementedError


class DINOv2ReferenceEncoder(ReferenceEncoder):
    """DINOv2 encoder loaded from the official torch hub repository."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cuda:0",
        checkpoint: Optional[str] = None,
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv2 requires PyTorch") from error
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"DINOv2 requested {device}, but CUDA is unavailable")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state.get("state_dict", state))
        self.model = self.model.to(self.device).eval()

    def _batch(self, images):
        torch = self.torch
        tensors = []
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(3, 1, 1)
        for image in images:
            tensor = torch.from_numpy(np.array(image, copy=True, order="C"))
            tensor = tensor.to(self.device).permute(2, 0, 1).float() / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors)

    def encode_global(self, images):
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        torch = self.torch
        descriptors = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_fp16,
                ):
                    features = self.model.forward_features(
                        self._batch(images[start : start + self.batch_size])
                    )
                    descriptor = features["x_norm_clstoken"]
                    descriptor = torch.nn.functional.normalize(descriptor, dim=1)
                descriptors.append(descriptor.float().cpu())
        return torch.cat(descriptors).numpy().astype(np.float32)

    def encode_local(self, images):
        if not images:
            return np.empty((0, 0, 0), dtype=np.float32)
        torch = self.torch
        with torch.inference_mode():
            features = self.model.forward_features(self._batch(images))
            patches = features["x_norm_patchtokens"]
            patches = torch.nn.functional.normalize(patches, dim=2)
        return patches.float().cpu().numpy().astype(np.float32)
