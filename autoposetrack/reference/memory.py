"""One shared reference memory for initialization and later verification."""

from __future__ import annotations

from dataclasses import dataclass

from .encoder import ReferenceEncoder
from .preprocessing import CropConfig, preprocess_masked_crop
from .types import EncodedReference, ReferenceFrame


@dataclass(frozen=True)
class ReferenceMemory:
    references: tuple[EncodedReference, ...]

    @classmethod
    def build(cls, frames: list[ReferenceFrame], encoder: ReferenceEncoder, crop: CropConfig):
        crops = [preprocess_masked_crop(item.image, item.mask, crop) for item in frames]
        globals_ = encoder.encode_global(crops)
        return cls(tuple(EncodedReference(frame, image, descriptor) for frame, image, descriptor in zip(frames, crops, globals_)))
