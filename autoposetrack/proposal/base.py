"""Class-agnostic segmentation proposal contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SegmentationProposal:
    mask: npt.NDArray[np.bool_]
    bbox_xyxy: Tuple[float, float, float, float]
    area: int
    segmentation_score: float
    stability_score: float
    source_id: int


class ProposalGenerator(ABC):
    @abstractmethod
    def generate(self, image: npt.NDArray[np.uint8]) -> list[SegmentationProposal]:
        """Generate class-agnostic masks from the full RGB frame."""
