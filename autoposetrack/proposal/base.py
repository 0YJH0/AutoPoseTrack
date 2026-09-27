"""Class-agnostic segmentation proposal contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SegmentationProposal:
    mask: npt.NDArray[np.bool_]
    bbox_xyxy: tuple[float, float, float, float]
    area: int
    segmentation_score: float
    stability_score: float
    source_id: int


class ProposalInvocation(str, Enum):
    INITIALIZATION = "initialization"
    RELOCALIZATION = "relocalization"


class ProposalGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        image: npt.NDArray[np.uint8],
        invocation: ProposalInvocation = ProposalInvocation.INITIALIZATION,
    ) -> list[SegmentationProposal]:
        """Generate class-agnostic masks from the full RGB frame."""

    @staticmethod
    def validate_invocation(invocation: ProposalInvocation) -> None:
        if invocation not in {
            ProposalInvocation.INITIALIZATION,
            ProposalInvocation.RELOCALIZATION,
        }:
            raise ValueError(
                "full-frame proposals are allowed only for initialization or "
                "relocalization"
            )
