"""Dataset contracts separating inference observations from annotations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from autoposetrack.contracts import FrameObservation, ModelReference


@dataclass(frozen=True)
class PoseAnnotation:
    sequence_id: str
    frame_id: int
    object_id: int
    object_to_camera: npt.NDArray[np.floating]
    visibility_fraction: Optional[float] = None


class PoseSequenceDataset(ABC):
    @abstractmethod
    def sequence_ids(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def observations(self, sequence_id: str) -> Iterator[FrameObservation]:
        pass

    @abstractmethod
    def model_reference(self, object_id: int) -> ModelReference:
        pass


class EvaluationAnnotations(ABC):
    """Separate interface so inference components cannot receive GT by accident."""

    @abstractmethod
    def annotation(
        self, sequence_id: str, frame_id: int, object_id: int
    ) -> PoseAnnotation:
        pass

