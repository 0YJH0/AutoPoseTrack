"""Full-frame RGB motion proposals, independent of target identity and pose."""

from .motion_proposer import FullFrameMotionProposer
from .types import (
    BackgroundMotionModel,
    MotionProposal,
    MotionProposalConfig,
    MotionProposalResult,
    MotionTracklet,
)

__all__ = [
    "BackgroundMotionModel",
    "FullFrameMotionProposer",
    "MotionProposal",
    "MotionProposalConfig",
    "MotionProposalResult",
    "MotionTracklet",
]
