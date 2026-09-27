"""Full-frame class-agnostic segmentation proposals."""

from .base import ProposalGenerator, ProposalInvocation, SegmentationProposal
from .fastsam_adapter import FastSamProposalGenerator
from .filtering import ProposalFilterConfig, filter_proposals, mask_iou
from .sam_adapter import SamProposalGenerator

__all__ = [
    "FastSamProposalGenerator",
    "ProposalFilterConfig",
    "ProposalGenerator",
    "ProposalInvocation",
    "SamProposalGenerator",
    "SegmentationProposal",
    "filter_proposals",
    "mask_iou",
]
