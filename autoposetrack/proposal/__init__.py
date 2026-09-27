"""Full-frame class-agnostic segmentation proposals."""

from .base import ProposalGenerator, SegmentationProposal
from .filtering import ProposalFilterConfig, filter_proposals, mask_iou
from .sam_adapter import SamProposalGenerator

__all__ = ["ProposalFilterConfig", "ProposalGenerator", "SamProposalGenerator", "SegmentationProposal", "filter_proposals", "mask_iou"]
