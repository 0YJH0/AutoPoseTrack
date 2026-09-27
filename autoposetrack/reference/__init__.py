"""Reference encoding, memory, preprocessing, and matching."""

from .encoder import DINOv2ReferenceEncoder, ReferenceEncoder
from .global_matcher import GlobalReferenceMatcher, MatchingConfig
from .memory import ReferenceMemory
from .preprocessing import CropConfig, preprocess_masked_crop
from .types import InitializationResult, ProposalMatch, ReferenceFrame

__all__ = ["CropConfig", "DINOv2ReferenceEncoder", "GlobalReferenceMatcher", "InitializationResult", "MatchingConfig", "ProposalMatch", "ReferenceEncoder", "ReferenceFrame", "ReferenceMemory", "preprocess_masked_crop"]
