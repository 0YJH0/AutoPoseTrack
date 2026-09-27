"""Cosine reference matching over class-agnostic mask proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoposetrack.proposal import SegmentationProposal

from .encoder import ReferenceEncoder
from .memory import ReferenceMemory
from .preprocessing import CropConfig, preprocess_masked_crop
from .types import InitializationResult, ProposalMatch


@dataclass(frozen=True)
class MatchingConfig:
    aggregation: str = "max"
    top_k_average: int = 2
    acceptance_score: float = 0.45
    acceptance_margin: float = 0.02


class GlobalReferenceMatcher:
    def __init__(self, encoder: ReferenceEncoder, crop: CropConfig, config: MatchingConfig):
        self.encoder, self.crop, self.config = encoder, crop, config

    def match(self, image, proposals: list[SegmentationProposal], memory: ReferenceMemory) -> InitializationResult:
        if not proposals:
            return InitializationResult(False, None, None, 0.0, 0.0, None, 0.0, 0.0, 0.0, ())
        crops = [preprocess_masked_crop(image, proposal.mask, self.crop) for proposal in proposals]
        query = self.encoder.encode_global(crops)
        refs = np.stack([item.global_descriptor for item in memory.references])
        similarities = query @ refs.T
        if self.config.aggregation == "max":
            scores = similarities.max(axis=1)
        elif self.config.aggregation == "mean":
            scores = similarities.mean(axis=1)
        elif self.config.aggregation == "topk_mean":
            count = min(self.config.top_k_average, similarities.shape[1])
            scores = np.sort(similarities, axis=1)[:, -count:].mean(axis=1)
        else:
            raise ValueError("aggregation must be max, mean, or topk_mean")
        order = np.argsort(-scores)
        matches = []
        for proposal_id in order:
            reference_index = int(np.argmax(similarities[proposal_id]))
            matches.append(ProposalMatch(int(proposal_id), float(scores[proposal_id]), float(scores[proposal_id]), 0.0, memory.references[reference_index].frame.reference_id))
        best = matches[0]
        second = matches[1].score if len(matches) > 1 else -1.0
        margin = best.score - second
        accepted = best.score >= self.config.acceptance_score and margin >= self.config.acceptance_margin
        selected = proposals[best.proposal_id]
        confidence = float(np.clip(0.5 * (best.score + 1.0) * min(1.0, max(0.0, margin) / max(self.config.acceptance_margin, 1e-6)), 0.0, 1.0))
        return InitializationResult(accepted, selected.mask if accepted else None, selected.bbox_xyxy if accepted else None, best.score, confidence, best.matched_reference_id, best.score, second, margin, tuple(matches))
