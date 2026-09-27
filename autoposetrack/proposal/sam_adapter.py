"""Segment Anything automatic-mask proposal adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ProposalGenerator, SegmentationProposal
from .filtering import ProposalFilterConfig, filter_proposals


class SamProposalGenerator(ProposalGenerator):
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda:0",
        points_per_side: int = 24,
        pred_iou_thresh: float = 0.72,
        stability_score_thresh: float = 0.78,
        filter_config: ProposalFilterConfig = ProposalFilterConfig(),
    ) -> None:
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as error:
            raise RuntimeError("SAM requires the official segment-anything package") from error
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device=device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            output_mode="binary_mask",
        )
        self.filter_config = filter_config

    def generate(self, image):
        raw = self.generator.generate(image)
        proposals = []
        for source_id, item in enumerate(raw):
            x, y, width, height = map(float, item["bbox"])
            proposals.append(
                SegmentationProposal(
                    mask=np.asarray(item["segmentation"], dtype=bool),
                    bbox_xyxy=(x, y, x + width, y + height),
                    area=int(item["area"]),
                    segmentation_score=float(item.get("predicted_iou", 0.0)),
                    stability_score=float(item.get("stability_score", 0.0)),
                    source_id=source_id,
                )
            )
        return filter_proposals(proposals, image.shape[:2], self.filter_config)
