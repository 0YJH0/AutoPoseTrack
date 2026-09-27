"""Ultralytics FastSAM full-frame proposal adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import ProposalGenerator, ProposalInvocation, SegmentationProposal
from .filtering import ProposalFilterConfig, filter_proposals


class FastSamProposalGenerator(ProposalGenerator):
    """Generate all-instance masks with FastSAM-s at a bounded resolution."""

    def __init__(
        self,
        checkpoint: str = "FastSAM-s.pt",
        device: str = "cuda:0",
        image_size: int = 640,
        confidence: float = 0.25,
        iou_threshold: float = 0.9,
        use_fp16: bool = True,
        filter_config: Optional[ProposalFilterConfig] = None,
    ) -> None:
        try:
            from ultralytics import FastSAM
        except ImportError as error:
            raise RuntimeError("FastSAM requires the ultralytics package") from error
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.model = FastSAM(checkpoint)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16
        self.filter_config = filter_config or ProposalFilterConfig()

    def generate(
        self,
        image,
        invocation: ProposalInvocation = ProposalInvocation.INITIALIZATION,
    ):
        self.validate_invocation(invocation)
        # Ultralytics treats ndarray input as OpenCV BGR, while this project uses RGB.
        source_bgr = np.ascontiguousarray(image[..., ::-1])
        results = self.model(
            source=source_bgr,
            device=self.device,
            retina_masks=True,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            half=self.use_fp16 and str(self.device).startswith("cuda"),
            verbose=False,
        )
        if not results or results[0].masks is None:
            return []
        result = results[0]
        masks = result.masks.data.detach().bool().cpu().numpy()
        boxes = result.boxes.xyxy.detach().float().cpu().numpy()
        scores = result.boxes.conf.detach().float().cpu().numpy()
        if masks.shape[1:] != image.shape[:2]:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("FastSAM mask restoration requires OpenCV") from error
            masks = np.stack(
                [
                    cv2.resize(
                        mask.astype(np.uint8),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    for mask in masks
                ]
            )
        proposals = [
            SegmentationProposal(
                mask=mask,
                bbox_xyxy=tuple(map(float, box)),
                area=int(mask.sum()),
                segmentation_score=float(score),
                stability_score=float(score),
                source_id=index,
            )
            for index, (mask, box, score) in enumerate(zip(masks, boxes, scores))
        ]
        return filter_proposals(proposals, image.shape[:2], self.filter_config)
