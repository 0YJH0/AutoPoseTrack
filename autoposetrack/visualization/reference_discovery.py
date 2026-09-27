"""Debug visualization for reference-guided full-frame discovery."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def discovery_panel(reference_rgb, reference_mask, query_rgb, proposals, result, gt_mask=None, top_k=5):
    reference = reference_rgb.copy()
    reference[~reference_mask] = (reference[~reference_mask] * 0.25).astype(np.uint8)
    query = query_rgb.copy()
    overlay = Image.fromarray(query)
    draw = ImageDraw.Draw(overlay)
    palette = ((255, 40, 40), (255, 160, 20), (255, 230, 40), (80, 190, 255), (200, 80, 255))
    for rank, match in enumerate(result.matches[:top_k]):
        proposal = proposals[match.proposal_id]
        color = palette[rank % len(palette)]
        draw.rectangle(proposal.bbox_xyxy, outline=color, width=3)
        draw.text((proposal.bbox_xyxy[0] + 2, proposal.bbox_xyxy[1] + 2), f"#{rank+1} {match.score:.3f}", fill=color)
    if gt_mask is not None and np.any(gt_mask):
        ys, xs = np.nonzero(gt_mask)
        draw.rectangle((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), outline=(40, 255, 40), width=2)
        draw.text((xs.min() + 2, ys.min() + 2), "GT evaluator only", fill=(40, 255, 40))
    reference = np.asarray(Image.fromarray(reference).resize((query.shape[1], query.shape[0])))
    canvas = np.concatenate((reference, np.asarray(overlay)), axis=1)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((4, 4), "Reference (masked)", fill=(255, 255, 255))
    draw.text((query.shape[1] + 4, 4), f"Full-frame Top-{top_k} | accepted={result.success} margin={result.score_margin:.3f}", fill=(255, 255, 255))
    return np.asarray(image)
