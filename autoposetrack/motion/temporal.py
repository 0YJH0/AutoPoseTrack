"""Short-term proposal memory; it boosts persistence but never gates first hits."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .components import bbox_iou
from .types import MotionProposal, MotionTracklet, TemporalConfig


class MotionTrackletMemory:
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self._next_id = 1
        self._tracks: Dict[int, MotionTracklet] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _match_score(
        self, proposal: MotionProposal, track: MotionTracklet, diagonal: float
    ) -> float:
        overlap = bbox_iou(proposal.bbox_xyxy, track.bbox_history[-1])
        distance = float(
            np.linalg.norm(
                np.asarray(proposal.centroid) - np.asarray(track.centroid_history[-1])
            )
        )
        distance_score = max(
            0.0,
            1.0
            - distance
            / max(self.config.association_distance_ratio * diagonal, 1.0),
        )
        if overlap < self.config.association_iou and distance_score <= 0.0:
            return -1.0
        return max(overlap, distance_score)

    def _new_track(self, proposal: MotionProposal) -> MotionTracklet:
        track = MotionTracklet(
            tracklet_id=self._next_id,
            age=1,
            hits=1,
            missed=0,
            bbox_history=[proposal.bbox_xyxy],
            centroid_history=[proposal.centroid],
            motion_history=[proposal.mean_residual_flow],
            confidence=proposal.confidence,
        )
        self._next_id += 1
        self._tracks[track.tracklet_id] = track
        return track

    def update(
        self, proposals: Sequence[MotionProposal], image_shape: Tuple[int, int]
    ) -> List[MotionProposal]:
        if not self.config.enabled:
            return list(proposals)
        diagonal = math.hypot(*image_shape)
        unmatched_tracks = set(self._tracks)
        outputs: List[MotionProposal] = []
        ordered = sorted(
            proposals, key=lambda item: item.confidence, reverse=True
        )
        for proposal in ordered:
            candidates = [
                (
                    self._match_score(proposal, self._tracks[track_id], diagonal),
                    track_id,
                )
                for track_id in unmatched_tracks
            ]
            score, track_id = max(candidates, default=(-1.0, -1))
            track = (
                self._tracks[track_id]
                if score >= 0.0
                else self._new_track(proposal)
            )
            unmatched_tracks.discard(track.tracklet_id)
            if score >= 0.0:
                track.age += 1
                track.hits += 1
                track.missed = 0
                track.bbox_history.append(proposal.bbox_xyxy)
                track.centroid_history.append(proposal.centroid)
                track.motion_history.append(proposal.mean_residual_flow)
                track.confidence = proposal.confidence
                for history in (
                    track.bbox_history,
                    track.centroid_history,
                    track.motion_history,
                ):
                    del history[: -self.config.history_length]
            persistence = min(track.hits, self.config.history_length) / float(
                self.config.history_length
            )
            outputs.append(
                proposal.with_temporal(
                    track.tracklet_id,
                    track.age,
                    persistence,
                    proposal.confidence,
                )
            )
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.age += 1
            track.missed += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed <= self.config.max_missed
        }
        return outputs

    @property
    def tracklets(self) -> Tuple[MotionTracklet, ...]:
        return tuple(self._tracks.values())
