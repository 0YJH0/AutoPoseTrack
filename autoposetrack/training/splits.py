"""Deterministic, sequence-level splits that reject temporal leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SequenceSplit:
    train: npt.NDArray[np.int64]
    validation: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]

    def validate(self, sequence_ids: npt.ArrayLike) -> None:
        ids = np.asarray(sequence_ids, dtype=np.str_)
        sets = [set(ids[index].tolist()) for index in (self.train, self.validation, self.test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("sequence leakage detected between splits")
        indices = np.concatenate((self.train, self.validation, self.test))
        if len(indices) != len(ids) or set(indices.tolist()) != set(range(len(ids))):
            raise ValueError("split indices must cover every row exactly once")


def _sequence_score(sequence_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_by_sequence(
    sequence_ids: npt.ArrayLike,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SequenceSplit:
    ids = np.asarray(sequence_ids, dtype=np.str_)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("sequence_ids must be a non-empty 1D array")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    sequence_group: dict[str, str] = {}
    for sequence_id in sorted(set(ids.tolist())):
        score = _sequence_score(sequence_id, seed)
        if score < test_fraction:
            sequence_group[sequence_id] = "test"
        elif score < test_fraction + validation_fraction:
            sequence_group[sequence_id] = "validation"
        else:
            sequence_group[sequence_id] = "train"

    result = SequenceSplit(
        train=np.flatnonzero([sequence_group[value] == "train" for value in ids]).astype(np.int64),
        validation=np.flatnonzero([sequence_group[value] == "validation" for value in ids]).astype(np.int64),
        test=np.flatnonzero([sequence_group[value] == "test" for value in ids]).astype(np.int64),
    )
    result.validate(ids)
    if not len(result.train):
        raise ValueError("sequence split produced an empty training set")
    return result
