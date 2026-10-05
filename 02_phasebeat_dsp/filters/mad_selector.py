"""Robust temporal-MAD subcarrier ranking."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def select_by_mad(
    bandpassed_data: ArrayLike,
    top_k: int = 3,
    candidate_subcarriers: ArrayLike | None = None,
) -> tuple[int, NDArray[np.int64], NDArray[np.float64]]:
    """Rank filtered channels by temporal MAD and choose the middle of top-k.

    MAD estimates robust temporal scale as ``median(|x - median(x)|)``. The
    returned candidate indices are ordered from lower to higher MAD; for an
    even ``top_k``, the upper-middle candidate is selected.
    """
    values = np.asarray(bandpassed_data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("bandpassed_data must be a non-empty samples-by-subcarriers matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("bandpassed_data must contain only finite values")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    channel_medians = np.median(values, axis=0)
    mads = np.median(np.abs(values - channel_medians), axis=0)
    if candidate_subcarriers is None:
        candidate_indices = np.arange(values.shape[1])
    else:
        candidate_indices = np.asarray(candidate_subcarriers, dtype=np.int64).reshape(-1)
        if candidate_indices.size == 0:
            raise ValueError("candidate_subcarriers must not be empty")
        if np.any(candidate_indices < 0) or np.any(candidate_indices >= values.shape[1]):
            raise ValueError("candidate_subcarriers contains an out-of-range index")
        if np.unique(candidate_indices).size != candidate_indices.size:
            raise ValueError("candidate_subcarriers must not contain duplicates")

    valid = candidate_indices[mads[candidate_indices] > np.finfo(np.float64).eps]
    if valid.size == 0:
        raise ValueError("No varying subcarriers found after band-pass filtering")

    count = min(int(top_k), valid.size)
    ranked = valid[np.argsort(mads[valid], kind="stable")]
    candidates = ranked[-count:]
    selected = int(candidates[count // 2])
    return selected, candidates.astype(np.int64), mads
