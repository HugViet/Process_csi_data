"""Robust removal of short CSI amplitude spikes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter


def remove_spikes(
    data: ArrayLike,
    fs: float,
    window_seconds: float = 0.3,
    n_sigma: float = 3.0,
) -> NDArray[np.float64]:
    """Replace local Hampel outliers with the local temporal median.

    The local robust scale is ``1.4826 * MAD``. Samples farther than
    ``n_sigma`` robust standard deviations from the local median are replaced;
    the operation is applied independently to each subcarrier.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a non-empty samples-by-subcarriers matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if fs <= 0 or window_seconds <= 0 or n_sigma <= 0:
        raise ValueError("fs, window_seconds, and n_sigma must be positive")

    window = max(3, int(round(fs * window_seconds)))
    if window % 2 == 0:
        window += 1
    window = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if window < 3:
        return values.copy()

    size = (window, 1)
    local_median = median_filter(values, size=size, mode="reflect")
    local_mad = median_filter(np.abs(values - local_median), size=size, mode="reflect")
    robust_sigma = 1.4826 * local_mad
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values)))) * 32
    outliers = np.abs(values - local_median) > n_sigma * np.maximum(robust_sigma, tolerance)
    cleaned = values.copy()
    cleaned[outliers] = local_median[outliers]
    return cleaned
