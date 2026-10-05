"""Baseline removal for CSI amplitude time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend as linear_detrend
from scipy.signal import savgol_filter


def detrend_amplitude(
    data: ArrayLike,
    fs: float,
    window_seconds: float = 12.0,
    polyorder: int = 2,
) -> NDArray[np.float64]:
    """Subtract a Savitzky-Golay estimate of the slowly varying baseline.

    The smooth baseline is a local polynomial low-pass estimate; subtracting it
    suppresses DC and slow drift while retaining faster respiration variation.
    For records too short to contain the requested window, a linear least-
    squares trend is removed instead. The chosen window should be longer than
    the respiration period of interest to limit attenuation near the low cutoff.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a non-empty samples-by-subcarriers matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if fs <= 0 or window_seconds <= 0:
        raise ValueError("fs and window_seconds must be positive")
    if polyorder < 1:
        raise ValueError("polyorder must be at least 1")

    requested_window = max(polyorder + 2, int(round(fs * window_seconds)))
    if requested_window % 2 == 0:
        requested_window += 1
    window = min(requested_window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)

    if window <= polyorder:
        return linear_detrend(values, axis=0, type="linear")

    baseline = savgol_filter(values, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    return values - baseline
