"""Butterworth band-pass filtering for respiration-band CSI signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfiltfilt


def butterworth_bandpass(
    data: ArrayLike,
    fs: float,
    lowcut: float = 0.15,
    highcut: float = 0.5,
    order: int = 3,
) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth band-pass independently per subcarrier.

    Second-order sections improve numerical stability; forward/backward
    filtering (`sosfiltfilt`) avoids phase delay in offline processing.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a non-empty samples-by-subcarriers matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if fs <= 0 or order < 1:
        raise ValueError("fs and order must be positive")
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < Nyquist")

    sos = butter(order, (lowcut, highcut), btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, values, axis=0)
    except ValueError as exc:
        raise ValueError(
            "signal is too short for zero-phase band-pass filtering; collect more samples"
        ) from exc
