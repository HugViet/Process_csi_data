"""Estimate breathing rate from a cleaned, band-passed CSI amplitude signal."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


PROMINENCE_STD_FRACTION = 0.2
MAX_IBI_SECONDS = 6.0


def estimate_breathing_rate(
    clean_signal: np.ndarray,
    fs: float,
    p2p_threshold: float = 2.0,
    window_sec: float = 10.0,
    p2p_max_threshold: float = 12.0,
) -> dict[str, float | str | NDArray[np.int64]]:
    """Estimate breathing rate using global peak finding and local energy gates.

    Peaks are first detected once over the complete signal, avoiding artificial
    chunk boundaries that can discard breaths. Each candidate is then tested
    against peak-to-peak amplitude in a centered local window. Only candidates
    whose local P2P lies between ``p2p_threshold`` and ``p2p_max_threshold`` are
    retained. The lower gate suppresses weak apnea/filter-ringing oscillations;
    the upper gate rejects high-amplitude body-motion bursts. Local windows are
    clipped to the signal limits, so peaks near either edge are safe to check.

    BPM is ``60 / median(IBI)`` after removing non-positive intervals and
    intervals longer than ``MAX_IBI_SECONDS``. The median resists missed-peak
    outliers; the maximum interval rejects gaps that would imply under 10 BPM.
    If no valid IBI remains, the result is 0 BPM and ``Apnea`` status.
    ``p2p_threshold`` and ``p2p_max_threshold`` use the same units as the input
    and should be tuned against real breathing, apnea, and motion data.

    Args:
        clean_signal: One-dimensional filtered CSI amplitude waveform.
        fs: Sampling frequency in Hz.
        p2p_threshold: Minimum window peak-to-peak amplitude to permit counting.
        window_sec: Duration in seconds of the local peak-to-peak window.
        p2p_max_threshold: Maximum local peak-to-peak amplitude accepted as
            breathing; larger excursions are treated as movement artifacts.

    Returns:
        A dictionary with ``bpm`` (float), ``status`` (``"Breathing"`` or
        ``"Apnea"``), and ``peaks_indices`` (integer NumPy array).

    Raises:
        ValueError: If the signal or parameters are invalid.
    """
    signal = np.asarray(clean_signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("clean_signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite value")
    if not np.isfinite(p2p_threshold) or p2p_threshold < 0:
        raise ValueError("p2p_threshold must be a non-negative finite value")
    if not np.isfinite(p2p_max_threshold) or p2p_max_threshold <= 0:
        raise ValueError("p2p_max_threshold must be a positive finite value")
    if p2p_max_threshold < p2p_threshold:
        raise ValueError("p2p_max_threshold must be greater than or equal to p2p_threshold")
    if not np.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("window_sec must be a positive finite value")

    if signal.size < 3:
        return {"bpm": 0.0, "status": "Apnea", "peaks_indices": np.empty(0, dtype=np.int64)}

    prominence = max(PROMINENCE_STD_FRACTION * float(np.std(signal)), np.finfo(np.float64).eps)
    minimum_peak_distance = max(1, int(fs * 0.9))
    all_peaks, _ = find_peaks(
        signal,
        distance=minimum_peak_distance,
        prominence=prominence,
    )

    half_window = max(1, int(round(window_sec * fs / 2.0)))
    valid_peaks: list[int] = []
    for peak in all_peaks:
        start = max(0, int(peak) - half_window)
        end = min(signal.size, int(peak) + half_window + 1)
        if end - start < 2:
            continue

        local_p2p = float(np.ptp(signal[start:end]))
        if p2p_threshold <= local_p2p <= p2p_max_threshold:
            valid_peaks.append(int(peak))

    peaks = np.asarray(valid_peaks, dtype=np.int64)
    if peaks.size >= 2:
        ibi_seconds = np.diff(peaks).astype(np.float64) / fs
        valid_ibis = ibi_seconds[
            np.isfinite(ibi_seconds)
            & (ibi_seconds > 0)
            & (ibi_seconds <= MAX_IBI_SECONDS)
        ]
        median_ibi = float(np.median(valid_ibis)) if valid_ibis.size else 0.0
        bpm = float(60.0 / median_ibi) if median_ibi > 0 else 0.0
    else:
        bpm = 0.0
    status = "Breathing" if bpm > 0 else "Apnea"

    return {
        "bpm": float(bpm),
        "status": status,
        "peaks_indices": peaks,
    }
