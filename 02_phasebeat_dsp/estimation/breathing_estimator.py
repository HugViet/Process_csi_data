"""Estimate breathing rate from a cleaned, band-passed CSI signal.

Two complementary methods are provided:

* **FFT-based** (``estimate_breathing_rate_fft``) — finds the dominant
  frequency in the breathing band via Welch PSD.  More stable and robust
  for recordings ≥ 15 s.
* **Peak-counting** (``estimate_breathing_rate_peaks``) — counts validated
  inhalation peaks in the time domain.  Better suited for short or
  real-time windows.
* **Combined** (``estimate_breathing_rate``) — runs both methods and returns
  all results together.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, welch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMINENCE_STD_FRACTION = 0.25
MAX_IBI_SECONDS = 6.5

# Breathing band limits (Hz).  6–45 breaths per minute.
DEFAULT_BREATH_LOW_HZ = 0.1
DEFAULT_BREATH_HIGH_HZ = 0.75

# Energy & SNR thresholds for Apnea / Breath-holding detection
DEFAULT_MIN_VARIANCE = 0.01
DEFAULT_MIN_SNR = 1.8


# ---------------------------------------------------------------------------
# FFT-based estimation
# ---------------------------------------------------------------------------

def estimate_breathing_rate_fft(
    clean_signal: np.ndarray,
    fs: float,
    breath_low_hz: float = DEFAULT_BREATH_LOW_HZ,
    breath_high_hz: float = DEFAULT_BREATH_HIGH_HZ,
    window_seconds: float = 15.0,
    min_variance: float = DEFAULT_MIN_VARIANCE,
    min_snr: float = DEFAULT_MIN_SNR,
) -> dict[str, float | str | NDArray[np.float64]]:
    """Estimate breathing rate from the dominant Welch PSD peak with zero-padding and harmonic resolution.

    Parameters
    ----------
    clean_signal : 1-D array
        Band-passed and cleaned CSI waveform.
    fs : float
        Sampling frequency in Hz.
    breath_low_hz, breath_high_hz : float
        Frequency limits of the breathing band.
    window_seconds : float
        Welch segment length in seconds.
    min_variance : float
        Minimum signal variance required to be considered active respiration.
        Signals below this threshold are marked as "Apnea" (breath-holding).
    min_snr : float
        Minimum peak-to-average spectral ratio in the breathing band.
    """
    signal = np.asarray(clean_signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("clean_signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite value")

    sig_var = float(np.var(signal))

    # Welch PSD with zero-padding (nfft) to get fine frequency resolution (<0.1 BPM)
    nperseg = min(signal.size, max(16, int(fs * window_seconds)))
    nfft = max(2048, int(fs * 60.0))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, nfft=nfft)

    # Restrict to breathing band.
    mask = (freqs >= breath_low_hz) & (freqs <= breath_high_hz)
    if not np.any(mask):
        return {
            "fft_bpm": 0.0,
            "fft_status": "Apnea",
            "fft_dominant_freq": 0.0,
            "fft_freqs": freqs,
            "fft_psd": psd,
            "snr": 0.0,
            "var": sig_var,
        }

    band_freqs = freqs[mask]
    band_psd = psd[mask]

    mean_psd = float(np.mean(band_psd))
    peak_psd = float(np.max(band_psd))
    snr = peak_psd / mean_psd if mean_psd > 0 else 0.0

    # Apnea / Holding Breath detection based on energy or spectral flatness
    if sig_var < min_variance or snr < min_snr:
        return {
            "fft_bpm": 0.0,
            "fft_status": "Apnea",
            "fft_dominant_freq": 0.0,
            "fft_freqs": freqs,
            "fft_psd": psd,
            "snr": snr,
            "var": sig_var,
        }

    # Detect candidate peaks in PSD band
    peaks, _ = find_peaks(band_psd, prominence=max(0.12 * peak_psd, 1e-9))
    if len(peaks) == 0:
        dominant_idx = int(np.argmax(band_psd))
        dominant_freq = float(band_freqs[dominant_idx])
    else:
        # Sort candidate peaks by power descending
        sorted_indices = peaks[np.argsort(band_psd[peaks])[::-1]]
        dominant_freq = float(band_freqs[sorted_indices[0]])

        # Harmonic suppression: If primary peak is f2 (e.g. 26 BPM), check if there is
        # a fundamental f1 ~ f2 / 2 (e.g. 13 BPM) that has significant energy (>= 30% of peak)
        for cand_idx in sorted_indices[1:]:
            cand_freq = float(band_freqs[cand_idx])
            ratio = dominant_freq / cand_freq
            if 1.75 <= ratio <= 2.25 and band_psd[cand_idx] >= 0.30 * peak_psd:
                dominant_freq = cand_freq
                break

    fft_bpm = dominant_freq * 60.0

    return {
        "fft_bpm": fft_bpm,
        "fft_status": "Breathing",
        "fft_dominant_freq": dominant_freq,
        "fft_freqs": freqs,
        "fft_psd": psd,
        "snr": snr,
        "var": sig_var,
    }


# ---------------------------------------------------------------------------
# Peak-counting estimation
# ---------------------------------------------------------------------------

def estimate_breathing_rate_peaks(
    clean_signal: np.ndarray,
    fs: float,
    min_std_mult: float = 0.25,
    max_std_mult: float = 4.0,
    window_sec: float = 3.0,
    min_peak_distance_sec: float = 1.0,
    min_variance: float = DEFAULT_MIN_VARIANCE,
) -> dict[str, float | str | NDArray[np.int64]]:
    """Estimate breathing rate using global peak finding and dynamic energy gates.

    Parameters
    ----------
    clean_signal : 1-D array
        Band-passed and cleaned CSI waveform.
    fs : float
        Sampling frequency in Hz.
    min_std_mult, max_std_mult : float
        Accepted local peak-to-peak range as multiples of signal std.
    window_sec : float
        Half-window (seconds) around each peak for local P2P calculation.
    min_peak_distance_sec : float
        Minimum distance between consecutive peaks in seconds.
        Default 1.0s → supports rates up to 60 BPM (capturing rapid breathing).
    min_variance : float
        Minimum signal variance; below this, Apnea is declared.
    """
    signal = np.asarray(clean_signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("clean_signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite value")
    if not np.isfinite(min_std_mult) or min_std_mult < 0:
        raise ValueError("min_std_mult must be a non-negative finite value")
    if not np.isfinite(max_std_mult) or max_std_mult <= 0:
        raise ValueError("max_std_mult must be a positive finite value")
    if max_std_mult < min_std_mult:
        raise ValueError("max_std_mult must be greater than or equal to min_std_mult")
    if not np.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("window_sec must be a positive finite value")

    sig_var = float(np.var(signal))
    if signal.size < 3 or sig_var < min_variance:
        return {"peak_bpm": 0.0, "peak_status": "Apnea", "peaks_indices": np.empty(0, dtype=np.int64)}

    signal_std = float(np.std(signal))
    p2p_threshold = min_std_mult * signal_std
    p2p_max_threshold = max_std_mult * signal_std

    # Absolute prominence floor prevents triggering on flat baseline noise
    prominence = max(PROMINENCE_STD_FRACTION * signal_std, 0.04)

    # Minimum distance between peaks: default 1.0s corresponds to 60 BPM
    minimum_peak_distance = max(1, int(fs * min_peak_distance_sec))
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
        valid_ibis = ibi_seconds[(ibi_seconds >= 0.8) & (ibi_seconds <= MAX_IBI_SECONDS)]
        median_ibi = float(np.median(valid_ibis)) if valid_ibis.size else 0.0
        peak_bpm = float(60.0 / median_ibi) if median_ibi > 0 else 0.0
    else:
        peak_bpm = 0.0

    return {
        "peak_bpm": float(peak_bpm),
        "peak_status": "Breathing" if peak_bpm > 0 else "Apnea",
        "peaks_indices": peaks,
    }


# ---------------------------------------------------------------------------
# Combined estimation
# ---------------------------------------------------------------------------

def estimate_breathing_rate(
    clean_signal: np.ndarray,
    fs: float,
    min_std_mult: float = 0.25,
    max_std_mult: float = 4.0,
    window_sec: float = 3.0,
    min_peak_distance_sec: float = 1.0,
    fft_window_seconds: float = 15.0,
    min_variance: float = DEFAULT_MIN_VARIANCE,
    min_snr: float = DEFAULT_MIN_SNR,
) -> dict[str, float | str | NDArray]:
    """Run both FFT and peak-counting estimation, return robust fused results.

    Returns a merged dict with keys from both methods, plus:
        ``bpm`` : float — robust fused BPM (stable FFT guided by peak count).
        ``status`` : str — ``"Breathing"`` or ``"Apnea"``.
        ``peaks_indices`` : array — peak locations from peak-counting.
    """
    fft_result = estimate_breathing_rate_fft(
        clean_signal, fs,
        window_seconds=fft_window_seconds,
        min_variance=min_variance,
        min_snr=min_snr,
    )
    peak_result = estimate_breathing_rate_peaks(
        clean_signal, fs,
        min_std_mult=min_std_mult,
        max_std_mult=max_std_mult,
        window_sec=window_sec,
        min_peak_distance_sec=min_peak_distance_sec,
        min_variance=min_variance,
    )

    combined: dict[str, float | str | NDArray] = {}
    combined.update(fft_result)
    combined.update(peak_result)

    f_bpm = float(fft_result["fft_bpm"])
    p_bpm = float(peak_result["peak_bpm"])
    f_status = str(fft_result["fft_status"])
    p_status = str(peak_result["peak_status"])

    # Apnea check: If both detect apnea or FFT reports apnea with low energy
    if f_status == "Apnea" and (p_status == "Apnea" or p_bpm == 0.0):
        combined["bpm"] = 0.0
        combined["status"] = "Apnea"
    elif f_bpm > 0 and p_bpm > 0:
        # If both methods provide values, check agreement
        diff = abs(f_bpm - p_bpm)
        if diff <= 4.0:
            # Strong agreement: smooth blend (60% FFT, 40% Peaks)
            combined["bpm"] = float(0.6 * f_bpm + 0.4 * p_bpm)
        elif 1.8 <= (p_bpm / f_bpm) <= 2.2:
            # Peak finder caught second harmonic: trust fundamental FFT
            combined["bpm"] = f_bpm
        elif 1.8 <= (f_bpm / p_bpm) <= 2.2:
            # FFT caught harmonic: trust peak finder fundamental
            combined["bpm"] = p_bpm
        else:
            # Prefer FFT as primary robust metric
            combined["bpm"] = f_bpm
        combined["status"] = "Breathing"
    elif f_bpm > 0:
        combined["bpm"] = f_bpm
        combined["status"] = "Breathing"
    elif p_bpm > 0:
        combined["bpm"] = p_bpm
        combined["status"] = "Breathing"
    else:
        combined["bpm"] = 0.0
        combined["status"] = "Apnea"

    combined["peaks_indices"] = peak_result["peaks_indices"]

    return combined
