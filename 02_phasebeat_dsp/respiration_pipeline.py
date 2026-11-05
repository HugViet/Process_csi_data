"""Multi-subcarrier respiration extraction pipeline (v2).

Designed for device-free sensing when the subject is NOT necessarily on the
TX-RX line of sight. Steps (benchmarked on the labelled dataset, see
``scratch/bench_methods.py``):

1. Uniform resampling to a fixed rate (ESP32 packet timing jitters 70-100 Hz).
2. Restrict to active data subcarriers (drop guard/null/DC bins).
3. Per-packet AGC normalisation: divide every packet by its mean amplitude
   over all active subcarriers. The ESP32 automatic gain control scales all
   subcarriers of a packet by the same factor; this removes those common
   jumps while keeping the frequency-selective changes caused by the chest.
4. Hampel spike removal -> Savitzky-Golay detrend -> Butterworth band-pass.
   Detrend + band-pass remove the *static* multipath (walls, furniture),
   leaving only the time-varying (dynamic) component.
5. PCA fusion: instead of picking one subcarrier (which may sit in a
   Fresnel-zone "blind spot"), project all subcarriers onto their principal
   components and keep the component with the highest breathing-band SNR.
6. Rate estimation: Hann periodogram over the whole window with parabolic
   peak interpolation, cross-checked with autocorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt

try:
    from .filters import remove_spikes
except ImportError:
    from filters import remove_spikes

# 64-point ESP32 HT20/LLTF: 0-5 & 59-63 are guard bands, 32 is DC.
ACTIVE_SUBCARRIERS_64 = np.r_[6:32, 33:59]

# 46 active data subcarriers (excluding pilots 13, 21, 29, 37, 45, 53 as well)
ACTIVE_SUBCARRIERS_46 = np.array([
    6, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 18, 19, 20,
    22, 23, 24, 25, 26, 27, 28,
    30, 31,
    33, 34, 35, 36,
    38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52,
    54, 55, 56, 57, 58,
], dtype=np.int64)


@dataclass
class RespirationResult:
    """Output of :func:`process_window`."""
    waveform: NDArray[np.float64]          # fused 1-D respiration waveform
    bpm: float                             # estimated rate (0 if not computable)
    spec_bpm: float                        # periodogram estimate
    acf_bpm: float                         # autocorrelation estimate
    snr: float                             # breathing-band peak / mean power
    acf_peak: float                        # autocorrelation peak value (periodicity, 0-1)
    freqs: NDArray[np.float64]
    psd: NDArray[np.float64]
    dominant_freq: float
    peaks_indices: NDArray[np.int64]
    component: int                         # chosen principal component (0-based)
    bandpassed: NDArray[np.float64] = field(repr=False)  # (samples, active subcarriers)


def resample_uniform(
    times: NDArray[np.float64],
    *arrays: NDArray[np.float64],
    fs: float = 90.0,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Linearly interpolate irregular packets onto a uniform ``fs`` grid."""
    t = np.asarray(times, dtype=np.float64)
    _, uniq = np.unique(t, return_index=True)
    t = t[uniq]
    if t.size < 2:
        raise ValueError("need at least 2 distinct timestamps")
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    out = [interp1d(t, np.asarray(a, dtype=np.float64)[uniq], axis=0)(tu) for a in arrays]
    return tu, out


def agc_normalize(amps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Divide each packet by its mean amplitude across subcarriers."""
    return amps / (amps.mean(axis=1, keepdims=True) + 1e-9)


def preprocess(
    x: NDArray[np.float64],
    fs: float,
    lowcut: float = 0.1,
    highcut: float = 0.75,
    detrend_seconds: float = 12.0,
) -> NDArray[np.float64]:
    """Hampel -> Savitzky-Golay detrend -> zero-phase Butterworth band-pass."""
    x = remove_spikes(x, fs=fs, window_seconds=0.3)
    w = int(round(fs * detrend_seconds)) | 1
    if w >= x.shape[0]:
        w = x.shape[0] - 1 if x.shape[0] % 2 == 0 else x.shape[0]
    if w > 3:
        x = x - savgol_filter(x, w, 2, axis=0)
    else:
        x = x - x.mean(axis=0)
    sos = butter(2, (lowcut, highcut), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _spectrum(s: NDArray[np.float64], fs: float, nfft_min: int = 16384) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s = (s - s.mean()) * np.hanning(s.size)
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(max(2, s.size) * 8))))
    psd = np.abs(np.fft.rfft(s, nfft)) ** 2 / s.size
    return np.fft.rfftfreq(nfft, 1.0 / fs), psd


def band_snr(s: NDArray[np.float64], fs: float, lowcut: float = 0.1, highcut: float = 0.75) -> float:
    f, p = _spectrum(s, fs)
    m = (f >= lowcut) & (f <= highcut)
    return float(p[m].max() / (p[m].mean() + 1e-12))


def spectral_bpm(s, fs, lowcut=0.1, highcut=0.75) -> tuple[float, NDArray, NDArray, float]:
    """Periodogram peak with parabolic (log-power) interpolation."""
    f, p = _spectrum(s, fs)
    idx = np.where((f >= lowcut) & (f <= highcut))[0]
    i = int(idx[np.argmax(p[idx])])
    delta = 0.0
    if 0 < i < p.size - 1:
        a, b, c = np.log(p[i - 1:i + 2] + 1e-20)
        den = a - 2 * b + c
        if den != 0:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    fpk = float(f[i] + delta * (f[1] - f[0]))
    return fpk * 60.0, f, p, fpk


def acf_bpm(s, fs, lowcut=0.1, highcut=0.75) -> tuple[float, float]:
    """Breathing rate from the highest autocorrelation peak in the valid lag range."""
    s = s - s.mean()
    n = s.size
    r = np.fft.irfft(np.abs(np.fft.rfft(s, 2 * n)) ** 2)[:n]
    r = r / (r[0] + 1e-12)
    lmin, lmax = int(fs / highcut), min(int(fs / lowcut), n - 1)
    seg = r[lmin:lmax]
    if seg.size < 3:
        return 0.0, 0.0
    pk, _ = find_peaks(seg)
    if pk.size == 0:
        return 0.0, 0.0
    best = int(pk[np.argmax(seg[pk])])
    return 60.0 * fs / (best + lmin), float(seg[best])


def process_window(
    amps: NDArray[np.float64],
    fs: float,
    lowcut: float = 0.1,
    highcut: float = 0.75,
    n_components: int = 3,
    active_subcarriers: NDArray[np.int64] | None = None,
) -> RespirationResult:
    """Run the full v2 pipeline on a uniformly sampled amplitude window."""
    a = np.asarray(amps, dtype=np.float64)
    if active_subcarriers is None:
        active_subcarriers = ACTIVE_SUBCARRIERS_64 if a.shape[1] == 64 else np.arange(a.shape[1])
    a = a[:, active_subcarriers]
    bp = preprocess(agc_normalize(a), fs, lowcut, highcut)

    x = bp - bp.mean(axis=0)
    x = x / (x.std(axis=0) + 1e-9)
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    k = min(n_components, u.shape[1])
    comps = [u[:, i] * sv[i] / np.sqrt(x.shape[0]) for i in range(k)]
    snrs = [band_snr(c, fs, lowcut, highcut) for c in comps]
    ci = int(np.argmax(snrs))
    s = comps[ci]

    sp_bpm, freqs, psd, fdom = spectral_bpm(s, fs, lowcut, highcut)
    ac_bpm, ac_peak = acf_bpm(s, fs, lowcut, highcut)
    bpm = 0.5 * (sp_bpm + ac_bpm) if (ac_bpm > 0 and abs(sp_bpm - ac_bpm) <= 3.0) else sp_bpm

    # Peaks for display: spacing >= 60% of the estimated period
    dist = max(1, int(0.6 * fs * 60.0 / max(bpm, 1.0)))
    peaks, _ = find_peaks(s, distance=dist, prominence=0.3 * np.std(s))

    return RespirationResult(
        waveform=s, bpm=float(bpm), spec_bpm=float(sp_bpm), acf_bpm=float(ac_bpm),
        snr=float(snrs[ci]), acf_peak=float(ac_peak), freqs=freqs, psd=psd,
        dominant_freq=float(fdom), peaks_indices=peaks.astype(np.int64), component=ci,
        bandpassed=bp,
    )


def extract_ai_features(
    raw_amps: NDArray[np.float64],
    fs: float = 90.0,
    window_sec: float = 16.0,
    target_timesteps: int = 360,
    spec_bins: int = 131,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], RespirationResult]:
    """Extract identical (wave, spec, aux) tensors for training and realtime inference."""
    win_n = int(round(window_sec * fs))
    a = np.asarray(raw_amps, dtype=np.float64)
    if a.shape[0] > win_n:
        a = a[-win_n:]
    elif a.shape[0] < win_n:
        t_old = np.linspace(0.0, 1.0, a.shape[0])
        t_new = np.linspace(0.0, 1.0, win_n)
        a = interp1d(t_old, a, axis=0, kind="linear", fill_value="extrapolate")(t_new)

    sub_idx = ACTIVE_SUBCARRIERS_46 if a.shape[1] == 64 else np.arange(a.shape[1])
    res = process_window(a, fs=fs, active_subcarriers=sub_idx)
    bp = res.bandpassed  # (1440, 46)

    # Downsample to target_timesteps (1440 / 4 = 360 timesteps at 22.5 Hz)
    if bp.shape[0] == target_timesteps * 4:
        bp_ds = bp[::4, :].T.astype(np.float32)
    else:
        t_old = np.linspace(0.0, 1.0, bp.shape[0])
        t_new = np.linspace(0.0, 1.0, target_timesteps)
        bp_ds = interp1d(t_old, bp, axis=0, kind="linear")(t_new).T.astype(np.float32)

    mask = (res.freqs >= 0.08) & (res.freqs <= 0.80)
    psd_band = res.psd[mask].astype(np.float32)
    if psd_band.size != spec_bins:
        psd_band = np.interp(
            np.linspace(0.0, 1.0, spec_bins),
            np.linspace(0.0, 1.0, psd_band.size),
            psd_band,
        ).astype(np.float32)
    psd_band = psd_band / (psd_band.sum() + 1e-12)

    aux = np.array([
        res.bpm / 30.0,
        res.spec_bpm / 30.0,
        res.acf_bpm / 30.0,
        res.snr / 10.0,
        res.acf_peak,
        float(np.median(np.std(bp, axis=0))) * 100.0,
    ], dtype=np.float32)

    return bp_ds, psd_band, aux, res

