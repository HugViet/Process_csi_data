"""Run breathing-rate estimation on a CSI recording and visualize detected peaks & spectrum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02_phasebeat_dsp"))

from phasebeat_core import PhaseBeatCore  # noqa: E402
from estimation.breathing_estimator import estimate_breathing_rate  # noqa: E402
from utils import load_csi_recording  # noqa: E402


def run(file_name: str, mode: str = "phase") -> None:
    """Load one capture, run DSP preprocessing, and estimate breathing BPM via FFT & Peak counting."""
    rec = load_csi_recording(file_name)

    if mode == "phase" and rec.phases is None:
        print("[!] No 'phases' in file, falling back to amplitude mode")
        mode = "amplitude"

    core = PhaseBeatCore(fs=rec.fs, lowcut=0.1, highcut=0.75, mode=mode)
    clean_signal = core.process(rec.amplitudes, phases=rec.phases)
    selected_sc = core.last_selected_subcarrier

    result = estimate_breathing_rate(
        clean_signal=clean_signal,
        fs=rec.fs,
        min_std_mult=0.2,
        max_std_mult=3.5,
        window_sec=4.0,
        min_peak_distance_sec=2.0,
    )

    fft_bpm = float(result["fft_bpm"])
    peak_bpm = float(result["peak_bpm"])
    status = str(result["status"])
    peaks_indices = np.asarray(result["peaks_indices"], dtype=np.int64)
    fft_freqs = np.asarray(result["fft_freqs"], dtype=np.float64)
    fft_psd = np.asarray(result["fft_psd"], dtype=np.float64)
    dom_freq = float(result["fft_dominant_freq"])

    mode_label = "Phase" if mode == "phase" else "Amplitude"
    print("=" * 60)
    print(f"File: {rec.name} | Mode: {mode_label} | Fs: {rec.fs:.2f} Hz | Duration: {rec.duration:.1f} s")
    print(f"Selected Subcarrier (MAD): {selected_sc}")
    print(f"Status: {status}")
    print(f"  -> Breathing Rate (FFT):      {fft_bpm:.1f} BPM (Dominant Freq: {dom_freq:.2f} Hz)")
    print(f"  -> Breathing Rate (Peaks):    {peak_bpm:.1f} BPM ({len(peaks_indices)} peaks detected)")
    print("=" * 60)

    time_axis = rec.relative_time

    figure, (ax_time, ax_freq) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.2, 1.0]}
    )

    # 1. Time-Domain Plot
    ax_time.plot(time_axis, clean_signal, label=f"Cleaned Signal ({mode_label})", color="teal", linewidth=1.5)
    if peaks_indices.size > 0:
        ax_time.scatter(
            time_axis[peaks_indices],
            clean_signal[peaks_indices],
            color="red",
            s=60,
            zorder=5,
            label=f"Detected Peaks ({len(peaks_indices)} peaks, {peak_bpm:.1f} BPM)",
        )
    ax_time.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_time.set_title(
        f"Breathing Waveform [{mode_label} - SC {selected_sc}] - {rec.name} | "
        f"Peak: {peak_bpm:.1f} BPM | FFT: {fft_bpm:.1f} BPM ({status})"
    )
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel(f"Filtered {mode_label}")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    # 2. Frequency-Domain Plot (Welch PSD)
    mask = (fft_freqs >= 0.05) & (fft_freqs <= 1.0)
    ax_freq.plot(fft_freqs[mask], fft_psd[mask], color="darkblue", linewidth=1.5, label="PSD")
    ax_freq.axvspan(0.1, 0.75, color="green", alpha=0.1, label="Respiration Band (0.1 - 0.75 Hz / 6 - 45 BPM)")
    if dom_freq > 0:
        ax_freq.axvline(dom_freq, color="red", linestyle="--", linewidth=1.2, label=f"Dominant: {dom_freq:.2f} Hz ({fft_bpm:.1f} BPM)")
    ax_freq.set_title("Power Spectral Density (PSD - Welch)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Power")
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend(loc="upper right")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate breathing rate from a cleaned CSI waveform.")
    parser.add_argument("file", nargs="?", default="test_2.npz", help="NPZ file under data/raw")
    parser.add_argument(
        "--mode", choices=["phase", "amplitude"], default="phase",
        help="Processing mode: 'phase' (default) or 'amplitude'",
    )
    args = parser.parse_args()
    run(args.file, mode=args.mode)
