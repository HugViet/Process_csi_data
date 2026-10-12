"""Run and visualize the CSI amplitude DSP pipeline without rate estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02_phasebeat_dsp"))

from config.settings import RAW_DATA_DIR  # noqa: E402
from phasebeat_core import PhaseBeatCore  # noqa: E402


def run(file_name: str, show_plot: bool = False) -> None:
    """Process one capture and optionally compare raw and cleaned amplitudes."""
    data_path = RAW_DATA_DIR / file_name
    if not data_path.is_file():
        raise FileNotFoundError(f"Recording not found: {data_path}")

    with np.load(data_path) as data:
        if "amps" not in data:
            raise KeyError(f"{file_name} has no 'amps' array")
        if "times" not in data:
            raise KeyError(f"{file_name} has no 'times' array; cannot calculate sampling rate")
        amplitudes = np.asarray(data["amps"], dtype=np.float64)
        timestamps = np.asarray(data["times"], dtype=np.float64)

    if amplitudes.ndim != 2 or amplitudes.shape[0] < 2:
        raise ValueError("'amps' must be a samples-by-subcarriers matrix with at least two samples")
    if timestamps.ndim != 1 or timestamps.size != amplitudes.shape[0]:
        raise ValueError("'times' must be one-dimensional and match the number of amplitude samples")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError("'times' must contain finite, non-decreasing timestamps")

    elapsed = float(timestamps[-1] - timestamps[0])
    fs = (timestamps.size - 1) / elapsed
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError("Could not calculate a positive sampling frequency")

    core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=0.75)
    clean_signal = core.process(amplitudes)
    selected = core.last_selected_subcarrier
    if selected is None:
        raise RuntimeError("DSP pipeline did not select a subcarrier")

    print(f"File: {file_name}")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Selected subcarrier: {selected}")

    if show_plot:
        time_axis = timestamps - timestamps[0]
        raw_amplitude = amplitudes[:, selected]
        figure, (ax_raw, ax_clean) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
        )

        ax_raw.plot(time_axis, raw_amplitude, color="steelblue", label="Raw CSI amplitude")
        ax_raw.set_title(f"Raw vs. DSP-cleaned amplitude — {file_name} (SC {selected})")
        ax_raw.set_ylabel("Raw amplitude")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.legend()

        ax_clean.plot(time_axis, clean_signal, color="darkgreen", label="Cleaned band-passed signal")
        ax_clean.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax_clean.set_xlabel("Time (s)")
        ax_clean.set_ylabel("Filtered amplitude")
        ax_clean.grid(True, alpha=0.3)
        ax_clean.legend()

        figure.tight_layout()
        plt.show()


def main() -> None:
    """Parse command-line arguments and run the DSP diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file",
        nargs="?",
        default="test_1_sfn.npz",
        help="NPZ filename under data/raw",
    )
    parser.add_argument("--plot", action="store_true", help="Plot raw and cleaned amplitudes")
    args = parser.parse_args()
    run(args.file, show_plot=args.plot)


if __name__ == "__main__":
    main()
