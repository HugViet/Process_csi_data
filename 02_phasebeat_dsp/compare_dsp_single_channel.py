"""Controlled DSP comparison across recordings using one fixed subcarrier."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import RAW_DATA_DIR  # noqa: E402
from phasebeat_core import PhaseBeatCore  # noqa: E402


DEFAULT_SUBCARRIER = 25
DEFAULT_FILES = [f"test_{index}.npz" for index in range(1, 5)]


@dataclass
class ProcessedRecord:
    """Raw and DSP-filtered amplitude for one recording and fixed carrier."""

    name: str
    time: NDArray[np.float64]
    raw_amplitude: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    sampling_rate: float


def load_and_process(file_name: str, subcarrier: int) -> ProcessedRecord:
    """Load one capture and process only the fixed subcarrier."""
    path = RAW_DATA_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    with np.load(path) as archive:
        if "amps" not in archive or "times" not in archive:
            raise KeyError(f"{file_name} must contain both 'amps' and 'times'")
        amplitudes = np.asarray(archive["amps"], dtype=np.float64)
        timestamps = np.asarray(archive["times"], dtype=np.float64)

    if amplitudes.ndim != 2 or amplitudes.shape[0] < 2:
        raise ValueError(f"{file_name}: 'amps' must be a samples-by-subcarriers matrix")
    if subcarrier < 0 or amplitudes.shape[1] <= subcarrier:
        raise ValueError(
            f"{file_name}: subcarrier {subcarrier} is out of range "
            f"(available indices: 0-{amplitudes.shape[1] - 1})"
        )
    if timestamps.ndim != 1 or timestamps.size != amplitudes.shape[0]:
        raise ValueError(f"{file_name}: 'times' length must match the amplitude sample count")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{file_name}: timestamps must be finite and non-decreasing")

    elapsed = float(timestamps[-1] - timestamps[0])
    if elapsed <= 0:
        raise ValueError(f"{file_name}: timestamps do not span a positive duration")
    fs = (timestamps.size - 1) / elapsed

    core = PhaseBeatCore(
        fs=fs,
        lowcut=0.1,
        highcut=0.75,
        top_k=1,
        candidate_subcarriers=[subcarrier],
    )
    clean_signal = core.process(amplitudes)
    if core.last_selected_subcarrier != subcarrier:
        raise RuntimeError(
            f"{file_name}: expected SC {subcarrier}, "
            f"got SC {core.last_selected_subcarrier}"
        )

    return ProcessedRecord(
        name=file_name,
        time=timestamps - timestamps[0],
        raw_amplitude=amplitudes[:, subcarrier],
        clean_signal=clean_signal,
        sampling_rate=fs,
    )


def plot_comparison(records: list[ProcessedRecord], subcarrier: int) -> None:
    """Compare raw and unnormalized cleaned traces for one fixed subcarrier."""
    figure, (ax_raw, ax_clean) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    palette = plt.get_cmap("tab10")

    for index, record in enumerate(records):
        color = palette(index % 10)
        label = record.name
        ax_raw.plot(
            record.time,
            record.raw_amplitude,
            color=color,
            alpha=0.85,
            linewidth=1.0,
            label=label,
        )
        ax_clean.plot(
            record.time,
            record.clean_signal,
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=label,
        )

    ax_raw.set_title(
        f"Raw CSI Amplitude - Forced Subcarrier {subcarrier} (Showing DC Offset)"
    )
    ax_raw.set_ylabel("Raw Amplitude")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_clean.set_title(
        "DSP Output - Detrended & Bandpassed (Showing Zero-mean Convergence) "
        f"(SC {subcarrier})"
    )
    ax_clean.set_xlabel("Time from recording start (s)")
    ax_clean.set_ylabel("Filtered Amplitude")
    ax_clean.axhline(0.0, color="black", linewidth=1.2, alpha=0.9)
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Process four same-breathing-condition recordings and plot together."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subcarrier",
        nargs="?",
        type=int,
        default=DEFAULT_SUBCARRIER,
        help=f"Subcarrier index to compare (default: {DEFAULT_SUBCARRIER})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="NPZ files under data/raw (default: test_1.npz through test_4.npz)",
    )
    args = parser.parse_args()
    if args.subcarrier < 0:
        parser.error("subcarrier index must be non-negative")

    records = [load_and_process(file_name, args.subcarrier) for file_name in args.files]
    for record in records:
        print(
            f"File: {record.name} | Fs: {record.sampling_rate:.2f} Hz | "
            f"Forced subcarrier: {args.subcarrier}"
        )
    plot_comparison(records, args.subcarrier)


if __name__ == "__main__":
    main()
