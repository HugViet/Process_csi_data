"""Controlled DSP comparison across recordings using one fixed subcarrier (amplitude or phase)."""

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

from phasebeat_core import PhaseBeatCore  # noqa: E402
from utils import load_csi_recording  # noqa: E402


DEFAULT_SUBCARRIER = 7
DEFAULT_FILES = [f"test_{index}.npz" for index in range(1, 5)]


@dataclass
class ProcessedRecord:
    """Raw and DSP-filtered data for one recording and fixed carrier."""

    name: str
    time: NDArray[np.float64]
    raw_signal: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    sampling_rate: float
    mode: str


def load_and_process(file_name: str, subcarrier: int, mode: str = "phase") -> ProcessedRecord:
    """Load one capture and process only the fixed subcarrier."""
    rec = load_csi_recording(file_name)

    if subcarrier < 0 or rec.num_subcarriers <= subcarrier:
        raise ValueError(
            f"{rec.name}: subcarrier {subcarrier} is out of range "
            f"(available indices: 0-{rec.num_subcarriers - 1})"
        )

    if mode == "phase" and rec.phases is None:
        print(f"[!] {rec.name}: No 'phases' in file, falling back to amplitude mode")
        mode = "amplitude"

    core = PhaseBeatCore(
        fs=rec.fs,
        lowcut=0.1,
        highcut=0.75,
        top_k=1,
        candidate_subcarriers=[subcarrier],
        mode=mode,
    )
    clean_signal = core.process(rec.amplitudes, phases=rec.phases)
    if core.last_selected_subcarrier != subcarrier:
        raise RuntimeError(
            f"{rec.name}: expected SC {subcarrier}, "
            f"got SC {core.last_selected_subcarrier}"
        )

    if mode == "phase" and rec.phases is not None:
        raw_signal = rec.phases[:, subcarrier]
    else:
        raw_signal = rec.amplitudes[:, subcarrier]

    return ProcessedRecord(
        name=rec.name,
        time=rec.relative_time,
        raw_signal=raw_signal,
        clean_signal=clean_signal,
        sampling_rate=rec.fs,
        mode=mode,
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
    mode_label = records[0].mode.capitalize() if records else "Signal"

    for index, record in enumerate(records):
        color = palette(index % 10)
        label = record.name
        ax_raw.plot(
            record.time,
            record.raw_signal,
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
        f"Raw CSI {mode_label} - Forced Subcarrier {subcarrier} (Showing DC Offset)"
    )
    ax_raw.set_ylabel(f"Raw {mode_label}")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_clean.set_title(
        f"DSP Output [{mode_label}] - Detrended & Bandpassed (SC {subcarrier})"
    )
    ax_clean.set_xlabel("Time from recording start (s)")
    ax_clean.set_ylabel(f"Filtered {mode_label}")
    ax_clean.axhline(0.0, color="black", linewidth=1.2, alpha=0.9)
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Process recordings and plot together for a fixed subcarrier."""
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
    parser.add_argument(
        "--mode",
        choices=["phase", "amplitude"],
        default="phase",
        help="Processing mode: 'phase' (default) or 'amplitude'",
    )
    args = parser.parse_args()
    if args.subcarrier < 0:
        parser.error("subcarrier index must be non-negative")

    records = [load_and_process(file_name, args.subcarrier, mode=args.mode) for file_name in args.files]
    for record in records:
        print(
            f"File: {record.name} | Mode: {record.mode} | Fs: {record.sampling_rate:.2f} Hz | "
            f"Forced subcarrier: {args.subcarrier}"
        )
    plot_comparison(records, args.subcarrier)


if __name__ == "__main__":
    main()
