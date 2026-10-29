"""Compare raw CSI and DSP-cleaned signals across recordings (amplitude or phase)."""

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


DEFAULT_FILES = [f"test_{index}.npz" for index in range(1, 5)]


@dataclass
class ProcessedRecord:
    """Raw and filtered views of one capture."""

    name: str
    time: NDArray[np.float64]
    raw_signal: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    sampling_rate: float
    selected_subcarrier: int
    mode: str


def load_and_process(file_name: str, mode: str = "phase") -> ProcessedRecord:
    """Load one NPZ capture, run the DSP pipeline, and keep its chosen carrier."""
    rec = load_csi_recording(file_name)

    if mode == "phase" and rec.phases is None:
        print(f"[!] {rec.name}: No 'phases' in file, falling back to amplitude mode")
        mode = "amplitude"

    core = PhaseBeatCore(fs=rec.fs, lowcut=0.1, highcut=0.75, mode=mode)
    clean_signal = core.process(rec.amplitudes, phases=rec.phases)
    selected = core.last_selected_subcarrier
    if selected is None:
        raise RuntimeError(f"{rec.name}: pipeline did not select a subcarrier")

    # For the raw signal view, show the selected subcarrier from the source data.
    if mode == "phase" and rec.phases is not None:
        raw_signal = rec.phases[:, selected]
    else:
        raw_signal = rec.amplitudes[:, selected]

    return ProcessedRecord(
        name=rec.name,
        time=rec.relative_time,
        raw_signal=raw_signal,
        clean_signal=clean_signal,
        sampling_rate=rec.fs,
        selected_subcarrier=selected,
        mode=mode,
    )


def _shape_normalize(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove residual mean and scale to unit standard deviation for shape review."""
    centered = signal - np.mean(signal)
    scale = float(np.std(centered))
    if scale <= np.finfo(np.float64).eps:
        return centered
    return centered / scale


def plot_comparison(records: list[ProcessedRecord]) -> None:
    """Plot raw selected-carrier signals and normalized cleaned traces together."""
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
        label = f"{record.name} | SC {record.selected_subcarrier}"
        ax_raw.plot(
            record.time,
            record.raw_signal,
            color=color,
            alpha=0.8,
            linewidth=1.0,
            label=label,
        )
        ax_clean.plot(
            record.time,
            _shape_normalize(record.clean_signal),
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=label,
        )

    ax_raw.set_title(f"Raw CSI {mode_label} — selected subcarrier per recording")
    ax_raw.set_ylabel(f"Raw {mode_label}")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_clean.set_title(f"DSP output [{mode_label}] — shape comparison (normalized)")
    ax_clean.set_xlabel("Time from recording start (s)")
    ax_clean.set_ylabel(f"Normalized filtered {mode_label}")
    ax_clean.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Process recordings and display comparison plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="NPZ filenames under data/raw (default: test_1 through test_4)",
    )
    parser.add_argument(
        "--mode", choices=["phase", "amplitude"], default="phase",
        help="Processing mode: 'phase' (default) or 'amplitude'",
    )
    args = parser.parse_args()

    records = [load_and_process(file_name, mode=args.mode) for file_name in args.files]
    for record in records:
        print(
            f"File: {record.name} | Mode: {record.mode} | Fs: {record.sampling_rate:.2f} Hz | "
            f"Selected subcarrier: {record.selected_subcarrier}"
        )
    plot_comparison(records)


if __name__ == "__main__":
    main()
