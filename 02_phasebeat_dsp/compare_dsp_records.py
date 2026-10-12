"""Compare raw CSI amplitudes and DSP-cleaned signals across recordings."""

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


DEFAULT_FILES = [f"test_{index}.npz" for index in range(1, 5)]


@dataclass
class ProcessedRecord:
    """Raw and filtered views of one capture."""

    name: str
    time: NDArray[np.float64]
    raw_amplitude: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    sampling_rate: float
    selected_subcarrier: int


def load_and_process(file_name: str) -> ProcessedRecord:
    """Load one NPZ capture, run the DSP pipeline, and keep its chosen carrier."""
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
    if timestamps.ndim != 1 or timestamps.size != amplitudes.shape[0]:
        raise ValueError(f"{file_name}: 'times' length must match the amplitude sample count")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{file_name}: timestamps must be finite and non-decreasing")

    elapsed = float(timestamps[-1] - timestamps[0])
    if elapsed <= 0:
        raise ValueError(f"{file_name}: timestamps do not span a positive duration")
    fs = (timestamps.size - 1) / elapsed

    core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=0.75)
    clean_signal = core.process(amplitudes)
    selected = core.last_selected_subcarrier
    if selected is None:
        raise RuntimeError(f"{file_name}: pipeline did not select a subcarrier")

    relative_time = timestamps - timestamps[0]
    return ProcessedRecord(
        name=file_name,
        time=relative_time,
        raw_amplitude=amplitudes[:, selected],
        clean_signal=clean_signal,
        sampling_rate=fs,
        selected_subcarrier=selected,
    )


def _shape_normalize(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove residual mean and scale to unit standard deviation for shape review."""
    centered = signal - np.mean(signal)
    scale = float(np.std(centered))
    if scale <= np.finfo(np.float64).eps:
        return centered
    return centered / scale


def plot_comparison(records: list[ProcessedRecord]) -> None:
    """Plot raw selected-carrier amplitudes and normalized cleaned traces together."""
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
        label = f"{record.name} | SC {record.selected_subcarrier}"
        ax_raw.plot(
            record.time,
            record.raw_amplitude,
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

    ax_raw.set_title("Raw CSI amplitude — selected subcarrier per recording")
    ax_raw.set_ylabel("Raw amplitude")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_clean.set_title("DSP output — shape comparison (each trace normalized to unit standard deviation)")
    ax_clean.set_xlabel("Time from recording start (s)")
    ax_clean.set_ylabel("Normalized filtered amplitude")
    ax_clean.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Process five recordings at once and display comparison plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Five NPZ filenames under data/raw (default: test_1.npz through test_5.npz)",
    )
    args = parser.parse_args()
    # if len(args.files) != 2:
    #     parser.error(f"expected exactly five recordings, received {len(args.files)}")

    records = [load_and_process(file_name) for file_name in args.files]
    for record in records:
        print(
            f"File: {record.name} | Fs: {record.sampling_rate:.2f} Hz | "
            f"Selected subcarrier: {record.selected_subcarrier}"
        )
    plot_comparison(records)


if __name__ == "__main__":
    main()
