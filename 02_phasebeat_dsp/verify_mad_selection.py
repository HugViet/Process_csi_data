"""Visualize best-versus-worst valid MAD subcarriers for one CSI recording (amplitude or phase)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phasebeat_core import PhaseBeatCore  # noqa: E402
from utils import load_csi_recording  # noqa: E402


def run(file_name: str, mode: str = "phase") -> None:
    """Process a capture and compare its selected channel to the lowest-MAD candidate."""
    rec = load_csi_recording(file_name)

    if mode == "phase" and rec.phases is None:
        print("[!] No 'phases' in file, falling back to amplitude mode")
        mode = "amplitude"

    core = PhaseBeatCore(fs=rec.fs, lowcut=0.1, highcut=0.75, mode=mode)
    core.process(rec.amplitudes, phases=rec.phases)

    best_sc = core.last_selected_subcarrier
    mads = core.last_subcarrier_mads
    candidates = core.last_candidate_subcarriers
    bandpassed = core.last_bandpassed
    if best_sc is None or mads is None or candidates is None or bandpassed is None:
        raise RuntimeError("DSP pipeline did not retain MAD selection diagnostics")

    valid_candidates = candidates[mads[candidates] > np.finfo(np.float64).eps]
    if valid_candidates.size == 0:
        raise RuntimeError("No valid candidate subcarriers with positive MAD")
    worst_sc = int(valid_candidates[np.argmin(mads[valid_candidates])])
    best_sc = int(best_sc)

    mode_label = "Phase" if mode == "phase" else "Amplitude"
    print(f"File: {rec.name} | Mode: {mode_label}")
    print(f"Sampling rate: {rec.fs:.2f} Hz")
    print(f"Best SC: {best_sc} | MAD={mads[best_sc]:.6g}")
    print(f"Worst SC: {worst_sc} | MAD={mads[worst_sc]:.6g}")

    figure, (ax_mad, ax_signal) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # Bar chart: show every MAD value and distinguish the chosen extremes.
    subcarrier_indices = np.arange(mads.size)
    bar_colors = np.full(mads.size, "lightgray", dtype=object)
    bar_colors[best_sc] = "red"
    bar_colors[worst_sc] = "black"
    bars = ax_mad.bar(subcarrier_indices, mads, color=bar_colors, width=0.85)
    ax_mad.annotate(
        "Best",
        xy=(best_sc, mads[best_sc]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color="red",
        fontweight="bold",
    )
    ax_mad.annotate(
        "Worst",
        xy=(worst_sc, mads[worst_sc]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color="black",
        fontweight="bold",
    )
    ax_mad.set_title(f"MAD Subcarrier Spectrum [{mode_label}]")
    ax_mad.set_xlabel("Subcarrier index")
    ax_mad.set_ylabel(f"Temporal MAD (filtered {mode_label})")
    ax_mad.set_xlim(-0.5, len(bars) - 0.5)
    ax_mad.grid(True, axis="y", alpha=0.3)

    time_axis = rec.relative_time
    ax_signal.plot(
        time_axis,
        bandpassed[:, best_sc],
        color="red",
        linewidth=1.5,
        label=f"Best MAD — SC {best_sc} ({mads[best_sc]:.4g})",
    )
    ax_signal.plot(
        time_axis,
        bandpassed[:, worst_sc],
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"Worst MAD — SC {worst_sc} ({mads[worst_sc]:.4g})",
    )
    ax_signal.axhline(0.0, color="gray", linewidth=0.7, alpha=0.7)
    ax_signal.set_title(f"Signal Quality Comparison: Best MAD vs Worst MAD [{mode_label}]")
    ax_signal.set_xlabel("Time (s)")
    ax_signal.set_ylabel(f"Band-passed {mode_label}")
    ax_signal.grid(True, alpha=0.3)
    ax_signal.legend()

    figure.tight_layout()
    plt.show()


def main() -> None:
    """Parse the recording name and display its MAD comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file",
        nargs="?",
        default="test_3",
        help="Recording name under data/raw, with or without .npz (default: test_3)",
    )
    parser.add_argument(
        "--mode",
        choices=["phase", "amplitude"],
        default="phase",
        help="Processing mode: 'phase' (default) or 'amplitude'",
    )
    args = parser.parse_args()
    run(args.file, mode=args.mode)


if __name__ == "__main__":
    main()
