"""Standard data loading and validation utility for CSI NPZ recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import RAW_DATA_DIR


@dataclass
class CSIRecording:
    """Parsed and validated CSI recording."""
    name: str
    path: Path
    amplitudes: NDArray[np.float64]
    timestamps: NDArray[np.float64]
    phases: NDArray[np.float64] | None
    fs: float
    duration: float
    num_samples: int
    num_subcarriers: int

    @property
    def relative_time(self) -> NDArray[np.float64]:
        """Timestamps normalized to start at 0.0 seconds."""
        return self.timestamps - self.timestamps[0]


def load_csi_recording(file_name_or_path: str | Path) -> CSIRecording:
    """Load and validate an NPZ CSI recording file.

    Parameters
    ----------
    file_name_or_path : str or Path
        Either an absolute/relative path or a file name under `data/raw/`
        (with or without `.npz` extension).

    Returns
    -------
    CSIRecording
        Validated data dataclass containing amplitudes, timestamps,
        phases (if present), sampling frequency `fs`, and duration.
    """
    path = Path(file_name_or_path)
    if not path.is_file():
        # Check under RAW_DATA_DIR
        candidate = RAW_DATA_DIR / path.name
        if candidate.is_file():
            path = candidate
        elif candidate.with_suffix(".npz").is_file():
            path = candidate.with_suffix(".npz")
        elif path.with_suffix(".npz").is_file():
            path = path.with_suffix(".npz")
        else:
            raise FileNotFoundError(f"Recording not found: {file_name_or_path} (checked {path})")

    with np.load(path) as archive:
        if "amps" not in archive or "times" not in archive:
            raise KeyError(f"{path.name} must contain both 'amps' and 'times' arrays")
        amplitudes = np.asarray(archive["amps"], dtype=np.float64)
        timestamps = np.asarray(archive["times"], dtype=np.float64)
        phases = np.asarray(archive["phases"], dtype=np.float64) if "phases" in archive else None

    if amplitudes.ndim != 2 or amplitudes.shape[0] < 2:
        raise ValueError(f"{path.name}: 'amps' must be a samples-by-subcarriers matrix")
    if timestamps.ndim != 1 or timestamps.size != amplitudes.shape[0]:
        raise ValueError(f"{path.name}: 'times' length ({timestamps.size}) must match amplitude rows ({amplitudes.shape[0]})")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{path.name}: timestamps must be finite and non-decreasing")

    elapsed = float(timestamps[-1] - timestamps[0])
    if elapsed <= 0:
        raise ValueError(f"{path.name}: timestamps do not span a positive duration")

    fs = (timestamps.size - 1) / elapsed
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{path.name}: calculated non-positive sampling frequency {fs}")

    return CSIRecording(
        name=path.stem,
        path=path,
        amplitudes=amplitudes,
        timestamps=timestamps,
        phases=phases,
        fs=fs,
        duration=elapsed,
        num_samples=amplitudes.shape[0],
        num_subcarriers=amplitudes.shape[1],
    )
