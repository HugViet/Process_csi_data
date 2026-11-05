"""Dataset preparation & sliding window generator for CSI respiration deep learning (v2).

Key improvements over v1:
- 100% Online-Consistent Preprocessing: Each sliding window is extracted from raw
  uniformly-resampled amplitudes and processed independently via ``extract_ai_features``
  (per-packet AGC normalization -> Hampel -> Savitzky-Golay -> Bandpass -> PCA),
  matching the exact realtime buffer path in ``RealtimeEngine``.
- 16.0-second windows (downsampled 4x to 360 timesteps at 22.5 Hz) so each window
  contains 2.7 to 6.4 full breathing cycles instead of less than 1 slow cycle.
- Dual-branch features: temporal multi-subcarrier tensor ``X`` (46, 360),
  normalized breathing-band PSD ``X_spec`` (131,), and periodicity/SNR summary ``X_aux`` (6,).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02_phasebeat_dsp"))

from config.settings import DATA_DIR, RAW_DATA_DIR
from respiration_pipeline import ACTIVE_SUBCARRIERS_46, extract_ai_features, resample_uniform
from utils.data_loader import load_csi_recording

CLASS_NAMES = ["apnea", "slow", "normal", "fast"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_TO_BPM = {"apnea": 0.0, "slow": 10.0, "normal": 16.0, "fast": 24.0}
ACTIVE_SUBCARRIERS = ACTIVE_SUBCARRIERS_46


def resample_recording(
    amplitudes: np.ndarray,
    times: np.ndarray,
    target_fs: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample multi-channel time series onto a strictly uniform time grid."""
    uniform_times, (uniform_amps,) = resample_uniform(times, amplitudes, fs=target_fs)
    return uniform_amps.astype(np.float64), uniform_times.astype(np.float64)


def prepare_dataset(
    window_sec: float = 16.0,
    stride_sec: float = 2.0,
    target_fs: float = 90.0,
    output_path: Path | None = None,
) -> Path:
    """Read all labeled files, apply online-consistent window preprocessing, and save NPZ."""
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = processed_dir / "csi_respiration_dataset_4s.npz"

    files = sorted(RAW_DATA_DIR.glob("*_202611*.npz"))
    if not files:
        raise FileNotFoundError(f"No labeled recording files found in {RAW_DATA_DIR}")

    print(f"[*] Found {len(files)} labeled recordings. Extracting {window_sec:.1f}s realtime-consistent windows...")

    window_len = int(round(window_sec * target_fs))
    stride_len = int(round(stride_sec * target_fs))

    x_wave_list: list[np.ndarray] = []
    x_spec_list: list[np.ndarray] = []
    x_aux_list: list[np.ndarray] = []
    y_class_list: list[int] = []
    y_bpm_list: list[float] = []
    groups_list: list[int] = []

    for file_idx, fpath in enumerate(files):
        try:
            raw_d = np.load(fpath)
            cls_name = str(raw_d.get("label_class", "")).lower()
            if cls_name not in CLASS_TO_ID:
                matched = [c for c in CLASS_NAMES if c in fpath.stem.lower()]
                if not matched:
                    print(f"  [!] Skipping unrecognized file: {fpath.name}")
                    continue
                cls_name = matched[0]

            cls_id = CLASS_TO_ID[cls_name]
            target_bpm = float(raw_d.get("target_bpm", CLASS_TO_BPM[cls_name]))

            rec = load_csi_recording(fpath)
            resampled_amps, _ = resample_recording(rec.amplitudes, rec.relative_time, target_fs=target_fs)

            total_samples = resampled_amps.shape[0]
            slices_from_file = 0

            for start in range(0, total_samples - window_len + 1, stride_len):
                raw_win = resampled_amps[start : start + window_len]
                xw, xs, xa, _ = extract_ai_features(raw_win, fs=target_fs, window_sec=window_sec)
                x_wave_list.append(xw)
                x_spec_list.append(xs)
                x_aux_list.append(xa)
                y_class_list.append(cls_id)
                y_bpm_list.append(target_bpm)
                groups_list.append(file_idx)
                slices_from_file += 1

            print(f"  [+] {fpath.name:<34s} -> {slices_from_file:3d} windows (Class {cls_id}: {cls_name.upper()})")
        except Exception as exc:
            print(f"  [!] Error processing {fpath.name}: {exc}")

    x_wave_arr = np.asarray(x_wave_list, dtype=np.float32)
    x_spec_arr = np.asarray(x_spec_list, dtype=np.float32)
    x_aux_arr = np.asarray(x_aux_list, dtype=np.float32)
    y_cls_arr = np.asarray(y_class_list, dtype=np.int64)
    y_bpm_arr = np.asarray(y_bpm_list, dtype=np.float32)
    groups_arr = np.asarray(groups_list, dtype=np.int64)

    np.savez_compressed(
        output_path,
        X=x_wave_arr,
        X_spec=x_spec_arr,
        X_aux=x_aux_arr,
        y_class=y_cls_arr,
        y_bpm=y_bpm_arr,
        groups=groups_arr,
        class_names=CLASS_NAMES,
        target_fs=target_fs,
        window_sec=window_sec,
    )

    print("\n" + "=" * 70)
    print("  [v] DATASET PREPARATION COMPLETED:")
    print(f"      - Output file:    {output_path}")
    print(f"      - Total samples:  {x_wave_arr.shape[0]} windows ({window_sec:.1f}s each)")
    print(f"      - Wave shape:     {x_wave_arr.shape} | Spec: {x_spec_arr.shape} | Aux: {x_aux_arr.shape}")
    print(f"      - Class balance:  {dict(zip(CLASS_NAMES, np.bincount(y_cls_arr, minlength=4)))}")
    print("=" * 70)
    return output_path


if __name__ == "__main__":
    prepare_dataset(window_sec=16.0, stride_sec=2.0, target_fs=90.0)
