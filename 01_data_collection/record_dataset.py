"""Interactive & CLI Tool for Collecting Labeled CSI Respiration Datasets with Live Metronome.

Designed for Master Thesis dataset acquisition with ground-truth labels:
- Apnea (0 BPM)
- Slow Breathing (~10-12 BPM)
- Normal Breathing (~16-18 BPM)
- Fast Breathing (~24-28 BPM)
- Custom BPM & Duration
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from config.settings import RAW_DATA_DIR, SERIAL_PORT
from serial_reader import CSISerialReader

PRESETS = {
    "1": {
        "class_name": "apnea",
        "class_id": 0,
        "label_bpm": 0.0,
        "default_duration": 25,
        "desc": "Nín thở (Apnea) - Ngồi yên, không cử động, nín thở tự nhiên",
    },
    "2": {
        "class_name": "slow",
        "class_id": 1,
        "label_bpm": 10.0,
        "default_duration": 50,
        "desc": "Thở chậm (Slow) - 10 BPM (Chu kỳ 6.0s: Hít sâu 3.0s, Thở ra 3.0s)",
    },
    "3": {
        "class_name": "normal",
        "class_id": 2,
        "label_bpm": 16.0,
        "default_duration": 50,
        "desc": "Thở bình thường (Normal) - 16 BPM (Chu kỳ 3.75s: Hít ~1.9s, Thở ~1.9s)",
    },
    "4": {
        "class_name": "fast",
        "class_id": 3,
        "label_bpm": 24.0,
        "default_duration": 50,
        "desc": "Thở nhanh (Fast) - 24 BPM (Chu kỳ 2.5s: Hít 1.25s, Thở 1.25s)",
    },
    "5": {
        "class_name": "custom",
        "class_id": 4,
        "label_bpm": 15.0,
        "default_duration": 50,
        "desc": "Tùy chỉnh (Custom) - Tự nhập nhịp thở và thời lượng",
    },
}


def render_metronome(
    elapsed: float,
    duration: float,
    target_bpm: float,
    label_class: str,
    pkt_count: int,
    actual_fs: float,
) -> str:
    """Generate dynamic visual metronome line for terminal display."""
    progress_pct = min(100, int((elapsed / duration) * 100)) if duration > 0 else 0
    bar_width = 15
    filled = int(round(bar_width * (progress_pct / 100.0)))
    prog_bar = "█" * filled + "░" * (bar_width - filled)

    if label_class == "apnea" or target_bpm <= 0:
        guide = "⏸️ [ ĐANG NÍN THỞ (HOLD BREATH) ]"
    else:
        cycle_sec = 60.0 / target_bpm
        pos_in_cycle = elapsed % cycle_sec
        half = cycle_sec / 2.0
        breath_pct = pos_in_cycle / half if pos_in_cycle < half else (pos_in_cycle - half) / half
        b_width = 10
        b_filled = min(b_width, max(1, int(round(b_width * breath_pct))))

        if pos_in_cycle < half:
            # Inhale phase (expanding)
            wave = "▶" * b_filled + " " * (b_width - b_filled)
            guide = f"🫁 HÍT VÀO [{wave}]"
        else:
            # Exhale phase (contracting)
            wave = "◀" * (b_width - b_filled + 1) + " " * (b_filled - 1)
            guide = f"💨 THỞ RA  [{wave}]"

    return f"\r⏱️ {elapsed:4.1f}s/{duration:.0f}s [{prog_bar}] {progress_pct:3d}% | {guide} | 📦 {pkt_count:4d} pkts ({actual_fs:4.1f} Hz)"


def run_collection(
    port: str,
    class_name: str,
    class_id: int,
    target_bpm: float,
    duration: int,
    subject: str = "subject_01",
    custom_name: str | None = None,
) -> Path | None:
    """Connect to serial port, run countdown, record CSI packets with visual metronome, and save."""
    reader = CSISerialReader(port=port)
    try:
        reader.serialReader_connect()
    except Exception as exc:
        print(f"\n[!] LỖI KẾT NỐI CỔNG {port}: {exc}")
        print("[*] Gợi ý: Kiểm tra dây cắm USB, rút ra cắm lại hoặc đổi số cổng COM tương ứng.")
        return None

    print("\n" + "=" * 70)
    print(f"  CHUẨN BỊ THU MẪU: {class_name.upper()} ({target_bpm:.1f} BPM, {duration}s)")
    print(f"  Đối tượng: {subject} | Cổng: {port}")
    print("=" * 70)

    # 3-second countdown
    print("  Ổn định tư thế ngồi đối diện thiết bị ESP32...")
    for c in [3, 2, 1]:
        print(f"  >>> BẮT ĐẦU TRONG: {c}s...", end="\r", flush=True)
        time.sleep(1.0)
    print("  >>> BẮT ĐẦU THỞ THEO NHỊP!                       \n")

    amps_hist: list[np.ndarray] = []
    phases_hist: list[np.ndarray] = []
    timestamps: list[float] = []

    start_t = time.time()
    last_print = 0.0

    try:
        while True:
            now = time.time()
            elapsed = now - start_t
            if elapsed >= duration:
                break

            amps, phases = reader.serialReader_readNext()
            if amps is not None and phases is not None:
                amps_hist.append(amps)
                phases_hist.append(phases)
                timestamps.append(elapsed)

            # Refresh metronome display ~15 times per second
            if now - last_print >= 0.06:
                fs_est = len(timestamps) / elapsed if elapsed > 0.5 else 0.0
                line = render_metronome(
                    elapsed=elapsed,
                    duration=duration,
                    target_bpm=target_bpm,
                    label_class=class_name,
                    pkt_count=len(timestamps),
                    actual_fs=fs_est,
                )
                sys.stdout.write(line)
                sys.stdout.flush()
                last_print = now

            if amps is None:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n[!] Người dùng bấm dừng thu mẫu sớm (Ctrl+C).")
    finally:
        reader.serialReader_close()

    total_pkts = len(timestamps)
    if total_pkts < 30:
        print("\n[x] Dữ liệu thu được quá ít (< 30 packets). Không lưu file.")
        return None

    actual_duration = timestamps[-1]
    actual_fs = total_pkts / actual_duration if actual_duration > 0 else 0.0

    # Generate standardized file name
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_name:
        filename = f"{custom_name}.npz"
    else:
        filename = f"{class_name}_{int(round(target_bpm))}bpm_{timestamp_str}.npz"

    out_path = RAW_DATA_DIR / filename
    amps_arr = np.asarray(amps_hist, dtype=np.float64)
    phases_arr = np.asarray(phases_hist, dtype=np.float64)
    times_arr = np.asarray(timestamps, dtype=np.float64)

    np.savez_compressed(
        out_path,
        amps=amps_arr,
        phases=phases_arr,
        times=times_arr,
        label_class=class_name,
        class_id=int(class_id),
        target_bpm=float(target_bpm),
        duration=float(actual_duration),
        actual_fs=float(actual_fs),
        subject=subject,
        created_at=timestamp_str,
    )

    print("\n\n" + "=" * 70)
    print("  [v] ĐÃ LƯU THÀNH CÔNG BẢN GHI DATASET:")
    print(f"      - File:        {out_path.name}")
    print(f"      - Đường dẫn:   {out_path}")
    print(f"      - Số gói:      {total_pkts} packets")
    print(f"      - Thời lượng:  {actual_duration:.2f}s (Tần số lấy mẫu: {actual_fs:.1f} Hz)")
    print(f"      - Nhãn Class:  {class_name.upper()} (Class ID: {class_id})")
    print(f"      - Nhãn BPM:    {target_bpm:.1f} BPM")
    print("=" * 70 + "\n")
    return out_path


def interactive_menu() -> None:
    """Friendly CLI menu for guided dataset collection."""
    print("=" * 75)
    print("          CSI DATASET RECORDER - THU THẬP MẪU HÔ HẤP CHO AI")
    print("=" * 75)
    print("Chọn kịch bản thu mẫu theo danh sách:")
    for k, v in PRESETS.items():
        print(f"  [{k}] {v['desc']}")

    choice = input("\nNhập lựa chọn [1-5] (Mặc định: 3 - Normal): ").strip() or "3"
    if choice not in PRESETS:
        choice = "3"

    preset = PRESETS[choice]
    class_name = preset["class_name"]
    class_id = preset["class_id"]
    target_bpm = preset["label_bpm"]
    duration = preset["default_duration"]

    if choice == "5":
        # Custom input
        c_name = input("Tên nhãn class (ví dụ: slow/normal/fast/apnea/custom): ").strip() or "custom"
        class_name = c_name.lower()
        try:
            target_bpm = float(input("Nhịp thở mục tiêu BPM (ví dụ: 15.0): ").strip() or 15.0)
        except ValueError:
            target_bpm = 15.0
        try:
            duration = int(input("Thời lượng thu (giây, ví dụ 50): ").strip() or 50)
        except ValueError:
            duration = 50
        class_id = 4
    else:
        dur_in = input(f"Thời lượng thu (giây, mặc định {duration}s): ").strip()
        if dur_in.isdigit():
            duration = int(dur_in)

    port_in = input(f"Cổng COM kết nối ESP32 (mặc định: {SERIAL_PORT}): ").strip() or SERIAL_PORT
    subject_in = input("Tên người thực hiện (mặc định: subject_01): ").strip() or "subject_01"

    run_collection(
        port=port_in,
        class_name=class_name,
        class_id=class_id,
        target_bpm=target_bpm,
        duration=duration,
        subject=subject_in,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI Breathing Dataset Acquisition Tool")
    parser.add_argument(
        "--class",
        dest="class_name",
        choices=["apnea", "slow", "normal", "fast", "custom"],
        default=None,
        help="Loại nhãn kịch bản thu mẫu",
    )
    parser.add_argument("--bpm", type=float, default=None, help="Nhãn nhịp thở mục tiêu BPM")
    parser.add_argument("--duration", type=int, default=None, help="Thời lượng thu mẫu tính bằng giây")
    parser.add_argument("--port", default=SERIAL_PORT, help=f"Cổng Serial COM (mặc định: {SERIAL_PORT})")
    parser.add_argument("--subject", default="subject_01", help="Tên đối tượng thu mẫu")
    parser.add_argument("--name", default=None, help="Tên file tùy chọn (không kèm .npz)")

    args = parser.parse_args()

    # If no class specified, open interactive guided menu
    if args.class_name is None:
        interactive_menu()
        return

    # Map class to default bpm and duration
    class_map = {
        "apnea": (0, 0.0, 25),
        "slow": (1, 10.0, 50),
        "normal": (2, 16.0, 50),
        "fast": (3, 24.0, 50),
        "custom": (4, 15.0, 50),
    }

    cid, def_bpm, def_dur = class_map[args.class_name]
    target_bpm = args.bpm if args.bpm is not None else def_bpm
    duration = args.duration if args.duration is not None else def_dur

    run_collection(
        port=args.port,
        class_name=args.class_name,
        class_id=cid,
        target_bpm=target_bpm,
        duration=duration,
        subject=args.subject,
        custom_name=args.name,
    )


if __name__ == "__main__":
    main()
