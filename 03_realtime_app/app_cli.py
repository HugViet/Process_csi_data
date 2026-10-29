"""Terminal live monitor for realtime CSI breathing sensing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import SERIAL_PORT, BAUD_RATE

try:
    from .stream_source import LiveSerialSource, SimulatedFileSource
    from .realtime_engine import RealtimeEngine
except ImportError:
    from stream_source import LiveSerialSource, SimulatedFileSource
    from realtime_engine import RealtimeEngine


def run_cli(
    source_type: str = "sim",
    file_name: str = "test_2.npz",
    port: str = SERIAL_PORT,
    mode: str = "phase",
    duration: float = 10.0,
) -> None:
    """Run CLI realtime monitor for a given duration (or until Ctrl+C)."""
    if source_type == "live":
        source = LiveSerialSource(port=port, baudrate=BAUD_RATE)
    else:
        source = SimulatedFileSource(file_name_or_path=file_name, loop=True)

    engine = RealtimeEngine(source=source, window_size_sec=18.0, mode=mode)
    print("=" * 85)
    print(f"CSI REALTIME MONITOR (CLI) | Source: {source_type.upper()} | Mode: {mode.upper()}")
    print("=" * 85)
    print(f"{'Time':<8} | {'Status':<10} | {'BPM':<7} | {'Peak BPM':<9} | {'FFT BPM':<9} | {'SC':<4} | {'Fs(Hz)':<7} | {'Energy':<7}")
    print("-" * 85)

    engine.start()
    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break
            time.sleep(1.0)
            snap = engine.compute_snapshot()
            if snap is None:
                print(f"{elapsed:>6.1f}s | Buffering initial window...")
            else:
                print(
                    f"{elapsed:>6.1f}s | {snap.status:<10} | "
                    f"{snap.bpm:>6.1f} | "
                    f"{snap.peak_bpm:>8.1f} | {snap.fft_bpm:>8.1f} | "
                    f"SC {snap.selected_subcarrier:<2} | {snap.fs:>6.1f} | "
                    f"{snap.variance:>7.3f}"
                )
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
    finally:
        engine.stop()
        print("=" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal CLI Realtime Monitor")
    parser.add_argument("--source", choices=["live", "sim"], default="sim", help="Data source")
    parser.add_argument("--file", default="test_2.npz", help="Simulation file")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port")
    parser.add_argument("--mode", choices=["phase", "amplitude"], default="amplitude", help="DSP mode (default: amplitude)")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    args = parser.parse_args()
    run_cli(
        source_type=args.source,
        file_name=args.file,
        port=args.port,
        mode=args.mode,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
