"""Realtime sliding-window buffer and DSP inference engine for CSI breathing sensing."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02_phasebeat_dsp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import RAW_DATA_DIR
from phasebeat_core import PhaseBeatCore
from estimation.breathing_estimator import estimate_breathing_rate

try:
    from .stream_source import CSIStreamSource
except ImportError:
    from stream_source import CSIStreamSource


@dataclass
class RealtimeSnapshot:
    """Current state of the realtime respiratory monitoring."""
    time_axis: NDArray[np.float64]
    clean_waveform: NDArray[np.float64]
    selected_subcarrier: int
    peaks_indices: NDArray[np.int64]
    bpm: float
    peak_bpm: float
    fft_bpm: float
    status: str
    fft_freqs: NDArray[np.float64]
    fft_psd: NDArray[np.float64]
    dominant_freq: float
    fs: float
    total_packets: int
    is_recording: bool
    mode: str
    variance: float = 0.0
    snr: float = 0.0
    recorded_packets: int = 0


class RealtimeEngine:
    """Engine that reads CSI stream packets, maintains a sliding window buffer, and computes vital signs."""

    def __init__(
        self,
        source: CSIStreamSource,
        window_size_sec: float = 18.0,
        mode: str = "amplitude",
        min_window_sec: float = 4.0,
    ) -> None:
        self.source = source
        self.window_size_sec = float(window_size_sec)
        self.min_window_sec = float(min_window_sec)
        self.mode = mode

        self._lock = threading.Lock()
        self._running = False
        self._receiver_thread: threading.Thread | None = None

        # Sliding window buffer
        self._timestamps: Deque[float] = deque()
        self._amplitudes: Deque[NDArray[np.float64]] = deque()
        self._phases: Deque[NDArray[np.float64]] = deque()
        self._total_packets_received = 0

        # Subcarrier stability management (lock carrier to prevent waveform morphing)
        self._active_subcarrier: int | None = None
        self._subcarrier_auto_select = True

        # BPM smoothing history
        self._bpm_history: Deque[float] = deque(maxlen=5)

        # Recording session management
        self._is_recording = False
        self._record_session_name = ""
        self._rec_amps: List[NDArray[np.float64]] = []
        self._rec_phases: List[NDArray[np.float64]] = []
        self._rec_times: List[float] = []

        # Last snapshot
        self._last_snapshot: RealtimeSnapshot | None = None

    def start(self) -> None:
        """Connect source and launch receiver background thread."""
        self.source.connect()
        self._running = True
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

    def stop(self) -> None:
        """Stop worker and release source."""
        self._running = False
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1.0)
        self.source.close()

    def _receiver_loop(self) -> None:
        """High-frequency polling loop for incoming packets."""
        while self._running:
            packet = self.source.read_packet()
            if packet is not None:
                amps, phases, t = packet
                with self._lock:
                    self._total_packets_received += 1
                    self._timestamps.append(t)
                    self._amplitudes.append(amps)
                    self._phases.append(phases)

                    # Prune old samples exceeding the sliding window duration
                    oldest_allowed = t - self.window_size_sec
                    while self._timestamps and self._timestamps[0] < oldest_allowed:
                        self._timestamps.popleft()
                        self._amplitudes.popleft()
                        self._phases.popleft()

                    # Save if recording session is active
                    if self._is_recording:
                        self._rec_amps.append(amps)
                        self._rec_phases.append(phases)
                        self._rec_times.append(t)
            else:
                # Sleep briefly to avoid 100% CPU usage
                time.sleep(0.001)

    def start_recording(self, session_name: str = "session_live") -> None:
        """Begin saving streamed packets to memory."""
        with self._lock:
            self._is_recording = True
            self._record_session_name = session_name
            self._rec_amps = []
            self._rec_phases = []
            self._rec_times = []
            print(f"[*] Started recording session: {session_name}")

    def stop_recording(self) -> Path | None:
        """Stop recording and persist captured packets to an NPZ archive."""
        with self._lock:
            if not self._is_recording or len(self._rec_amps) == 0:
                self._is_recording = False
                return None
            self._is_recording = False
            amps_arr = np.array(self._rec_amps, dtype=np.float64)
            phases_arr = np.array(self._rec_phases, dtype=np.float64)
            times_arr = np.array(self._rec_times, dtype=np.float64)
            count = len(self._rec_times)
            session = self._record_session_name

        times_arr = times_arr - times_arr[0]
        out_path = RAW_DATA_DIR / f"{session}.npz"
        np.savez(out_path, amps=amps_arr, phases=phases_arr, times=times_arr)
        print(f"[v] Saved recording {out_path.name} with {count} packets")
        return out_path

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def get_buffer_status(self) -> Tuple[float, float, int]:
        """Return (current_duration, min_window_sec, packet_count)."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0, self.min_window_sec, len(self._timestamps)
            dur = float(self._timestamps[-1] - self._timestamps[0])
            return dur, self.min_window_sec, len(self._timestamps)

    def set_mode(self, mode: str) -> None:
        """Switch processing mode between 'phase' and 'amplitude'."""
        if mode in ("phase", "amplitude"):
            self.mode = mode
            # Reset active subcarrier to re-evaluate on mode switch
            self._active_subcarrier = None

    def set_subcarrier(self, sc: int | None) -> None:
        """Explicitly set or unlock the active subcarrier."""
        self._active_subcarrier = sc
        self._subcarrier_auto_select = (sc is None)

    def recalibrate_subcarrier(self) -> None:
        """Trigger re-ranking of subcarriers on the next snapshot."""
        self._active_subcarrier = None
        self._subcarrier_auto_select = True

    def compute_snapshot(self) -> RealtimeSnapshot | None:
        """Run DSP pipeline and breathing estimation on current window snapshot."""
        with self._lock:
            if len(self._timestamps) < 10:
                return None
            times = np.array(self._timestamps, dtype=np.float64)
            amps = np.array(self._amplitudes, dtype=np.float64)
            phases = np.array(self._phases, dtype=np.float64)
            total_pkts = self._total_packets_received
            is_rec = self._is_recording
            rec_count = len(self._rec_times) if is_rec else 0

        duration = float(times[-1] - times[0])
        if duration < self.min_window_sec:
            # Need minimum duration to establish filter stability
            return None

        fs = (times.size - 1) / duration
        if fs <= 0 or not np.isfinite(fs):
            return None

        try:
            core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=0.75, mode=self.mode)

            # Subcarrier locking: lock chosen subcarrier so waveform shape doesn't constantly morph
            target_sc = self._active_subcarrier if not self._subcarrier_auto_select else self._active_subcarrier
            clean_signal = core.process(amps, phases=phases, subcarrier=target_sc)
            selected_sc = core.last_selected_subcarrier if core.last_selected_subcarrier is not None else 0

            # If auto-select is enabled and no subcarrier was locked yet, lock to the chosen one
            if self._subcarrier_auto_select and self._active_subcarrier is None:
                self._active_subcarrier = selected_sc

            # Robust breathing rate estimation
            # min_peak_distance_sec=1.0 allows detecting up to 60 BPM without period-doubling / peak-skipping
            est = estimate_breathing_rate(
                clean_signal,
                fs=fs,
                min_std_mult=0.25,
                max_std_mult=4.0,
                window_sec=3.0,
                min_peak_distance_sec=1.0,
                fft_window_seconds=15.0,
                min_variance=0.015,
                min_snr=1.8,
            )

            status = str(est["status"])
            raw_bpm = float(est["bpm"])

            # Temporal smoothing for displayed BPM to eliminate jumping
            if status == "Breathing" and raw_bpm > 0:
                self._bpm_history.append(raw_bpm)
                smoothed_bpm = float(np.median(self._bpm_history))
            else:
                self._bpm_history.clear()
                smoothed_bpm = 0.0

            time_axis = times - times[0]
            snapshot = RealtimeSnapshot(
                time_axis=time_axis,
                clean_waveform=clean_signal,
                selected_subcarrier=selected_sc,
                peaks_indices=np.asarray(est["peaks_indices"], dtype=np.int64),
                bpm=smoothed_bpm,
                peak_bpm=float(est["peak_bpm"]),
                fft_bpm=float(est["fft_bpm"]),
                status=status,
                fft_freqs=np.asarray(est["fft_freqs"], dtype=np.float64),
                fft_psd=np.asarray(est["fft_psd"], dtype=np.float64),
                dominant_freq=float(est["fft_dominant_freq"]),
                fs=fs,
                total_packets=total_pkts,
                is_recording=is_rec,
                mode=self.mode,
                variance=float(np.var(clean_signal)),
                snr=float(est.get("snr", 0.0)),
                recorded_packets=rec_count,
            )
            self._last_snapshot = snapshot
            return snapshot
        except Exception as exc:
            print(f"[!] DSP error in realtime loop: {exc}")
            return self._last_snapshot
