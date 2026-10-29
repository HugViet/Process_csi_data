"""Data sources for realtime CSI streaming (Hardware Serial or Simulated File)."""

from __future__ import annotations

import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BAUD_RATE, NUM_SUBCARRIERS, SERIAL_PORT
from utils import load_csi_recording


class CSIStreamSource(ABC):
    """Abstract interface for a CSI packet stream."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection or open stream."""
        pass

    @abstractmethod
    def read_packet(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
        """Read the next CSI packet.

        Returns
        -------
        tuple of (amplitudes, phases, timestamp) or None if no packet is available.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and clean up resources."""
        pass


class LiveSerialSource(CSIStreamSource):
    """Live streaming source from an ESP32 connected via USB Serial."""

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE, timeout: float = 0.5) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._start_time: float = 0.0
        self._packet_count = 0
        self._raw_log_count = 0
        self._last_warn_time = 0.0

    def connect(self) -> None:
        import serial
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self._start_time = time.time()
            self._packet_count = 0
            self._raw_log_count = 0
            self._last_warn_time = time.time()
            print(f"[+] LiveSerialSource connected to {self.port} at {self.baudrate} baud")
            print("[*] Waiting for CSI packets from ESP32...")
        except Exception as exc:
            raise ConnectionError(f"Failed to open serial port {self.port}: {exc}") from exc

    def read_packet(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
        if self.ser is None:
            return None

        if self.ser.in_waiting == 0:
            if self._packet_count == 0 and (time.time() - self._last_warn_time) > 4.0:
                print(f"[!] Warning: Connected to {self.port} but no bytes received yet. (Check if ESP32 is powered or press EN/Reset button).")
                self._last_warn_time = time.time()
            return None

        line = self.ser.readline()
        try:
            line_str = line.decode("utf-8", errors="ignore").strip()
            if not line_str:
                return None

            # Diagnostic: print first few non-empty lines so user sees what ESP32 is printing
            if self._packet_count == 0 and self._raw_log_count < 5:
                print(f"  [ESP32 raw]: {line_str[:120]}")
                self._raw_log_count += 1

            if "CSI_DATA" not in line_str:
                return None

            nums = np.array([int(n) for n in re.findall(r"[-+]?\d+", line_str)], dtype=np.float64)
            if len(nums) < NUM_SUBCARRIERS * 2:
                return None
            csi_complex = nums[-128:]
            i_vals = csi_complex[0::2]
            q_vals = csi_complex[1::2]
            amplitudes = np.sqrt(i_vals**2 + q_vals**2)
            phases = np.arctan2(q_vals, i_vals)
            t = time.time() - self._start_time

            self._packet_count += 1
            if self._packet_count == 1:
                print(f"[+] First CSI_DATA packet received successfully from {self.port}!")
            return amplitudes, phases, t
        except Exception:
            return None

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[+] LiveSerialSource closed")


class SimulatedFileSource(CSIStreamSource):
    """Playback recorded .npz files at their original real-time pace.

    Enables testing the entire realtime application and DSP pipeline without
    physical hardware attached. Seamlessly loops with continuous monotonic time.
    """

    def __init__(self, file_name_or_path: str | Path, loop: bool = True) -> None:
        self.file_name_or_path = file_name_or_path
        self.loop = loop
        self.rec = None
        self._current_index = 0
        self._stream_start_time = 0.0
        self._loop_time_offset = 0.0

    def connect(self) -> None:
        self.rec = load_csi_recording(self.file_name_or_path)
        self._current_index = 0
        self._loop_time_offset = 0.0
        self._stream_start_time = time.time()
        print(f"[+] SimulatedFileSource loaded: {self.rec.name} ({self.rec.num_samples} packets, Fs: {self.rec.fs:.1f} Hz, {self.rec.duration:.1f}s)")

    def read_packet(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
        if self.rec is None:
            return None

        # Check if we finished the file
        if self._current_index >= self.rec.num_samples:
            if self.loop:
                # Loop forward: advance time offset so timestamps are strictly monotonic!
                self._loop_time_offset += self.rec.duration + (1.0 / self.rec.fs)
                self._current_index = 0
            else:
                return None

        now = time.time() - self._stream_start_time
        # Monotonic virtual timestamp for this packet
        packet_time = self._loop_time_offset + self.rec.relative_time[self._current_index]

        # Pace packets to match original recording time
        if now < packet_time:
            return None

        idx = self._current_index
        amps = self.rec.amplitudes[idx]
        phases = self.rec.phases[idx] if self.rec.phases is not None else np.zeros_like(amps)
        self._current_index += 1
        return amps, phases, packet_time

    def close(self) -> None:
        print("[+] SimulatedFileSource closed")
