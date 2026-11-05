import os
import re
import sys
import numpy as np
import serial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import BAUD_RATE, NUM_SUBCARRIERS, SERIAL_PORT


class CSISerialReader:
    """Serial reader for raw CSI packet stream from ESP32."""

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: serial.Serial | None = None

    def serialReader_connect(self) -> None:
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[+] Connected to ESP32 on {self.port} at {self.baudrate} baud")
        except Exception as e:
            raise ConnectionError(f"Failed to open {self.port}: {e}") from e

    def serialReader_parsePacket(self, line: bytes):
        try:
            line_str = line.decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" not in line_str:
                return None, None
            nums = np.array([int(n) for n in re.findall(r"[-+]?\d+", line_str)], dtype=np.float32)
            if len(nums) < NUM_SUBCARRIERS * 2:
                return None, None
            csi_complex = nums[-128:]
            i_vals = csi_complex[0::2]
            q_vals = csi_complex[1::2]
            amps = np.sqrt(i_vals**2 + q_vals**2)
            phases = np.arctan2(q_vals, i_vals)
            return amps, phases
        except Exception:
            return None, None

    def serialReader_readNext(self):
        if self.ser and self.ser.in_waiting > 0:
            return self.serialReader_parsePacket(self.ser.readline())
        return None, None

    def serialReader_close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[+] Serial connection closed")
