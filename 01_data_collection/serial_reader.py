import serial, re, sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import SERIAL_PORT, BAUD_RATE, NUM_SUBCARRIERS

class CSISerialReader:
    def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.ser = None

    def serialReader_connect(self):
        try: 
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[+] Connected {self.port}")
        except Exception as e: 
            print(f"[-] Connect fail: {e}")
            sys.exit(1)

    def serialReader_parsePacket(self, line):
        try:
            line_str = line.decode('utf-8', errors='ignore').strip()
            if "CSI_DATA" not in line_str: return None, None
            nums = np.array([int(n) for n in re.findall(r'[-+]?\d+', line_str)], dtype=np.float32)
            if len(nums) < NUM_SUBCARRIERS * 2: return None, None
            csi_complex = nums[-128:]
            return np.sqrt(csi_complex[0::2]**2 + csi_complex[1::2]**2), np.arctan2(csi_complex[1::2], csi_complex[0::2])
        except Exception: return None, None

    def serialReader_readNext(self):
        return self.serialReader_parsePacket(self.ser.readline()) if self.ser and self.ser.in_waiting > 0 else (None, None)

    def serialReader_close(self):
        if self.ser and self.ser.is_open: 
            self.ser.close()
            print("[+] Serial closed")