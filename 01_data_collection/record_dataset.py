import time, sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import RAW_DATA_DIR
from serial_reader import CSISerialReader

def recordDataset_start():
    session_name = input("Session name (default: test_precise): ").strip() or "test_precise"
    duration = int(input("Duration (s, default: 20): ").strip() or 20)
    reader = CSISerialReader()
    reader.serialReader_connect()
    
    amps_hist, phases_hist, timestamps = [], [], []
    print(f"[*] Recording {duration}s...")
    start_t, pkts = time.time(), 0

    try:
        while time.time() - start_t < duration:
            amps, phases = reader.serialReader_readNext()
            if amps is not None and phases is not None:
                amps_hist.append(amps)
                phases_hist.append(phases)
                timestamps.append(time.time() - start_t)
                pkts += 1
                if pkts % 100 == 0: print(f"-> Pkts: {pkts} | Time: {timestamps[-1]:.1f}s")
    except KeyboardInterrupt: print("\n[!] Stopped")
    finally: reader.serialReader_close()

    if pkts > 0:
        actual_fs = pkts / timestamps[-1] if timestamps[-1] > 0 else 0
        out_path = RAW_DATA_DIR / f"{session_name}.npz"
        np.savez(out_path, amps=np.array(amps_hist), phases=np.array(phases_hist), times=np.array(timestamps))
        print(f"[v] Saved {pkts} pkts ({timestamps[-1]:.1f}s, Actual Fs: {actual_fs:.1f}Hz) to {out_path}")
    else:
        print("[x] No data")

if __name__ == "__main__":
    recordDataset_start()