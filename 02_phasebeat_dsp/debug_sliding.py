import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "..")
sys.path.insert(0, "../03_realtime_app")
from phasebeat_core import PhaseBeatCore
from estimation.breathing_estimator import estimate_breathing_rate
from utils import load_csi_recording

for fname in ["test_1.npz", "test_2.npz", "test_3.npz", "test_4.npz"]:
    rec = load_csi_recording(fname)
    fs = rec.fs
    amps = rec.amplitudes
    step = int(0.35 * fs)
    win_len = int(14.0 * fs)
    core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=0.75, mode="amplitude")
    print(f"\n=== Testing {fname} (fs={fs:.1f}Hz, dur={rec.duration:.1f}s) ===")
    sc = None
    for idx in range(0, len(amps) - win_len, step):
        win = amps[idx : idx + win_len]
        clean = core.process(win, subcarrier=sc)
        if sc is None:
            sc = core.last_selected_subcarrier
        res = estimate_breathing_rate(clean, fs=fs)
        st = res["status"]
        bpm = res["bpm"]
        fbpm = res["fft_bpm"]
        pbpm = res["peak_bpm"]
        snr = res.get("snr", 0)
        var = res.get("var", 0)
        print(f"t={idx/fs:4.1f}s | SC={sc:2d} | Status: {st:9s} | BPM: {bpm:4.1f} | FFT: {fbpm:4.1f} | Peak: {pbpm:4.1f} | SNR: {snr:4.2f} | Var: {var:.4f}")
