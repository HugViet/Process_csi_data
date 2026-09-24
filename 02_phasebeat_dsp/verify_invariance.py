import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import RAW_DATA_DIR
from phasebeat_core import PhaseBeatCore

def verifyInvariance_run(f1="test_1.npz", f2="test_2.npz", fs=30, active_sub=10):
    p1, p2 = RAW_DATA_DIR / f1, RAW_DATA_DIR / f2
    if not p1.exists() or not p2.exists():
        print(f"[-] Missing: {p1} or {p2}")
        return

    d1, d2 = np.load(p1)['phases'], np.load(p2)['phases']
    print(f"[+] Loaded {f1} {d1.shape} | {f2} {d2.shape}")

    core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=2.0, top_k=5)
    s1 = core.pbCore_process(d1)
    s2 = core.pbCore_process(d2)

    dc_raw_diff = abs(np.mean(d1[:, active_sub]) - np.mean(d2[:, active_sub]))
    dc_out_diff = abs(np.mean(s1) - np.mean(s2))

    print("=" * 48)
    print(f"[>] Raw DC Diff (Subcarrier {active_sub}) : {dc_raw_diff:.4f}")
    print(f"[>] Output DC Diff (Processed)    : {dc_out_diff:.8f}")
    print("=" * 48)

    if dc_out_diff < 1e-4:
        print("[+] VERDICT: PASSED (Static Multipath Eliminated)")
    else:
        print("[-] VERDICT: FAILED (DC Offset Remains)")

    min_len = min(len(s1), len(s2), 300)
    plt.figure(figsize=(10, 6), facecolor='w')

    plt.subplot(2, 1, 1)
    plt.plot(d1[:min_len, active_sub], label=f"{f1} (Sub {active_sub})", alpha=0.8)
    plt.plot(d2[:min_len, active_sub], label=f"{f2} (Sub {active_sub})", alpha=0.8)
    plt.title(f"Raw Phases Comparison (Subcarrier {active_sub} - Different DC Levels)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(s1[:min_len], label=f"{f1} (Clean)", color='red', linewidth=1.5)
    plt.plot(s2[:min_len], label=f"{f2} (Clean)", color='blue', linewidth=1.2, linestyle='--')
    plt.title("PhaseBeat Normalized Waveforms (Zero-Mean Across Environments)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verifyInvariance_run("test_1.npz", "test_2.npz")