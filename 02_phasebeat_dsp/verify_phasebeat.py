import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import RAW_DATA_DIR
from phasebeat_core import PhaseBeatCore

def verifyPhasebeat_execute(file_name="test_10_40s.npz"):
    target_path = RAW_DATA_DIR / file_name
    if not target_path.exists():
        print(f"[-] Missing: {target_path}")
        return

    data = np.load(target_path)
    amps = data['amps'] 
    n_samples = amps.shape[0]

    if 'times' in data:
        time_axis = data['times']
        duration = time_axis[-1] - time_axis[0]
        fs = n_samples / duration if duration > 0 else 84.0
    else:
        fs = 84.0
        time_axis = np.arange(n_samples) / fs
        duration = n_samples / fs

    print(f"[+] Loaded: {file_name} | Samples: {n_samples} | Fs: {fs:.1f}Hz | Dur: {duration:.1f}s")

    # Đã sửa: Khởi tạo đúng tham số của bộ lọc Bandpass
    core = PhaseBeatCore(fs=fs, lowcut=0.1, highcut=0.6, top_k=5)
    clean_signal = core.pbCore_process(amps)

    plt.figure(figsize=(11, 6), facecolor='w')
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, amps[:, 10], color='tab:blue', linewidth=1.0)
    plt.title(f"Raw Amplitude Subcarrier 10 ({file_name})")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, clean_signal, color='crimson', linewidth=1.5)
    plt.title(f"PhaseBeat 1D Waveform (Bandpass Filtered) - Duration: {duration:.1f}s")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered Amp")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_10_40s.npz"
    verifyPhasebeat_execute(target)