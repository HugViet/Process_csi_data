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
    if 'amps' not in data:
        print("[-] Dataset has no 'amps' array; this diagnostic now requires CSI amplitudes")
        return
    amplitudes = data['amps']
    n_samples = amplitudes.shape[0]

    if 'times' in data:
        time_axis = data['times']
        duration = time_axis[-1] - time_axis[0]
        fs = n_samples / duration if duration > 0 else 84.0
    else:
        fs = 84.0
        time_axis = np.arange(n_samples) / fs
        duration = n_samples / fs

    print(f"[+] Loaded: {file_name} | Samples: {n_samples} | Fs: {fs:.1f}Hz | Dur: {duration:.1f}s")

    core = PhaseBeatCore(fs=fs, lowcut=0.15, highcut=0.5, top_k=3)
    core.process(amplitudes)
    selected = core.last_selected_subcarrier
    top_details = [
        (int(index), float(core.last_subcarrier_mads[index]))
        for index in core.last_top_subcarriers
    ]
    print(f"[>] MAD top-3 after 0.15-0.5 Hz band-pass (subcarrier, MAD): {top_details}")
    print(f"[>] Selected median-MAD subcarrier: {selected}")
    print("[>] MAD was calculated from the band-passed amplitude matrix")

    fig, (ax_mad, ax_channels) = plt.subplots(
        2, 1, figsize=(12, 8), facecolor='w', gridspec_kw={"height_ratios": [1, 1.4]}
    )

    # Plot 1: the actual selection criterion over all subcarriers.
    all_indices = np.arange(len(core.last_subcarrier_mads))
    valid_mask = np.zeros(len(all_indices), dtype=bool)
    valid_mask[core.last_candidate_subcarriers] = True
    ax_mad.bar(
        all_indices[~valid_mask], core.last_subcarrier_mads[~valid_mask],
        color="lightgray", label="Excluded / invalid",
    )
    ax_mad.bar(
        all_indices[valid_mask], core.last_subcarrier_mads[valid_mask],
        color="steelblue", label="Candidate subcarriers",
    )
    ax_mad.bar(
        core.last_top_subcarriers,
        core.last_subcarrier_mads[core.last_top_subcarriers],
        color="darkorange", label=f"Top-{len(core.last_top_subcarriers)} by MAD",
    )
    ax_mad.bar(
        [selected], [core.last_subcarrier_mads[selected]],
        color="crimson", label=f"Selected median-MAD: #{selected}",
    )
    ax_mad.set_title(f"Subcarrier sensitivity ranking (MAD) — {file_name}")
    ax_mad.set_xlabel("Subcarrier index")
    ax_mad.set_ylabel("Temporal MAD (amplitude units)")
    ax_mad.grid(True, axis="y", alpha=0.3)
    ax_mad.legend(loc="upper right", ncol=2, fontsize=8)

    # Plot 2: compare top candidates after band-pass, on a shared scale.
    selection_data = core.last_bandpassed
    for index in core.last_top_subcarriers:
        is_selected = int(index) == selected
        mad_value = core.last_subcarrier_mads[index]
        ax_channels.plot(
            time_axis, selection_data[:, index],
            linewidth=1.8 if is_selected else 1.0,
            color="crimson" if is_selected else None,
            label=f"SC {index} | MAD={mad_value:.3g}" + (" (selected)" if is_selected else ""),
        )
    ax_channels.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax_channels.set_title("Band-passed amplitude traces used for MAD selection")
    ax_channels.set_xlabel("Time (seconds)")
    ax_channels.set_ylabel("Band-passed CSI amplitude")
    ax_channels.grid(True, alpha=0.3)
    ax_channels.legend(loc="upper right")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_10_40s.npz"
    verifyPhasebeat_execute(target)