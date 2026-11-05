"""Interactive Realtime Respiratory Monitoring Dashboard with Live DSP & Recording."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import SERIAL_PORT, BAUD_RATE

try:
    from .stream_source import LiveSerialSource, SimulatedFileSource
    from .realtime_engine import RealtimeEngine, RealtimeSnapshot
except ImportError:
    from stream_source import LiveSerialSource, SimulatedFileSource
    from realtime_engine import RealtimeEngine, RealtimeSnapshot


class RealtimeDashboard:
    """Matplotlib-powered realtime GUI dashboard for PhaseBeat CSI breathing sensing."""

    def __init__(self, engine: RealtimeEngine) -> None:
        self.engine = engine
        self.fig = plt.figure(figsize=(13, 8), facecolor="#181c24")
        self.fig.canvas.manager.set_window_title("CSI PhaseBeat - Realtime Vital Sign Monitor")

        # Grid layout: 3 rows (Header Info, Waveform, Frequency Spectrum)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[0.5, 1.4, 1.0], hspace=0.35)

        self.ax_info = self.fig.add_subplot(gs[0])
        self.ax_wave = self.fig.add_subplot(gs[1])
        self.ax_psd = self.fig.add_subplot(gs[2])

        self._init_info_panel()
        self._init_wave_panel()
        self._init_psd_panel()

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Plot lines
        (self.line_wave,) = self.ax_wave.plot([], [], color="#00ffff", linewidth=1.8, label="Cleaned Signal")
        (self.scatter_peaks,) = self.ax_wave.plot([], [], "ro", markersize=8, zorder=5, label="Breathing Peaks")
        (self.line_psd,) = self.ax_psd.plot([], [], color="#39ff14", linewidth=1.6, label="Welch PSD")
        self.vline_dom = self.ax_psd.axvline(0, color="red", linestyle="--", alpha=0.0)

        # Smooth Y-axis bounds
        self.cur_ymin: float = -1.5
        self.cur_ymax: float = 1.5

        self.ani = None

    def _init_info_panel(self) -> None:
        self.ax_info.set_facecolor("#181c24")
        self.ax_info.axis("off")

        # Left Column: System & Branding
        self.text_title = self.ax_info.text(
            0.015, 0.72, "CSI RESPIRATION MONITOR", color="#ffffff",
            fontsize=12, fontweight="bold", va="center"
        )
        self.text_subtitle = self.ax_info.text(
            0.015, 0.40, "Dual-Pipeline: DSP + BiLSTM", color="#8892b0",
            fontsize=8.5, va="center"
        )
        self.text_rec_tag = self.ax_info.text(
            0.015, 0.12, "⚪ Standby", color="#a4b0be",
            fontsize=9, fontweight="bold", va="center"
        )

        # Center-Left Column: DSP Vital Sign Estimate
        self.text_dsp_label = self.ax_info.text(
            0.32, 0.82, "DSP ESTIMATION", color="#8892b0",
            fontsize=8.5, fontweight="bold", va="center", ha="center"
        )
        self.text_bpm = self.ax_info.text(
            0.32, 0.48, "-- BPM", color="#00ffcc",
            fontsize=24, fontweight="bold", va="center", ha="center"
        )
        self.text_status = self.ax_info.text(
            0.32, 0.14, "[INITIALIZING]", color="#ffaa00",
            fontsize=9.5, fontweight="bold", va="center", ha="center"
        )

        # Center-Right Column: Deep Learning AI Prediction
        self.text_ai_label = self.ax_info.text(
            0.57, 0.82, "AI MODEL (1D-CNN + BiLSTM)", color="#8892b0",
            fontsize=8.5, fontweight="bold", va="center", ha="center"
        )
        self.text_ai_state = self.ax_info.text(
            0.57, 0.48, "--", color="#00ffcc",
            fontsize=18, fontweight="bold", va="center", ha="center"
        )
        self.text_ai_bpm = self.ax_info.text(
            0.57, 0.14, "AI BPM: --", color="#cccccc",
            fontsize=9.5, va="center", ha="center"
        )

        # Right Column: Technical Details & Hotkeys
        self.text_details = self.ax_info.text(
            0.78, 0.50,
            "SC: -- | Fs: -- Hz\n"
            "FFT: -- | Peak: --\n"
            "Mode: AMPLITUDE\n"
            "Keys: [R] Rec | [M] Mode",
            color="#cccccc", fontsize=8.5, va="center", family="monospace"
        )

    def _init_wave_panel(self) -> None:
        self.ax_wave.set_facecolor("#1f242d")
        self.ax_wave.tick_params(colors="#8892b0")
        for spine in self.ax_wave.spines.values():
            spine.set_color("#2d3748")
        self.ax_wave.grid(True, color="#2d3748", linestyle="--", alpha=0.7)
        self.ax_wave.set_title("Respiration Waveform (Sliding Window)", color="#e2e8f0", fontsize=11, loc="left")
        self.ax_wave.set_xlabel("Time in Window (s)", color="#8892b0", fontsize=9)
        self.ax_wave.set_ylabel("Filtered Signal", color="#8892b0", fontsize=9)
        self.ax_wave.axhline(0.0, color="#4a5568", linewidth=0.8)
        self.ax_wave.set_xlim(0.0, self.engine.window_size_sec)
        self.ax_wave.set_ylim(-1.5, 1.5)

    def _init_psd_panel(self) -> None:
        self.ax_psd.set_facecolor("#1f242d")
        self.ax_psd.tick_params(colors="#8892b0")
        for spine in self.ax_psd.spines.values():
            spine.set_color("#2d3748")
        self.ax_psd.grid(True, color="#2d3748", linestyle="--", alpha=0.7)
        self.ax_psd.set_title("Power Spectral Density (0.1 - 0.75 Hz)", color="#e2e8f0", fontsize=11, loc="left")
        self.ax_psd.set_xlabel("Frequency (Hz)", color="#8892b0", fontsize=9)
        self.ax_psd.set_ylabel("Power", color="#8892b0", fontsize=9)
        self.ax_psd.axvspan(0.1, 0.75, color="#00ffcc", alpha=0.08)
        self.ax_psd.set_xlim(0.05, 1.0)

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key == "r":
            if not self.engine.is_recording:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                self.engine.start_recording(f"rec_{stamp}")
            else:
                saved = self.engine.stop_recording()
                if saved:
                    print(f"[v] Recording saved to: {saved}")
        elif key == "m":
            new_mode = "amplitude" if self.engine.mode == "phase" else "phase"
            self.engine.set_mode(new_mode)
            print(f"[*] Switched mode to: {new_mode.upper()}")
        elif key == "c":
            self.engine.recalibrate_subcarrier()
            print("[*] Re-calibrating subcarrier ranking...")
        elif key == "[":
            cur = self.engine._active_subcarrier or 0
            new_sc = max(0, cur - 1)
            self.engine.set_subcarrier(new_sc)
            print(f"[*] Switched to Subcarrier: SC {new_sc}")
        elif key == "]":
            cur = self.engine._active_subcarrier or 0
            new_sc = min(63, cur + 1)
            self.engine.set_subcarrier(new_sc)
            print(f"[*] Switched to Subcarrier: SC {new_sc}")
        elif key == "q":
            plt.close(self.fig)

    def update(self, _frame) -> None:
        snapshot = self.engine.compute_snapshot()
        if snapshot is None:
            dur, target, pkts = self.engine.get_buffer_status()
            pct = min(99, int((dur / target) * 100)) if target > 0 else 0
            self.text_bpm.set_text(f"BUF {pct}%")
            self.text_bpm.set_color("#8892b0")
            self.text_status.set_text(f"{dur:.1f}s / {target:.1f}s ({pkts} pkts)")
            self.text_status.set_color("#ffaa00")
            self.text_ai_state.set_text("WARMING UP")
            self.text_ai_state.set_color("#8892b0")
            self.text_ai_bpm.set_text("Waiting for window...")
            return

        # Update Top Info Panel: Recording Tag
        rec_tag = f"🔴 REC ({snapshot.recorded_packets} pkts)" if snapshot.is_recording else "⚪ Standby"
        self.text_rec_tag.set_text(rec_tag)
        self.text_rec_tag.set_color("#ff4757" if snapshot.is_recording else "#a4b0be")

        # Update DSP Vital Sign
        if snapshot.status == "Apnea":
            self.text_bpm.set_text("APNEA")
            self.text_bpm.set_color("#ff4757")
            self.text_status.set_text(f"NÍN THỞ | Var: {snapshot.variance:.4f}")
            self.text_status.set_color("#ff4757")
        else:
            self.text_bpm.set_text(f"{snapshot.bpm:.1f} BPM")
            self.text_bpm.set_color("#00ffcc")
            self.text_status.set_text(f"BREATHING | SNR: {snapshot.snr:.1f}")
            self.text_status.set_color("#2ed573")

        # Update AI Model Prediction
        cls_colors = {
            "apnea": "#ff4757",
            "slow": "#ffa502",
            "normal": "#2ed573",
            "fast": "#00d2d3",
            "unknown": "#8892b0",
        }
        cls_labels = {
            "apnea": "NÍN THỞ (APNEA)",
            "slow": "THỞ CHẬM (SLOW)",
            "normal": "BÌNH THƯỜNG (NORMAL)",
            "fast": "THỞ NHANH (FAST)",
            "unknown": "UNKNOWN",
        }
        ai_cls = snapshot.ai_class
        ai_label = cls_labels.get(ai_cls, ai_cls.upper())
        ai_color = cls_colors.get(ai_cls, "#00ffcc")
        self.text_ai_state.set_text(f"{ai_label}\n({snapshot.ai_confidence * 100:.1f}%)")
        self.text_ai_state.set_color(ai_color)

        if ai_cls == "apnea":
            self.text_ai_bpm.set_text("AI: Zero Respiration")
        else:
            self.text_ai_bpm.set_text(f"AI Rate: {snapshot.ai_bpm:.1f} BPM")

        ch_str = f"PCA-PC{snapshot.selected_subcarrier}" if self.engine._subcarrier_auto_select else f"SC {snapshot.selected_subcarrier}"
        self.text_details.set_text(
            f"Ch: {ch_str} | Fs: {snapshot.fs:.1f}Hz\n"
            f"Spec: {snapshot.fft_bpm:.1f} | ACF: {snapshot.peak_bpm:.1f}\n"
            f"Mode: {snapshot.mode.upper()}\n"
            f"Keys: [R]Rec [M]Mode [C]Cal"
        )

        # Update Waveform
        t = snapshot.time_axis
        sig = snapshot.clean_waveform
        self.line_wave.set_data(t, sig)
        if len(t) > 0:
            self.ax_wave.set_xlim(0, max(self.engine.window_size_sec, float(t[-1])))
            
            # Smooth Y-axis scaling with a minimum floor
            # When holding breath, min_span keeps microscopic noise from filling the screen!
            min_span = 2.0 if snapshot.mode == "amplitude" else 0.4
            sig_min, sig_max = float(np.min(sig)), float(np.max(sig))
            center = (sig_min + sig_max) / 2.0
            span = max(min_span, (sig_max - sig_min) * 1.3)
            target_ymin = center - span / 2.0
            target_ymax = center + span / 2.0

            # Exponential moving smoothing to eliminate violent jumps
            self.cur_ymin = 0.82 * self.cur_ymin + 0.18 * target_ymin
            self.cur_ymax = 0.82 * self.cur_ymax + 0.18 * target_ymax
            self.ax_wave.set_ylim(self.cur_ymin, self.cur_ymax)

        if snapshot.peaks_indices.size > 0:
            pk_times = t[snapshot.peaks_indices]
            pk_vals = sig[snapshot.peaks_indices]
            self.scatter_peaks.set_data(pk_times, pk_vals)
        else:
            self.scatter_peaks.set_data([], [])

        # Update PSD
        freqs = snapshot.fft_freqs
        psd = snapshot.fft_psd
        mask = (freqs >= 0.05) & (freqs <= 1.0)
        if np.any(mask):
            self.line_psd.set_data(freqs[mask], psd[mask])
            self.ax_psd.set_ylim(0, max(0.001, float(np.max(psd[mask])) * 1.25))

        if snapshot.dominant_freq > 0:
            self.vline_dom.set_xdata([snapshot.dominant_freq, snapshot.dominant_freq])
            self.vline_dom.set_alpha(0.85)

    def run(self) -> None:
        self.engine.start()
        # Refresh every 350ms for smooth live rendering
        self.ani = FuncAnimation(self.fig, self.update, interval=350, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self.engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime CSI Breathing Monitoring Dashboard")
    parser.add_argument(
        "--source", choices=["live", "sim"], default="live",
        help="Data source: 'live' (USB Serial, default) or 'sim' (playback .npz file)",
    )
    parser.add_argument(
        "--file", default="test_2.npz",
        help="NPZ file name for simulation (default: test_2.npz)",
    )
    parser.add_argument(
        "--port", default=SERIAL_PORT,
        help=f"Serial port for live mode (default: {SERIAL_PORT})",
    )
    parser.add_argument(
        "--mode", choices=["phase", "amplitude"], default="amplitude",
        help="DSP mode: 'amplitude' (default, recommended for ESP32) or 'phase'",
    )
    parser.add_argument(
        "--window", type=float, default=20.0,
        help="Sliding buffer window size in seconds (default: 20.0)",
    )
    args = parser.parse_args()

    if args.source == "live":
        source = LiveSerialSource(port=args.port, baudrate=BAUD_RATE)
    else:
        source = SimulatedFileSource(file_name_or_path=args.file, loop=True)

    engine = RealtimeEngine(source=source, window_size_sec=args.window, mode=args.mode)
    app = RealtimeDashboard(engine=engine)
    app.run()


if __name__ == "__main__":
    main()
