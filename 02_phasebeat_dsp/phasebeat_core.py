"""Amplitude-based CSI respiration pipeline with MAD channel selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # Supports both package imports and direct script execution.
    from .filters import butterworth_bandpass, detrend_amplitude, remove_spikes, select_by_mad
except ImportError:
    from filters import butterworth_bandpass, detrend_amplitude, remove_spikes, select_by_mad


class PhaseBeatCore:
    """Preprocess CSI amplitudes, then select a respiration-band subcarrier.

    Inputs may be precomputed amplitudes ``(samples, subcarriers)``, complex
    CSI ``(samples, subcarriers)``, or real I/Q pairs
    ``(samples, subcarriers, 2)``. MAD is always calculated on the matrix that
    has passed through the mandatory respiration band-pass stage.
    """

    def __init__(
        self,
        fs: float = 75.5,
        lowcut: float = 0.1,
        highcut: float = 3.0,
        filter_order: int = 2,
        top_k: int = 3,
        spike_window_seconds: float = 0.3,
        detrend_window_seconds: float = 12.0,
        enable_spike_removal: bool = True,
        enable_detrending: bool = True,
        candidate_subcarriers: ArrayLike | None = None,
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < lowcut < highcut < fs / 2:
            raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < Nyquist")

        self.fs = float(fs)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.filter_order = int(filter_order)
        self.top_k = int(top_k)
        self.spike_window_seconds = float(spike_window_seconds)
        self.detrend_window_seconds = float(detrend_window_seconds)
        self.enable_spike_removal = enable_spike_removal
        self.enable_detrending = enable_detrending
        self.candidate_subcarriers = (
            None if candidate_subcarriers is None
            else np.asarray(candidate_subcarriers, dtype=np.int64).reshape(-1)
        )

        self.last_amplitudes: NDArray[np.float64] | None = None
        self.last_spike_removed: NDArray[np.float64] | None = None
        self.last_detrended: NDArray[np.float64] | None = None
        self.last_bandpassed: NDArray[np.float64] | None = None
        self.last_selection_data: NDArray[np.float64] | None = None
        self.last_selected_subcarrier: int | None = None
        self.last_top_subcarriers: NDArray[np.int64] | None = None
        self.last_subcarrier_mads: NDArray[np.float64] | None = None
        self.last_candidate_subcarriers: NDArray[np.int64] | None = None

    @staticmethod
    def extract_amplitude(csi: ArrayLike) -> NDArray[np.float64]:
        """Convert complex CSI or I/Q pairs to magnitude; pass amplitudes through.

        For I/Q input the magnitude is ``sqrt(I**2 + Q**2)`` per packet and
        subcarrier. This is normally already done by the serial reader.
        """
        values = np.asarray(csi)
        if values.ndim == 3 and values.shape[-1] == 2:
            amplitudes = np.hypot(values[..., 0], values[..., 1])
        elif values.ndim == 2 and np.iscomplexobj(values):
            amplitudes = np.abs(values)
        elif values.ndim == 2:
            amplitudes = values
        else:
            raise ValueError(
                "csi must be amplitudes/complex samples-by-subcarriers, "
                "or real I/Q samples-by-subcarriers-by-2"
            )

        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if amplitudes.shape[0] == 0 or amplitudes.shape[1] == 0:
            raise ValueError("CSI input must contain samples and subcarriers")
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError("CSI input must contain only finite values")
        return amplitudes

    @staticmethod
    def pbCore_selectSubcarrierByMAD(
        bandpassed_data: ArrayLike,
        top_k: int = 3,
        candidate_subcarriers: ArrayLike | None = None,
    ) -> tuple[int, NDArray[np.int64], NDArray[np.float64]]:
        """Compatibility wrapper for the MAD selector; input must be filtered."""
        return select_by_mad(
            bandpassed_data,
            top_k=top_k,
            candidate_subcarriers=candidate_subcarriers,
        )

    def process(self, csi: ArrayLike) -> NDArray[np.float64]:
        """Run amplitude extraction → spike cleanup → detrend → band-pass → MAD.

        The highest-MAD candidates are ranked only after band-pass filtering.
        The returned one-dimensional waveform is the selected filtered channel.
        """
        amplitudes = self.extract_amplitude(csi)
        self.last_amplitudes = amplitudes

        # Step 2: Hampel cleanup rejects isolated AGC/packet spikes.
        if self.enable_spike_removal:
            spike_removed = remove_spikes(
                amplitudes, fs=self.fs, window_seconds=self.spike_window_seconds
            )
        else:
            spike_removed = amplitudes.copy()
        self.last_spike_removed = spike_removed

        # Step 3: subtract a smooth baseline to reduce DC and slow drift.
        if self.enable_detrending:
            detrended = detrend_amplitude(
                spike_removed, fs=self.fs, window_seconds=self.detrend_window_seconds
            )
        else:
            detrended = spike_removed - np.median(spike_removed, axis=0, keepdims=True)
        self.last_detrended = detrended

        # Step 4: mandatory respiration-band filtering before channel ranking.
        bandpassed = butterworth_bandpass(
            detrended,
            fs=self.fs,
            lowcut=self.lowcut,
            highcut=self.highcut,
            order=self.filter_order,
        )
        self.last_bandpassed = bandpassed

        # Step 5: MAD sees only the band-passed matrix, never raw amplitudes.
        # ESP 64-point CSI commonly has guard/null edge bins and a DC bin.
        # Keep the project's prior active range by default; callers may pass
        # an explicit carrier list for another bandwidth or chipset layout.
        candidate_subcarriers = self.candidate_subcarriers
        if candidate_subcarriers is None:
            if bandpassed.shape[1] == 64:
                candidate_subcarriers = np.concatenate((np.arange(6, 32), np.arange(33, 59)))
            else:
                candidate_subcarriers = np.arange(bandpassed.shape[1])

        selected, candidates, mads = select_by_mad(
            bandpassed,
            top_k=self.top_k,
            candidate_subcarriers=candidate_subcarriers,
        )
        self.last_selected_subcarrier = selected
        self.last_top_subcarriers = candidates
        self.last_subcarrier_mads = mads
        self.last_candidate_subcarriers = candidate_subcarriers
        self.last_selection_data = bandpassed
        return bandpassed[:, selected].copy()

    def pbCore_process(self, amplitudes: ArrayLike) -> NDArray[np.float64]:
        """Legacy method name; now accepts amplitude/complex CSI, not phase."""
        return self.process(amplitudes)