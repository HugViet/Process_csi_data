"""CSI respiration pipeline with MAD channel selection (amplitude or phase mode)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # Supports both package imports and direct script execution.
    from .filters import butterworth_bandpass, detrend_amplitude, remove_spikes, sanitize_phase, select_by_mad
except ImportError:
    from filters import butterworth_bandpass, detrend_amplitude, remove_spikes, sanitize_phase, select_by_mad


class PhaseBeatCore:
    """Preprocess CSI data and select a respiration-band subcarrier.

    Supports two operating modes controlled by the ``mode`` parameter:

    * ``"amplitude"`` (default) — extract amplitude from raw CSI and apply
      spike removal → detrending → band-pass → MAD selection.
    * ``"phase"`` — extract phase from raw CSI, apply phase sanitization
      (unwrap + linear-fit STO/CFO removal) → spike removal → detrending
      → band-pass → MAD selection.

    Both modes share the same filter chain after the initial extraction step.
    """

    VALID_MODES = ("amplitude", "phase")

    def __init__(
        self,
        fs: float = 75.5,
        lowcut: float = 0.1,
        highcut: float = 0.75,
        filter_order: int = 2,
        top_k: int = 3,
        spike_window_seconds: float = 0.3,
        detrend_window_seconds: float = 12.0,
        enable_spike_removal: bool = True,
        enable_detrending: bool = True,
        candidate_subcarriers: ArrayLike | None = None,
        mode: str = "amplitude",
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < lowcut < highcut < fs / 2:
            raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < Nyquist")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.fs = float(fs)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.filter_order = int(filter_order)
        self.top_k = int(top_k)
        self.spike_window_seconds = float(spike_window_seconds)
        self.detrend_window_seconds = float(detrend_window_seconds)
        self.enable_spike_removal = enable_spike_removal
        self.enable_detrending = enable_detrending
        self.mode = mode
        self.candidate_subcarriers = (
            None if candidate_subcarriers is None
            else np.asarray(candidate_subcarriers, dtype=np.int64).reshape(-1)
        )

        # Intermediate results for inspection / plotting.
        self.last_amplitudes: NDArray[np.float64] | None = None
        self.last_raw_phase: NDArray[np.float64] | None = None
        self.last_sanitized_phase: NDArray[np.float64] | None = None
        self.last_extracted: NDArray[np.float64] | None = None
        self.last_spike_removed: NDArray[np.float64] | None = None
        self.last_detrended: NDArray[np.float64] | None = None
        self.last_bandpassed: NDArray[np.float64] | None = None
        self.last_selection_data: NDArray[np.float64] | None = None
        self.last_selected_subcarrier: int | None = None
        self.last_top_subcarriers: NDArray[np.int64] | None = None
        self.last_subcarrier_mads: NDArray[np.float64] | None = None
        self.last_candidate_subcarriers: NDArray[np.int64] | None = None

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

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
    def extract_phase(csi: ArrayLike) -> NDArray[np.float64]:
        """Extract raw phase from complex CSI or I/Q pairs.

        For I/Q input the phase is ``arctan2(Q, I)`` per packet and
        subcarrier.  For pre-extracted real phase arrays the data is
        passed through unchanged.
        """
        values = np.asarray(csi)
        if values.ndim == 3 and values.shape[-1] == 2:
            # I/Q format: shape (samples, subcarriers, 2) where [..,0]=I, [..,1]=Q
            phase = np.arctan2(values[..., 1], values[..., 0])
        elif values.ndim == 2 and np.iscomplexobj(values):
            phase = np.angle(values)
        elif values.ndim == 2:
            # Already extracted phase
            phase = values
        else:
            raise ValueError(
                "csi must be phase/complex samples-by-subcarriers, "
                "or real I/Q samples-by-subcarriers-by-2"
            )

        phase = np.asarray(phase, dtype=np.float64)
        if phase.shape[0] == 0 or phase.shape[1] == 0:
            raise ValueError("CSI input must contain samples and subcarriers")
        if not np.all(np.isfinite(phase)):
            raise ValueError("CSI input must contain only finite values")
        return phase

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

    # ------------------------------------------------------------------
    # ESP32 subcarrier helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esp32_active_subcarriers(n_subs: int) -> NDArray[np.int64]:
        """Return indices of active (non-guard/null/DC) subcarriers for ESP32."""
        if n_subs == 64:
            return np.concatenate((np.arange(6, 32), np.arange(33, 59)))
        return np.arange(n_subs)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        csi: ArrayLike,
        phases: ArrayLike | None = None,
        subcarrier: int | None = None,
    ) -> NDArray[np.float64]:
        """Run the full preprocessing pipeline and return the best 1-D waveform.

        Parameters
        ----------
        csi : array_like
            In ``"amplitude"`` mode: amplitude, complex CSI, or I/Q data with
            shape ``(samples, subcarriers)`` or ``(samples, subcarriers, 2)``.
            In ``"phase"`` mode when ``phases`` is *None*: same as above but
            phase is extracted instead of amplitude.
        phases : array_like or None
            Pre-extracted phase array ``(samples, subcarriers)`` in radians.
            Only used in ``"phase"`` mode.  When provided, ``csi`` is still
            used to compute amplitude for reference (stored in
            ``last_amplitudes``).
        subcarrier : int or None
            Explicit subcarrier index to extract. If *None*, MAD channel ranking
            automatically selects the optimal carrier. Locking the subcarrier
            prevents jumping between subcarriers during realtime sliding-window monitoring.

        Returns
        -------
        NDArray[np.float64]
            One-dimensional cleaned waveform from the selected subcarrier.
        """
        # --- Amplitude is always computed for reference / inspection. ---
        amplitudes = self.extract_amplitude(csi)
        self.last_amplitudes = amplitudes

        # --- Extract the primary signal based on mode. ---
        if self.mode == "phase":
            if phases is not None:
                raw_phase = np.asarray(phases, dtype=np.float64)
                if raw_phase.ndim != 2 or raw_phase.shape[0] == 0 or raw_phase.shape[1] == 0:
                    raise ValueError("phases must be a non-empty samples-by-subcarriers matrix")
                if raw_phase.shape[0] != amplitudes.shape[0]:
                    raise ValueError(
                        f"phases rows ({raw_phase.shape[0]}) must match "
                        f"csi rows ({amplitudes.shape[0]})"
                    )
            else:
                raw_phase = self.extract_phase(csi)
            self.last_raw_phase = raw_phase

            # Phase sanitization: unwrap + remove linear STO/CFO offset.
            sanitized = sanitize_phase(raw_phase)
            self.last_sanitized_phase = sanitized
            extracted = sanitized
        else:
            extracted = amplitudes

        self.last_extracted = extracted

        # --- Shared filter chain (same for amplitude and phase). ---

        # Step 2: Hampel cleanup rejects isolated AGC/packet spikes.
        if self.enable_spike_removal:
            spike_removed = remove_spikes(
                extracted, fs=self.fs, window_seconds=self.spike_window_seconds
            )
        else:
            spike_removed = extracted.copy()
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

        # Step 5: MAD sees only the band-passed matrix, never raw data.
        # ESP 64-point CSI commonly has guard/null edge bins and a DC bin.
        candidate_subcarriers = self.candidate_subcarriers
        if candidate_subcarriers is None:
            candidate_subcarriers = self._esp32_active_subcarriers(bandpassed.shape[1])

        selected, candidates, mads = select_by_mad(
            bandpassed,
            top_k=self.top_k,
            candidate_subcarriers=candidate_subcarriers,
        )
        if subcarrier is not None and 0 <= subcarrier < bandpassed.shape[1]:
            selected = int(subcarrier)

        self.last_selected_subcarrier = selected
        self.last_top_subcarriers = candidates
        self.last_subcarrier_mads = mads
        self.last_candidate_subcarriers = candidate_subcarriers
        self.last_selection_data = bandpassed
        return bandpassed[:, selected].copy()

    def pbCore_process(self, amplitudes: ArrayLike) -> NDArray[np.float64]:
        """Legacy method name; now accepts amplitude/complex CSI, not phase."""
        return self.process(amplitudes)
