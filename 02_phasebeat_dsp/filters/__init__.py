"""Composable filters for CSI respiration processing (amplitude and phase)."""

from .bandpass import butterworth_bandpass
from .detrending import detrend_amplitude
from .mad_selector import select_by_mad
from .phase_sanitizer import sanitize_phase
from .spike_removal import remove_spikes

__all__ = [
    "butterworth_bandpass",
    "detrend_amplitude",
    "sanitize_phase",
    "select_by_mad",
    "remove_spikes",
]
