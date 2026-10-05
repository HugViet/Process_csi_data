"""Composable filters for CSI amplitude respiration processing."""

from .bandpass import butterworth_bandpass
from .detrending import detrend_amplitude
from .mad_selector import select_by_mad
from .spike_removal import remove_spikes

__all__ = [
    "butterworth_bandpass",
    "detrend_amplitude",
    "select_by_mad",
    "remove_spikes",
]
