"""Breathing-rate estimation utilities for CSI signals (FFT and Peak methods)."""

from .breathing_estimator import (
    DEFAULT_BREATH_HIGH_HZ,
    DEFAULT_BREATH_LOW_HZ,
    estimate_breathing_rate,
    estimate_breathing_rate_fft,
    estimate_breathing_rate_peaks,
)

__all__ = [
    "DEFAULT_BREATH_HIGH_HZ",
    "DEFAULT_BREATH_LOW_HZ",
    "estimate_breathing_rate",
    "estimate_breathing_rate_fft",
    "estimate_breathing_rate_peaks",
]
