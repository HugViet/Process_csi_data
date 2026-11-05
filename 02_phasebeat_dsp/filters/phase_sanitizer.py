"""Phase sanitization for single-antenna CSI: unwrap and remove linear STO/CFO distortion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sanitize_phase(
    raw_phase: ArrayLike,
    subcarrier_indices: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Remove hardware-induced linear phase offset across subcarriers.

    On commodity WiFi hardware (including single-antenna ESP32), the measured
    phase is corrupted by Sampling Time Offset (STO) and Carrier Frequency
    Offset (CFO)::

        φ_measured[k] = φ_true[k] - 2π·δt·f[k] - β

    Since both STO and CFO contribute a *linear function* of subcarrier index,
    fitting and subtracting a line across subcarriers at each time step recovers
    the true phase variation (up to an irrelevant constant).

    Parameters
    ----------
    raw_phase : array_like, shape (samples, subcarriers)
        Raw CSI phase in radians. Each row is one packet, each column is one
        subcarrier.
    subcarrier_indices : array_like or None
        Integer indices of the subcarriers present in ``raw_phase`` columns.
        Used as the x-axis for the linear fit. If *None*, ``0, 1, …, N-1`` is
        used. For ESP32 64-point FFT data where only active subcarriers are
        kept, pass the original subcarrier indices so the fit models the true
        frequency spacing.

    Returns
    -------
    NDArray[np.float64], shape (samples, subcarriers)
        Sanitized phase with the per-packet linear trend removed.
    """
    phase = np.asarray(raw_phase, dtype=np.float64)
    if phase.ndim != 2 or phase.shape[0] == 0 or phase.shape[1] == 0:
        raise ValueError("raw_phase must be a non-empty samples-by-subcarriers matrix")

    n_samples, n_subs = phase.shape

    # Step 1+2: Spatial unwrap per packet (fix 2π jumps across subcarriers).
    # NOTE: temporal unwrap must NOT be applied to raw phase first - every
    # packet carries a random CFO/STO offset, so raw phase is uncorrelated in
    # time until the per-packet linear trend has been removed below.
    unwrapped = np.unwrap(phase, axis=1)

    # Step 3: Build the x-axis for the linear fit.
    if subcarrier_indices is not None:
        k = np.asarray(subcarrier_indices, dtype=np.float64).ravel()
        if k.size != n_subs:
            raise ValueError(
                f"subcarrier_indices length ({k.size}) must match the number "
                f"of subcarrier columns ({n_subs})"
            )
    else:
        k = np.arange(n_subs, dtype=np.float64)

    # Step 4: Linear least-squares fit per time step.
    # For each row: φ_measured[k] ≈ a·k + b
    # We subtract the fitted line to remove STO/CFO distortion.
    #
    # Vectorized: X = [k, 1], coeffs = (X^T X)^-1 X^T · φ^T
    X = np.column_stack([k, np.ones_like(k)])  # (n_subs, 2)
    # coeffs shape: (2, n_samples) — [slope; intercept] for each time step
    coeffs, _, _, _ = np.linalg.lstsq(X, unwrapped.T, rcond=None)
    linear_fit = (X @ coeffs).T  # (n_samples, n_subs)

    sanitized = unwrapped - linear_fit

    if not np.all(np.isfinite(sanitized)):
        raise ValueError(
            "Phase sanitization produced non-finite values; "
            "check for corrupted input data"
        )

    return sanitized
