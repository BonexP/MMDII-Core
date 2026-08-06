"""Audit force-signal bandwidth without changing source arrays."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class SpectrumAudit:
    sample_id: str
    channel: str
    original_fs_hz: float
    duration_seconds: float
    energy_fraction: float
    energy_cutoff_hz: float
    nyquist_hz: float


def audit_signal_spectrum(
    *,
    sample_id: str,
    channel: str,
    signal: np.ndarray,
    fs: float,
    duration_seconds: float,
    energy_fraction: float,
) -> SpectrumAudit:
    """Return the frequency containing the requested fraction of AC power."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Signal must be a finite one-dimensional array.")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite.")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite.")
    if not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must be in (0, 1].")

    frequencies, power = welch(
        values,
        fs=fs,
        detrend="constant",
        nperseg=min(8192, values.size),
    )
    total_power = float(np.sum(power))
    if not math.isfinite(total_power) or total_power <= 0:
        raise ValueError("Signal has no finite non-zero spectral power.")
    cumulative_power = np.cumsum(power)
    index = int(np.searchsorted(cumulative_power, energy_fraction * total_power))
    index = min(index, frequencies.size - 1)
    return SpectrumAudit(
        sample_id=sample_id,
        channel=channel,
        original_fs_hz=float(fs),
        duration_seconds=float(duration_seconds),
        energy_fraction=float(energy_fraction),
        energy_cutoff_hz=float(frequencies[index]),
        nyquist_hz=float(fs / 2.0),
    )


def recommend_target_fs(
    rows: Iterable[SpectrumAudit],
    *,
    record_percentile: float,
    nyquist_margin: float,
) -> float:
    """Recommend a common rate rounded up to 100 Hz and capped by source rates."""

    audits = tuple(rows)
    if not audits:
        raise ValueError("At least one spectrum audit is required.")
    if not 0 < record_percentile <= 1:
        raise ValueError("record_percentile must be in (0, 1].")
    if not math.isfinite(nyquist_margin) or nyquist_margin < 1:
        raise ValueError("nyquist_margin must be at least 1.")

    cutoff = float(
        np.quantile(
            [row.energy_cutoff_hz for row in audits],
            record_percentile,
        )
    )
    required = 2.0 * nyquist_margin * cutoff
    rounded = math.ceil(required / 100.0) * 100.0
    minimum_source_fs = min(2.0 * row.nyquist_hz for row in audits)
    return float(min(rounded, minimum_source_fs))
