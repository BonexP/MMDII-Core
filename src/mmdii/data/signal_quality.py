"""Load and quality-gate the primary processed MAT signal stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class SignalQualityConfig:
    """Required field names and numeric comparison tolerances."""

    fields: tuple[str, str, str, str]
    time_step_rtol: float
    time_step_atol: float
    rpm_fs_rtol: float
    rpm_fs_atol: float

    def __post_init__(self) -> None:
        if len(self.fields) != 4 or len(set(self.fields)) != 4:
            raise ValueError("Signal fields must contain four unique names.")
        if any(not field for field in self.fields):
            raise ValueError("Signal field names must not be empty.")
        tolerances = (
            self.time_step_rtol,
            self.time_step_atol,
            self.rpm_fs_rtol,
            self.rpm_fs_atol,
        )
        if any(value < 0 or not np.isfinite(value) for value in tolerances):
            raise ValueError("Signal tolerances must be finite and non-negative.")

    @property
    def force_fields(self) -> tuple[str, str, str]:
        return self.fields[:3]

    @property
    def time_field(self) -> str:
        return self.fields[3]


@dataclass(frozen=True)
class SignalAudit:
    """Accepted arrays or deterministic issue codes for one MAT file."""

    arrays: Mapping[str, np.ndarray] | None
    fs: float | None
    sample_count: int | None
    duration_seconds: float | None
    issues: tuple[str, ...]


def _as_vector(value: object) -> tuple[np.ndarray | None, str | None]:
    array = np.asarray(value)
    if array.size == 0:
        return None, "empty_signal"
    if array.ndim == 1:
        vector = array
    elif array.ndim == 2 and 1 in array.shape:
        vector = array.reshape(-1)
    else:
        return None, "not_vector"
    try:
        return np.asarray(vector, dtype=np.float64), None
    except (TypeError, ValueError):
        return None, "non_finite_value"


def audit_primary_signals(
    path: str | Path,
    *,
    rpm: int,
    config: SignalQualityConfig,
) -> SignalAudit:
    """Load only the primary group and reject any unresolved quality issue."""

    try:
        payload = loadmat(Path(path), variable_names=[*config.fields, "Fs"])
    except Exception:
        return SignalAudit(None, None, None, None, ("mat_read_error",))

    if any(field not in payload for field in config.fields):
        return SignalAudit(None, None, None, None, ("missing_required_field",))

    vectors: dict[str, np.ndarray] = {}
    vector_issues: list[str] = []
    for field in config.fields:
        vector, issue = _as_vector(payload[field])
        if issue is not None and issue not in vector_issues:
            vector_issues.append(issue)
        elif vector is not None:
            vectors[field] = vector
    if vector_issues:
        issue_order = ("empty_signal", "not_vector", "non_finite_value")
        ordered = tuple(issue for issue in issue_order if issue in vector_issues)
        return SignalAudit(None, None, None, None, ordered)

    lengths = {vector.size for vector in vectors.values()}
    if len(lengths) != 1:
        return SignalAudit(None, None, None, None, ("length_mismatch",))
    sample_count = lengths.pop()

    if any(not np.isfinite(vector).all() for vector in vectors.values()):
        return SignalAudit(
            None, None, sample_count, None, ("non_finite_value",)
        )

    fs_value = payload.get("Fs")
    if fs_value is None:
        return SignalAudit(
            None, None, sample_count, None, ("missing_or_invalid_fs",)
        )
    fs_array = np.asarray(fs_value)
    if fs_array.size != 1:
        return SignalAudit(
            None, None, sample_count, None, ("missing_or_invalid_fs",)
        )
    try:
        fs = float(fs_array.reshape(-1)[0])
    except (TypeError, ValueError):
        fs = float("nan")
    if not np.isfinite(fs) or fs <= 0:
        return SignalAudit(
            None, None, sample_count, None, ("missing_or_invalid_fs",)
        )

    time = vectors[config.time_field]
    differences = np.diff(time)
    issues: list[str] = []
    if np.any(differences <= 0):
        issues.append("time_not_increasing")
    if differences.size == 0 or not np.isclose(
        float(np.median(differences)),
        1.0 / fs,
        rtol=config.time_step_rtol,
        atol=config.time_step_atol,
    ):
        issues.append("time_step_mismatch")
    if not np.isclose(
        fs,
        0.6 * rpm,
        rtol=config.rpm_fs_rtol,
        atol=config.rpm_fs_atol,
    ):
        issues.append("rpm_fs_mismatch")

    duration = float(time[-1] - time[0]) if sample_count > 1 else 0.0
    if issues:
        return SignalAudit(None, fs, sample_count, duration, tuple(issues))
    return SignalAudit(vectors, fs, sample_count, duration, ())
