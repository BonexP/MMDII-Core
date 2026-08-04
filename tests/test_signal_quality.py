"""Tests for primary processed force-signal quality checks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.signal_quality import SignalQualityConfig, audit_primary_signals


CONFIG = SignalQualityConfig(
    fields=("af", "sf", "axialf", "time"),
    time_step_rtol=1e-6,
    time_step_atol=1e-12,
    rpm_fs_rtol=1e-6,
    rpm_fs_atol=1e-9,
)


def valid_payload(*, rpm: int = 10000, fs: float = 6000.0) -> dict[str, object]:
    time = np.arange(4, dtype=np.float64) / fs
    return {
        "af": np.array([[1.0, 2.0, 3.0, 4.0]]),
        "sf": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "axialf": np.array([[4.0, 3.0, 2.0, 1.0]]),
        "time": time.reshape(-1, 1),
        "Fs": np.array([[fs]]),
    }


class SignalQualityTests(unittest.TestCase):
    def test_accepts_row_and_column_vectors_as_float64(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "valid.mat"
            savemat(path, valid_payload())

            audit = audit_primary_signals(path, rpm=10000, config=CONFIG)

        self.assertEqual(audit.issues, ())
        self.assertEqual(audit.sample_count, 4)
        self.assertAlmostEqual(audit.fs or 0.0, 6000.0)
        self.assertIsNotNone(audit.arrays)
        assert audit.arrays is not None
        self.assertEqual(set(audit.arrays), {"af", "sf", "axialf", "time"})
        for array in audit.arrays.values():
            self.assertEqual(array.shape, (4,))
            self.assertEqual(array.dtype, np.dtype("float64"))

    def test_reports_each_quality_failure_deterministically(self) -> None:
        cases: list[tuple[str, dict[str, object], int, tuple[str, ...]]] = []

        missing = valid_payload()
        del missing["af"]
        cases.append(("missing", missing, 10000, ("missing_required_field",)))

        matrix = valid_payload()
        matrix["af"] = np.ones((2, 2))
        cases.append(("matrix", matrix, 10000, ("not_vector",)))

        empty = valid_payload()
        for field in ("af", "sf", "axialf", "time"):
            empty[field] = np.array([], dtype=np.float64)
        cases.append(("empty", empty, 10000, ("empty_signal",)))

        mismatch = valid_payload()
        mismatch["af"] = np.array([[1.0, 2.0, 3.0]])
        cases.append(("length", mismatch, 10000, ("length_mismatch",)))

        nonfinite = valid_payload()
        nonfinite["sf"] = np.array([[1.0, np.nan, 3.0, 4.0]])
        cases.append(("nonfinite", nonfinite, 10000, ("non_finite_value",)))

        decreasing = valid_payload()
        decreasing["time"] = np.array([[0.0, 0.1, 0.05, 0.2]])
        cases.append(("time_order", decreasing, 10000, ("time_not_increasing", "time_step_mismatch")))

        invalid_fs = valid_payload()
        invalid_fs["Fs"] = np.array([[-1.0]])
        cases.append(("invalid_fs", invalid_fs, 10000, ("missing_or_invalid_fs",)))

        bad_step = valid_payload()
        bad_step["time"] = (np.arange(4) / 5000.0).reshape(1, -1)
        cases.append(("time_step", bad_step, 10000, ("time_step_mismatch",)))

        cases.append(("rpm_fs", valid_payload(), 11000, ("rpm_fs_mismatch",)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, payload, rpm, expected in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.mat"
                    savemat(path, payload)
                    audit = audit_primary_signals(path, rpm=rpm, config=CONFIG)
                    self.assertEqual(audit.issues, expected)
                    self.assertIsNone(audit.arrays)


if __name__ == "__main__":
    unittest.main()
