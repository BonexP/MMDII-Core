from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.signal_spectrum import audit_signal_spectrum, recommend_target_fs


class SignalSpectrumTests(unittest.TestCase):
    def test_audits_sine_bandwidth_and_recommends_rounded_target_fs(self) -> None:
        fs = 6000.0
        time = np.arange(6000, dtype=np.float64) / fs
        signal = np.sin(2.0 * np.pi * 50.0 * time)

        row = audit_signal_spectrum(
            sample_id="sample-1",
            channel="af",
            signal=signal,
            fs=fs,
            duration_seconds=1.0,
            energy_fraction=0.995,
        )

        self.assertLess(abs(row.energy_cutoff_hz - 50.0), 3.0)
        self.assertEqual(row.nyquist_hz, 3000.0)
        self.assertEqual(
            recommend_target_fs(
                [row], record_percentile=0.95, nyquist_margin=1.25
            ),
            200.0,
        )

    def test_rejects_invalid_or_zero_energy_signals(self) -> None:
        values = (
            np.array([], dtype=np.float64),
            np.array([0.0, np.nan], dtype=np.float64),
            np.zeros(16, dtype=np.float64),
        )
        for signal in values:
            with self.subTest(signal=signal):
                with self.assertRaises(ValueError):
                    audit_signal_spectrum(
                        sample_id="sample-1",
                        channel="af",
                        signal=signal,
                        fs=6000.0,
                        duration_seconds=1.0,
                        energy_fraction=0.995,
                    )


if __name__ == "__main__":
    unittest.main()
