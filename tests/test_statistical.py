from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.models.statistical import (
    STATISTIC_NAMES,
    extract_statistical_features,
    fit_predict_logistic_ovr,
    fit_predict_random_forest_ovr,
)


class StatisticalBaselineTests(unittest.TestCase):
    def test_extracts_fixed_channel_statistics(self) -> None:
        signal = np.array(
            [[0.0, 2.0], [-1.0, 3.0]],
            dtype=np.float64,
        )

        features = extract_statistical_features(signal)

        self.assertEqual(
            STATISTIC_NAMES,
            ("mean", "std", "rms", "minimum", "maximum", "peak_to_peak"),
        )
        self.assertEqual(features.shape, (12,))
        np.testing.assert_allclose(
            features[:6],
            [1.0, 1.0, np.sqrt(2.0), 0.0, 2.0, 2.0],
        )

    def test_rejects_one_class_training_target(self) -> None:
        features = np.array([[0.0], [1.0]], dtype=np.float64)
        targets = np.zeros((2, 1), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "one class"):
            fit_predict_logistic_ovr(
                features,
                targets,
                features,
                target_codes=("flash",),
                random_state=7,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None,
        "scikit-learn train extra is not installed",
    )
    def test_returns_one_probability_per_sample_and_target(self) -> None:
        train_features = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)
        train_targets = np.array(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )

        probabilities = fit_predict_logistic_ovr(
            train_features,
            train_targets,
            np.array([[0.05], [1.05]], dtype=np.float64),
            target_codes=("flash", "blur"),
            random_state=7,
        )

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None,
        "scikit-learn train extra is not installed",
    )
    def test_random_forest_returns_one_probability_per_sample_and_target(self) -> None:
        train_features = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)
        train_targets = np.array(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        probabilities = fit_predict_random_forest_ovr(
            train_features,
            train_targets,
            np.array([[0.05], [1.05]], dtype=np.float64),
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
        )
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))


if __name__ == "__main__":
    unittest.main()
