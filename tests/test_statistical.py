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
    statistical_feature_names,
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

    def test_interval_features_use_elapsed_time_thirds_and_population_std(self) -> None:
        signal = np.tile(np.arange(6, dtype=np.float64), (3, 1))
        times = np.arange(6, dtype=np.float64)
        features = extract_statistical_features(
            signal, time=times, interval_features=True
        )
        self.assertEqual(features.shape, (36,))
        np.testing.assert_allclose(features[18:24], [0.5, 0.5, 2.5, 0.5, 4.5, 0.5])
        self.assertEqual(len(statistical_feature_names(interval_features=True)), 36)

    def test_interval_boundaries_use_nonuniform_timestamps(self) -> None:
        signal = np.tile(np.arange(6, dtype=np.float64), (3, 1))
        times = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 9.0])
        features = extract_statistical_features(
            signal, time=times, interval_features=True
        )
        # Elapsed thirds are 3 and 6 seconds, giving index ranges [0:3], [3:4], [4:6].
        np.testing.assert_allclose(features[18:24], [1.0, np.sqrt(2.0 / 3.0), 3.0, 0.0, 4.5, 0.5])

    def test_process_and_duration_features_are_ordered_and_derived(self) -> None:
        signal = np.tile(np.arange(6, dtype=np.float64), (3, 1))
        features = extract_statistical_features(
            signal,
            time=np.asarray([2.0, 2.5, 3.0, 4.0, 7.0, 9.0]),
            process_parameters={
                "depth_start": 1.25,
                "depth_end": 3.75,
                "rpm": 800.0,
                "welding_speed": 4.5,
            },
            process_features=True,
            duration_feature=True,
        )
        self.assertEqual(features.shape, (23,))
        np.testing.assert_allclose(features[-5:], [1.25, 2.5, 800.0, 4.5, 7.0])
        names = statistical_feature_names(
            process_features=True, duration_feature=True
        )
        self.assertEqual(names[-5:], (
            "depth_start", "plunge_delta", "rpm", "welding_speed", "duration_seconds"
        ))

    def test_metadata_features_reject_invalid_time_empty_intervals_and_process(self) -> None:
        signal = np.ones((3, 4))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_statistical_features(
                signal, time=np.asarray([0.0, 1.0, 1.0, 3.0]), duration_feature=True
            )
        with self.assertRaisesRegex(ValueError, "no signal samples"):
            extract_statistical_features(
                signal, time=np.asarray([0.0, 0.1, 0.2, 100.0]), interval_features=True
            )
        with self.assertRaisesRegex(ValueError, "process_parameters"):
            extract_statistical_features(signal, process_features=True)

    def test_stats18_default_remains_byte_for_byte_same(self) -> None:
        signal = np.asarray([[0.0, 2.0], [-1.0, 3.0]], dtype=np.float64)
        expected = np.asarray(
            [1.0, 1.0, np.sqrt(2.0), 0.0, 2.0, 2.0,
             1.0, 2.0, np.sqrt(5.0), -1.0, 3.0, 4.0]
        )
        np.testing.assert_array_equal(extract_statistical_features(signal), expected)

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
