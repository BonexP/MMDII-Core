from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.models.statistical import (  # noqa: E402
    LEARNERS,
    fit_predict_extra_trees_ovr,
    fit_predict_forest_ovr,
    fit_predict_random_forest_ovr,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def _toy_problem():
    train_features = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)
    train_targets = np.array(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    test_features = np.array([[0.05], [1.05]], dtype=np.float64)
    return train_features, train_targets, test_features


class TreeLearnerTests(unittest.TestCase):
    def test_learner_names_are_the_supported_pair(self) -> None:
        self.assertEqual(LEARNERS, ("random_forest", "extra_trees"))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_extra_trees_returns_one_probability_per_sample_and_target(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        probabilities = fit_predict_extra_trees_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
        )

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_extra_trees_differs_from_random_forest_but_shares_the_contract(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        forest = fit_predict_random_forest_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
        )
        extra = fit_predict_extra_trees_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
        )

        self.assertEqual(forest.shape, extra.shape)
        self.assertFalse(np.array_equal(forest, extra))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_learner_argument_selects_the_same_path_as_the_wrappers(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        direct = fit_predict_forest_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            learner="extra_trees",
            n_estimators=8,
        )
        wrapper = fit_predict_extra_trees_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
        )

        np.testing.assert_array_equal(direct, wrapper)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_min_samples_leaf_is_forwarded_rather_than_ignored(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        single = fit_predict_extra_trees_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
            min_samples_leaf=1,
        )
        coarse = fit_predict_extra_trees_ovr(
            train_features,
            train_targets,
            test_features,
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=8,
            min_samples_leaf=20,
        )

        self.assertFalse(np.array_equal(single, coarse))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_rejects_unknown_learner(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        with self.assertRaisesRegex(ValueError, "tree-ensemble settings"):
            fit_predict_forest_ovr(
                train_features,
                train_targets,
                test_features,
                target_codes=("flash", "blur"),
                random_state=7,
                learner="gradient_boosting",
            )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_rejects_non_positive_min_samples_leaf(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        with self.assertRaisesRegex(ValueError, "tree-ensemble settings"):
            fit_predict_extra_trees_ovr(
                train_features,
                train_targets,
                test_features,
                target_codes=("flash", "blur"),
                random_state=7,
                n_estimators=8,
                min_samples_leaf=0,
            )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_rejects_one_class_training_target(self) -> None:
        train_features, train_targets, test_features = _toy_problem()

        with self.assertRaisesRegex(ValueError, "one class"):
            fit_predict_extra_trees_ovr(
                train_features,
                np.zeros_like(train_targets),
                test_features,
                target_codes=("flash", "blur"),
                random_state=7,
                n_estimators=8,
            )


if __name__ == "__main__":
    unittest.main()
