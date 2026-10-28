from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import ModuleType

import numpy as np

SKLEARN_AVAILABLE = __import__("importlib").util.find_spec("sklearn") is not None


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.models.statistical import (  # noqa: E402
    extract_statistical_features,
    fit_predict_forest_ovr,
    fit_predict_gradient_boosting_ovr,
    fit_predict_rbf_svm_ovr,
)
from mmdii.training import tree_tuning  # noqa: E402


def _fake_sklearn(*modules: tuple[str, ModuleType]):
    root = ModuleType("sklearn")
    root.__path__ = []
    entries = {"sklearn": root}
    for name, module in modules:
        module.__package__ = "sklearn"
        entries[f"sklearn.{name}"] = module
        setattr(root, name, module)
    return patch.dict(sys.modules, entries)


class V032StatisticalPrimitiveTests(unittest.TestCase):
    def test_feature_sets_keep_channel_order_and_remove_only_named_features(self) -> None:
        signal = np.asarray(
            [[-2.0, 0.0, 2.0], [1.0, 2.0, 3.0], [-1.0, -1.0, 1.0]]
        )

        full = extract_statistical_features(signal, feature_set="stats18")
        no_ptp = extract_statistical_features(signal, feature_set="stats15_no_ptp")
        no_ptp_rms = extract_statistical_features(
            signal, feature_set="stats12_no_ptp_rms"
        )

        self.assertEqual(full.shape, (18,))
        self.assertEqual(no_ptp.shape, (15,))
        self.assertEqual(no_ptp_rms.shape, (12,))
        np.testing.assert_array_equal(no_ptp.reshape(3, 5), full.reshape(3, 6)[:, :5])
        np.testing.assert_array_equal(
            no_ptp_rms.reshape(3, 4), full.reshape(3, 6)[:, [0, 1, 3, 4]]
        )

    def test_unknown_feature_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_set"):
            extract_statistical_features(np.ones((3, 4)), feature_set="stats17")

    def test_unspecified_bootstrap_is_omitted_from_estimator_kwargs(self) -> None:
        train_x = np.asarray([[0.0], [0.1], [1.0], [1.1]])
        train_y = np.asarray([[0.0], [0.0], [1.0], [1.0]])

        class FakeClassifier:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def fit(self, _features, _target):
                return self

            def predict_proba(self, features):
                return np.tile((0.4, 0.6), (len(features), 1))

        ensemble = ModuleType("ensemble")
        ensemble.RandomForestClassifier = FakeClassifier
        ensemble.ExtraTreesClassifier = FakeClassifier
        with _fake_sklearn(("ensemble", ensemble)):
            fit_predict_forest_ovr(
                train_x,
                train_y,
                np.asarray([[0.8]]),
                target_codes=("flash",),
                random_state=7,
                bootstrap=None,
            )

        self.assertNotIn("bootstrap", FakeClassifier.instances[-1].kwargs)

    def test_explicit_bootstrap_is_forwarded(self) -> None:
        train_x = np.asarray([[0.0], [0.1], [1.0], [1.1]])
        train_y = np.asarray([[0.0], [0.0], [1.0], [1.0]])

        class FakeClassifier:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def fit(self, _features, _target):
                return self

            def predict_proba(self, features):
                return np.tile((0.4, 0.6), (len(features), 1))

        ensemble = ModuleType("ensemble")
        ensemble.RandomForestClassifier = FakeClassifier
        ensemble.ExtraTreesClassifier = FakeClassifier
        with _fake_sklearn(("ensemble", ensemble)):
            fit_predict_forest_ovr(
                train_x,
                train_y,
                np.asarray([[0.8]]),
                target_codes=("flash",),
                random_state=7,
                learner="extra_trees",
                bootstrap=True,
            )

        self.assertIs(FakeClassifier.instances[-1].kwargs["bootstrap"], True)

    def test_gradient_boosting_passes_balanced_weights_from_fit_rows(self) -> None:
        train_x = np.arange(4, dtype=np.float64).reshape(-1, 1)
        train_y = np.asarray([[0.0], [0.0], [0.0], [1.0]])

        class FakeClassifier:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def fit(self, _features, _target, *, sample_weight):
                self.sample_weight = sample_weight
                return self

            def predict_proba(self, features):
                return np.tile((0.4, 0.6), (len(features), 1))

        ensemble = ModuleType("ensemble")
        ensemble.GradientBoostingClassifier = FakeClassifier
        with _fake_sklearn(("ensemble", ensemble)):
            result = fit_predict_gradient_boosting_ovr(
                train_x,
                train_y,
                np.asarray([[1.0]]),
                target_codes=("flash",),
                random_state=7,
                n_estimators=5,
            )

        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_allclose(
            FakeClassifier.instances[0].sample_weight,
            [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0],
        )
        self.assertEqual(FakeClassifier.instances[0].kwargs["n_estimators"], 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_gradient_boosting_returns_one_probability_per_target(self) -> None:
        train_x = np.asarray([[0.0], [0.1], [1.0], [1.1], [0.2], [0.9]])
        train_y = np.asarray(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )
        probabilities = fit_predict_gradient_boosting_ovr(
            train_x,
            train_y,
            np.asarray([[0.05], [1.05]]),
            target_codes=("flash", "blur"),
            random_state=7,
            n_estimators=5,
        )

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_rbf_svm_returns_raw_scores_and_calibrated_probabilities(self) -> None:
        train_x = np.asarray(
            [[float(index), float(index % 3)] for index in range(24)]
        )
        train_y = np.asarray(
            [[float(index % 2), float((index // 2) % 2)] for index in range(24)]
        )
        train_fold_ids = np.repeat(np.arange(4), 6)
        fitted_calibrators = []
        predefined_splits = []

        class FakePipeline:
            def fit(self, features, target):
                self.offset = float(np.mean(features[:, 0]) + np.mean(target))
                return self

            def decision_function(self, features):
                return np.asarray(features[:, 0] - self.offset)

        class FakeCalibratedClassifier:
            def __init__(self, *, estimator, method, cv, ensemble):
                self.estimator = estimator
                self.method = method
                self.cv = cv
                self.ensemble = ensemble
                fitted_calibrators.append(self)

            def fit(self, _features, _target):
                return self

            def predict_proba(self, features):
                return np.tile((0.4, 0.6), (len(features), 1))

        class FakePredefinedSplit:
            def __init__(self, *, test_fold):
                self.test_fold = test_fold
                predefined_splits.append(self)

            def split(self):
                for fold in np.unique(self.test_fold):
                    validation = np.flatnonzero(self.test_fold == fold)
                    training = np.flatnonzero(self.test_fold != fold)
                    yield training, validation

        pipeline_module = ModuleType("pipeline")
        pipeline_module.make_pipeline = lambda *_parts: FakePipeline()
        preprocessing_module = ModuleType("preprocessing")
        preprocessing_module.StandardScaler = type("StandardScaler", (), {})
        svm_module = ModuleType("svm")
        svm_module.SVC = lambda **_kwargs: object()
        calibration_module = ModuleType("calibration")
        calibration_module.CalibratedClassifierCV = FakeCalibratedClassifier
        model_selection_module = ModuleType("model_selection")
        model_selection_module.PredefinedSplit = FakePredefinedSplit

        with _fake_sklearn(
            ("pipeline", pipeline_module),
            ("preprocessing", preprocessing_module),
            ("svm", svm_module),
            ("calibration", calibration_module),
            ("model_selection", model_selection_module),
        ):
            scores, probabilities = fit_predict_rbf_svm_ovr(
                train_x,
                train_y,
                np.asarray([[1.5, 1.0], [18.0, 0.0]]),
                train_fold_ids=train_fold_ids,
                target_codes=("flash", "blur"),
                random_state=7,
                c_value=1.0,
            )

        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertEqual(len(fitted_calibrators), 2)
        self.assertTrue(all(item.method == "sigmoid" for item in fitted_calibrators))
        self.assertTrue(all(item.ensemble is False for item in fitted_calibrators))
        np.testing.assert_array_equal(
            predefined_splits[0].test_fold,
            np.repeat(np.arange(4), 6),
        )
        for calibrator in fitted_calibrators:
            for training_rows, validation_rows in calibrator.cv.split():
                self.assertEqual(len(training_rows) + len(validation_rows), len(train_x))
                self.assertFalse(
                    set(calibrator.cv.test_fold[training_rows])
                    & set(calibrator.cv.test_fold[validation_rows])
                )

    def test_gradient_boosting_tuning_uses_only_inner_training_rows(self) -> None:
        features = np.column_stack((np.repeat((1, 2, 3, 4), 4), np.arange(16)))
        targets = np.asarray(
            [
                [position % 2, (position // 2) % 2]
                for _fold in range(4)
                for position in range(4)
            ],
            dtype=np.float64,
        )
        fold_ids = features[:, 0]
        calls = []

        def fake_fit(train_x, train_y, test_x, **kwargs):
            calls.append((train_x.copy(), test_x.copy(), kwargs))
            return np.full((len(test_x), 2), 0.5)

        with patch(
            "mmdii.training.tree_tuning.fit_predict_gradient_boosting_ovr",
            side_effect=fake_fit,
        ):
            with patch(
                "mmdii.training.tree_tuning._macro_average_precision",
                return_value=0.5,
            ):
                probabilities, diagnostics = (
                    tree_tuning.tune_and_predict_gradient_boosting_ovr(
                        features,
                        targets,
                        np.zeros((2, 2)),
                        train_fold_ids=fold_ids,
                        target_codes=("flash", "blur"),
                        random_state=7,
                    )
                )

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(diagnostics["candidate_count"], 4)
        self.assertEqual(len(calls), 4 * 4 + 1)
        for train_x, valid_x, _kwargs in calls[:-1]:
            self.assertFalse(set(train_x[:, 0]) & set(valid_x[:, 0]))

    def test_svm_tuning_uses_raw_scores_for_selection_and_calibrates_once(self) -> None:
        features = np.column_stack((np.repeat((1, 2, 3, 4), 4), np.arange(16)))
        targets = np.asarray(
            [
                [position % 2, (position // 2) % 2]
                for _fold in range(4)
                for position in range(4)
            ],
            dtype=np.float64,
        )
        fold_ids = features[:, 0]
        calls = []

        def fake_scores(train_x, train_y, test_x, **kwargs):
            calls.append((train_x.copy(), test_x.copy(), kwargs))
            return np.zeros((len(test_x), 2))

        with patch(
            "mmdii.training.tree_tuning.fit_predict_rbf_svm_scores_ovr",
            side_effect=fake_scores,
        ):
            with patch(
                "mmdii.training.tree_tuning.fit_predict_rbf_svm_ovr",
                return_value=(np.ones((2, 2)), np.full((2, 2), 0.5)),
            ) as final_fit:
                with patch(
                    "mmdii.training.tree_tuning._macro_average_precision",
                    return_value=0.5,
                ):
                    scores, probabilities, diagnostics = (
                        tree_tuning.tune_and_predict_rbf_svm_ovr(
                            features,
                            targets,
                            np.zeros((2, 2)),
                            train_fold_ids=fold_ids,
                            target_codes=("flash", "blur"),
                            random_state=7,
                        )
                    )

        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(diagnostics["candidate_count"], 3)
        self.assertEqual(len(calls), 3 * 4)
        self.assertEqual(final_fit.call_count, 1)
        self.assertEqual(final_fit.call_args.kwargs["c_value"], 0.1)
        for train_x, valid_x, _kwargs in calls:
            self.assertFalse(set(train_x[:, 0]) & set(valid_x[:, 0]))


if __name__ == "__main__":
    unittest.main()
