from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.cross_validation import (  # noqa: E402
    ExperimentConfig,
    TreeConfig,
    _validate_config,
    load_experiment_config,
)

BASE_EXPERIMENT = """
[experiment]
release_directory = "release"
output_directory = "outputs/run"
target_codes = ["flash", "blur", "tunnel"]
mode = "random_forest"
aggregator = "mean"
seed = 7
fold_count = 5
epochs = 2
batch_size = 4
learning_rate = 0.001
weight_decay = 0.0001
device = "cpu"
target_fs = 5400.0
window_seconds = 2.0
stride_seconds = 1.0
full_signal_samples = 256
"""


def _write(temporary_directory: str, extra: str) -> Path:
    path = Path(temporary_directory) / "experiment.toml"
    path.write_text(BASE_EXPERIMENT.strip() + "\n" + extra, encoding="utf-8")
    return path


class TreeConfigTests(unittest.TestCase):
    def test_defaults_match_the_historical_random_forest_behaviour(self) -> None:
        config = TreeConfig()

        self.assertEqual(config.learner, "random_forest")
        self.assertIsNone(config.bootstrap)
        self.assertEqual(config.feature_set, "stats18")
        self.assertEqual(config.n_estimators, 300)
        self.assertEqual(config.min_samples_leaf, 1)
        self.assertEqual(config.max_features, "sqrt")
        self.assertIsNone(config.max_depth)
        self.assertEqual(config.criterion, "gini")
        self.assertEqual(config.class_weight, "balanced")
        self.assertFalse(config.tune)
        self.assertFalse(config.interval_features)
        self.assertFalse(config.process_features)
        self.assertFalse(config.duration_feature)

    def test_absent_tree_table_keeps_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = load_experiment_config(_write(temporary_directory, ""))

        self.assertEqual(config.tree, TreeConfig())

    def test_tree_table_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write(
                temporary_directory,
                """
[tree]
learner = "extra_trees"
n_estimators = 1000
min_samples_leaf = 3
max_features = "all"
max_depth = 8
criterion = "log_loss"
class_weight = "none"
tune = true
bootstrap = false
feature_set = "stats15_no_ptp"
interval_features = true
process_features = true
duration_feature = true
""",
            )
            config = load_experiment_config(path)

        self.assertEqual(config.tree.learner, "extra_trees")
        self.assertEqual(config.tree.n_estimators, 1000)
        self.assertEqual(config.tree.min_samples_leaf, 3)
        self.assertEqual(config.tree.max_features, "all")
        self.assertEqual(config.tree.max_depth, 8)
        self.assertEqual(config.tree.criterion, "log_loss")
        self.assertEqual(config.tree.class_weight, "none")
        self.assertTrue(config.tree.tune)
        self.assertFalse(config.tree.bootstrap)
        self.assertEqual(config.tree.feature_set, "stats15_no_ptp")
        self.assertTrue(config.tree.interval_features)
        self.assertTrue(config.tree.process_features)
        self.assertTrue(config.tree.duration_feature)

    def test_partial_tree_table_falls_back_per_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write(temporary_directory, '\n[tree]\nlearner = "extra_trees"\n')
            config = load_experiment_config(path)

        self.assertEqual(config.tree.learner, "extra_trees")
        self.assertEqual(config.tree.n_estimators, TreeConfig().n_estimators)
        self.assertEqual(config.tree.min_samples_leaf, TreeConfig().min_samples_leaf)

    def test_unknown_learner_is_rejected_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write(temporary_directory, '\n[tree]\nlearner = "gradient_boosting"\n')

            with self.assertRaisesRegex(ValueError, "tree.learner"):
                load_experiment_config(path)

    def test_unknown_feature_set_is_rejected_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write(temporary_directory, '\n[tree]\nfeature_set = "stats17"\n')

            with self.assertRaisesRegex(ValueError, "tree.feature_set"):
                load_experiment_config(path)

    def test_non_boolean_bootstrap_is_rejected_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write(temporary_directory, "\n[tree]\nbootstrap = 1\n")

            with self.assertRaisesRegex(ValueError, "tree.bootstrap"):
                load_experiment_config(path)

    def test_effective_bootstrap_preserves_each_sklearn_default(self) -> None:
        from mmdii.training.cross_validation import _json_config

        random_forest = _json_config(
            replace(
                ExperimentConfig.for_test(),
                mode="random_forest",
                tree=TreeConfig(learner="random_forest"),
            )
        )
        extra_trees = _json_config(
            replace(
                ExperimentConfig.for_test(),
                mode="random_forest",
                tree=TreeConfig(learner="extra_trees"),
            )
        )

        self.assertTrue(random_forest["tree"]["effective_bootstrap"])
        self.assertFalse(extra_trees["tree"]["effective_bootstrap"])

    def test_new_tabular_modes_are_valid(self) -> None:
        for mode in ("gradient_boosting", "rbf_svm"):
            with self.subTest(mode=mode):
                _validate_config(
                    replace(
                        ExperimentConfig.for_test(),
                        mode=mode,
                        tree=TreeConfig(tune=True),
                    )
                )

    def test_non_positive_tree_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tree.n_estimators"):
            _validate_config(replace(ExperimentConfig.for_test(), tree=TreeConfig(n_estimators=0)))

        with self.assertRaisesRegex(ValueError, "min_samples_leaf"):
            _validate_config(replace(ExperimentConfig.for_test(), tree=TreeConfig(min_samples_leaf=0)))

    def test_non_boolean_feature_switches_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            _validate_config(
                replace(
                    ExperimentConfig.for_test(),
                    tree=replace(TreeConfig(), interval_features="yes"),
                )
            )

    def test_tree_settings_are_dumped_into_the_run_config(self) -> None:
        from mmdii.training.cross_validation import _json_config

        payload = _json_config(
            replace(
                ExperimentConfig.for_test(),
                mode="random_forest",
                tree=TreeConfig(learner="extra_trees", n_estimators=600),
            )
        )

        self.assertEqual(payload["tree"]["learner"], "extra_trees")
        self.assertEqual(payload["tree"]["n_estimators"], 600)
        self.assertEqual(payload["tree"]["min_samples_leaf"], 1)
        self.assertEqual(payload["tree"]["max_features"], "sqrt")
        self.assertIsNone(payload["tree"]["max_depth"])
        self.assertEqual(payload["tree"]["criterion"], "gini")
        self.assertEqual(payload["tree"]["class_weight"], "balanced")
        self.assertIsNone(payload["tree"]["bootstrap"])
        self.assertEqual(payload["tree"]["feature_set"], "stats18")
        self.assertFalse(payload["tree"]["tune"])
        self.assertFalse(payload["tree"]["effective_bootstrap"])
        self.assertEqual(
            payload["tree"]["statistical_features"]["feature_count"], 18
        )
        self.assertFalse(
            payload["tree"]["statistical_features"]["blocks"]["interval3"]["enabled"]
        )


if __name__ == "__main__":
    unittest.main()
