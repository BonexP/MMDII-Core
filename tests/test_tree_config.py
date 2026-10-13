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
        self.assertEqual(config.n_estimators, 300)
        self.assertEqual(config.min_samples_leaf, 1)

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
""",
            )
            config = load_experiment_config(path)

        self.assertEqual(config.tree.learner, "extra_trees")
        self.assertEqual(config.tree.n_estimators, 1000)
        self.assertEqual(config.tree.min_samples_leaf, 3)

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

    def test_non_positive_tree_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tree.n_estimators"):
            _validate_config(replace(ExperimentConfig.for_test(), tree=TreeConfig(n_estimators=0)))

        with self.assertRaisesRegex(ValueError, "min_samples_leaf"):
            _validate_config(replace(ExperimentConfig.for_test(), tree=TreeConfig(min_samples_leaf=0)))

    def test_tree_settings_are_dumped_into_the_run_config(self) -> None:
        from mmdii.training.cross_validation import _json_config

        payload = _json_config(
            replace(ExperimentConfig.for_test(), tree=TreeConfig(learner="extra_trees", n_estimators=600))
        )

        self.assertEqual(payload["tree"]["learner"], "extra_trees")
        self.assertEqual(payload["tree"]["n_estimators"], 600)
        self.assertEqual(payload["tree"]["min_samples_leaf"], 1)


if __name__ == "__main__":
    unittest.main()
