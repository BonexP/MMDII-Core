from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.cross_validation import (
    ExperimentConfig,
    load_experiment_config,
    run_cross_validation,
)


class CrossValidationTests(unittest.TestCase):
    def test_loads_explicit_experiment_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "experiment.toml"
            path.write_text(
                """
[experiment]
release_directory = "release"
output_directory = "outputs/run"
target_codes = ["flash", "blur", "tunnel"]
mode = "window_mil"
aggregator = "gated_attention"
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

[model]
hidden_channels = 8
embedding_dim = 12
kernel_size = 7
block_count = 2
dropout = 0.0
top_k = 2
attention_dim = 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_experiment_config(path)

            self.assertIsInstance(config, ExperimentConfig)
            self.assertEqual(config.release_directory, (path.parent / "release").resolve())
            self.assertEqual(config.target_codes, ("flash", "blur", "tunnel"))
            self.assertEqual(config.mode, "window_mil")
            self.assertEqual(config.aggregator, "gated_attention")
            self.assertEqual(config.fold_count, 5)
            self.assertEqual(config.model.embedding_dim, 12)

    def test_committed_baseline_configuration_has_three_targets(self) -> None:
        config = load_experiment_config(
            CORE_ROOT / "configs" / "moderntcn_mil_v0_1.toml"
        )

        self.assertEqual(config.mode, "window_mil")
        self.assertEqual(config.target_codes, ("flash", "blur", "tunnel"))
        self.assertEqual(config.fold_count, 5)
        self.assertEqual(config.window_seconds, 2.0)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is not None,
        "PyTorch train extra is installed",
    )
    def test_runner_requires_train_extra(self) -> None:
        config = ExperimentConfig.for_test()

        with self.assertRaisesRegex(RuntimeError, "train extra"):
            run_cross_validation(None, config)


if __name__ == "__main__":
    unittest.main()
