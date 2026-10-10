from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.cross_validation import (
    ExperimentConfig,
    RepresentationConfig,
    _json_config,
    _time_frequency_datasets,
    load_experiment_config,
    run_cross_validation,
)
from mmdii.data.training_dataset import (
    DatasetIndex,
    FoldNormalizer,
    WeldRecord,
    WeldWindowDataset,
    WindowSpec,
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
encoder_chunk_size = 16
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
            self.assertEqual(config.model.encoder_chunk_size, 16)

    def test_committed_baseline_configuration_has_three_targets(self) -> None:
        config = load_experiment_config(
            CORE_ROOT / "configs" / "moderntcn_mil_v0_1.toml"
        )

        self.assertEqual(config.mode, "window_mil")
        self.assertEqual(config.target_codes, ("flash", "blur", "tunnel"))
        self.assertEqual(config.fold_count, 5)
        self.assertEqual(config.window_seconds, 2.0)

    def test_deep_weld_configuration_only_increases_depth(self) -> None:
        baseline = load_experiment_config(
            CORE_ROOT / "configs" / "moderntcn_mil_weld_independent_v0_2_1.toml"
        )
        deep = load_experiment_config(
            CORE_ROOT / "configs" / "moderntcn_mil_weld_independent_deep_v0_2_1.toml"
        )

        self.assertEqual(deep.model.block_count, 4)
        self.assertEqual(deep.fold_scheme, baseline.fold_scheme)
        self.assertEqual(deep.epochs, baseline.epochs)
        self.assertEqual(deep.model.hidden_channels, baseline.model.hidden_channels)
        self.assertEqual(deep.model.embedding_dim, baseline.model.embedding_dim)
        self.assertEqual(deep.model.kernel_size, baseline.model.kernel_size)

    def test_run_config_records_resolved_representation_parameters(self) -> None:
        base = ExperimentConfig.for_test()
        config = ExperimentConfig(
            **{
                **base.__dict__,
                "representation": RepresentationConfig(
                    name="stft_512",
                    encoder="cnn2d",
                    stft_n_fft=1024,
                    stft_hop_length=256,
                ),
            }
        )

        payload = _json_config(config)

        self.assertEqual(payload["representation_parameters"]["n_fft"], 1024)
        self.assertEqual(payload["representation_parameters"]["hop_length"], 256)
        self.assertEqual(payload["representation_parameters"]["window"], "hann")
        self.assertEqual(
            payload["representation_parameters"]["normalization"],
            "per_channel_zscore",
        )

    def test_training_strategy_defaults_are_backward_compatible(self) -> None:
        config = ExperimentConfig.for_test()
        self.assertEqual(config.training.scheduler, "none")
        self.assertFalse(config.augmentation.enabled)

    def test_time_frequency_cache_uses_weld_window_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "dataset_manifest.json").write_text("{}", encoding="utf-8")
            time = np.arange(256, dtype=np.float64) / 32.0
            records = []
            for index, fold in enumerate((0, 1)):
                sample_id = f"sample-{index}"
                signal_path = root / f"{sample_id}.npz"
                signal = np.vstack(
                    [np.sin(time * (index + channel + 1)) for channel in range(3)]
                )
                np.savez(
                    signal_path,
                    time=time,
                    af=signal[0],
                    sf=signal[1],
                    axialf=signal[2],
                )
                records.append(
                    WeldRecord(
                        sample_id=sample_id,
                        weld_id=sample_id,
                        image_group=sample_id,
                        fold=fold,
                        target=(float(index), 0.0, 0.0),
                        defect_codes=(),
                        is_normal=index == 0,
                        signal_path=signal_path,
                        metadata=(),
                    )
                )
            index = DatasetIndex(root, ("flash", "blur", "tunnel"), tuple(records))
            spec = WindowSpec(target_fs=32.0, window_seconds=8.0, stride_seconds=8.0)
            normalizer = FoldNormalizer.fit(index, (records[1],))
            train = WeldWindowDataset(index, {1}, spec, normalizer)
            valid = WeldWindowDataset(index, {0}, spec, normalizer)
            config = replace(
                ExperimentConfig.for_test(),
                target_fs=32.0,
                window_seconds=8.0,
                stride_seconds=8.0,
                preprocess_cache_directory=root / "cache",
                representation=RepresentationConfig(
                    name="stft_256", encoder="cnn2d", output_time_bins=16
                ),
            )

            wrapped_datasets = []
            try:
                for _ in range(2):
                    train_tf, valid_tf = _time_frequency_datasets(train, valid, config)
                    wrapped_datasets.extend((train_tf, valid_tf))
                    self.assertEqual(train_tf[0]["representations"].shape[-1], 16)
                    self.assertEqual(valid_tf[0]["representations"].shape[-1], 16)

                cache_entries = list((root / "cache").iterdir())
                self.assertEqual(len(cache_entries), 2)
                self.assertTrue(all((entry / "COMPLETE").is_file() for entry in cache_entries))
            finally:
                for dataset in wrapped_datasets:
                    for arrays in dataset._precomputed.values():
                        for array in arrays:
                            if isinstance(array, np.memmap):
                                array._mmap.close()

    def test_loads_training_and_augmentation_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "experiment.toml"
            path.write_text(
                """
[experiment]
release_directory = "release"
output_directory = "outputs/run"
target_codes = ["flash", "blur", "tunnel"]
mode = "window_mil"
aggregator = "mean"
seed = 7
fold_count = 5
epochs = 4
batch_size = 2
learning_rate = 0.001
weight_decay = 0.0001
device = "cpu"
target_fs = 5400.0
window_seconds = 2.0
stride_seconds = 1.0
full_signal_samples = 256

[training]
scheduler = "cosine"
warmup_epochs = 1
gradient_clip_norm = 1.0
amp = false

[augmentation]
enabled = true
amplitude_scale = 0.02
noise_std = 0.01
time_mask_ratio = 0.1
max_time_masks = 1
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_experiment_config(path)
            self.assertEqual(config.training.scheduler, "cosine")
            self.assertTrue(config.augmentation.enabled)

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
