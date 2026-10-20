from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.cross_validation import (
    ExperimentConfig,
    RepresentationConfig,
    SecondStageConfig,
    TreeConfig,
    _json_config,
    _cache_dataset_on_gpu,
    _torch_collate,
    _time_frequency_datasets,
    _validate_config,
    load_experiment_config,
    run_cross_validation,
)
import mmdii.training.cross_validation as cross_validation
from mmdii.data.training_dataset import (
    DatasetIndex,
    FoldNormalizer,
    WeldRecord,
    WeldWindowDataset,
    WindowSpec,
)


class CrossValidationTests(unittest.TestCase):
    def test_v031_encoder_variants_pass_experiment_config_validation(self) -> None:
        for encoder in (
            "resnet2d_small",
            "resnet2d_wide",
            "resnet2d_separable",
            "convnext2d_lite",
            "convnext2d_deep",
        ):
            with self.subTest(encoder=encoder):
                config = replace(
                    ExperimentConfig.for_test(),
                    mode="window_mil",
                    representation=RepresentationConfig(
                        name="dwt_swt_db4",
                        encoder=encoder,
                        fusion="raw_plus_dwt",
                    ),
                )
                _validate_config(config)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_model_casts_float64_signal_to_encoder_parameter_dtype(self) -> None:
        import torch

        config = replace(
            ExperimentConfig.for_test(),
            mode="window_mil",
            representation=RepresentationConfig(name="raw"),
        )
        model = cross_validation._build_deep_model(config, "mean", torch)
        windows = torch.ones((1, 1, 3, 32), dtype=torch.float64)
        window_mask = torch.ones((1, 1), dtype=torch.bool)
        sample_mask = torch.ones((1, 1, 32), dtype=torch.bool)

        logits, _ = model(windows, window_mask, sample_mask)

        self.assertEqual(logits.dtype, torch.float32)

    def test_torch_batch_contract_uses_model_dtypes(self) -> None:
        import torch

        item = {
            "windows": np.ones((2, 3, 8), dtype=np.float64),
            "window_mask": np.array([True, True]),
            "sample_mask": np.ones((2, 8), dtype=bool),
            "starts_seconds": np.array([0.0, 1.0], dtype=np.float64),
            "targets": np.array([1.0, 0.0, 1.0], dtype=np.float64),
            "sample_id": "sample-0",
            "weld_id": "weld-0",
            "image_group": "image-0",
            "fold": 0,
        }

        batch = _torch_collate([item])

        self.assertEqual(batch["windows"].dtype, torch.float32)
        self.assertEqual(batch["targets"].dtype, torch.float32)
        self.assertEqual(batch["window_mask"].dtype, torch.bool)
        self.assertEqual(batch["sample_mask"].dtype, torch.bool)

    def test_gpu_dataset_cache_uses_model_dtypes(self) -> None:
        import torch

        item = {
            "windows": np.ones((2, 3, 8), dtype=np.float64),
            "window_mask": np.array([True, True]),
            "sample_mask": np.ones((2, 8), dtype=bool),
            "targets": np.array([1.0, 0.0, 1.0], dtype=np.float64),
            "representations": np.ones((2, 3, 4, 5), dtype=np.float64),
            "sample_id": "sample-0",
            "weld_id": "weld-0",
            "image_group": "image-0",
            "fold": 0,
        }

        cached = _cache_dataset_on_gpu([item], torch.device("cpu"), torch)

        self.assertEqual(cached["windows"][0].dtype, torch.float32)
        self.assertEqual(cached["targets"][0].dtype, torch.float32)
        self.assertEqual(cached["representations"][0].dtype, torch.float32)
        self.assertEqual(cached["window_mask"][0].dtype, torch.bool)
        self.assertEqual(cached["sample_mask"][0].dtype, torch.bool)

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

    def test_raw_dwt_fusion_is_a_valid_representation_contract(self) -> None:
        config = replace(
            ExperimentConfig.for_test(),
            representation=RepresentationConfig(
                name="dwt_swt_db4",
                encoder="cnn2d",
                fusion="raw_plus_dwt",
            ),
        )

        _validate_config(config)
        self.assertEqual(_json_config(config)["representation"]["fusion"], "raw_plus_dwt")

    def test_raw_dwt_fusion_rejects_a_non_dwt_representation(self) -> None:
        config = replace(
            ExperimentConfig.for_test(),
            representation=RepresentationConfig(
                name="cwt_morl",
                encoder="cnn2d",
                fusion="raw_plus_dwt",
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires dwt_swt_db4"):
            _validate_config(config)

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
                device="cpu",
                pipeline_backend="cuda",
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
                with patch.dict(os.environ, {"MMDII_GPU_STFT": "0"}):
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_gpu_stft_cufft_error_falls_back_to_cpu(self) -> None:
        import torch

        config = replace(
            ExperimentConfig.for_test(),
            device="auto",
            pipeline_backend="cuda",
            target_fs=32.0,
            representation=RepresentationConfig(
                name="stft_256", encoder="cnn2d", output_time_bins=16
            ),
        )
        dataset = [
            {
                "windows": np.ones((1, 3, 256), dtype=np.float64),
                "sample_mask": np.ones((1, 256), dtype=bool),
            }
        ]

        with (
            patch.dict(os.environ, {"MMDII_GPU_STFT": "auto", "MMDII_TRANSFORM_WORKERS": "1"}),
            patch.object(cross_validation, "_device", return_value=torch.device("cuda")),
            patch(
                "mmdii.data.time_frequency.stft_representation_torch",
                side_effect=RuntimeError("cuFFT error: CUFFT_INTERNAL_ERROR"),
            ) as gpu_stft,
        ):
            train_tf, valid_tf = _time_frequency_datasets(dataset, dataset, config)
            representations = train_tf[0]["representations"]
            valid_representations = valid_tf[0]["representations"]

        self.assertEqual(gpu_stft.call_count, 1)
        self.assertEqual(representations.dtype, np.float32)
        self.assertEqual(valid_representations.dtype, np.float32)
        self.assertTrue(np.isfinite(representations).all())

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


def _synthetic_index(root: Path, weld_count: int = 10):
    """A tiny five-fold release used to pin artifact bytes."""

    (root / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    time = np.arange(128, dtype=np.float64) / 32.0
    records = []
    for position in range(weld_count):
        sample_id = f"sample-{position}"
        signal = np.vstack(
            [np.sin(time * (position + channel + 1)) for channel in range(3)]
        )
        signal_path = root / f"{sample_id}.npz"
        np.savez(signal_path, time=time, af=signal[0], sf=signal[1], axialf=signal[2])
        records.append(
            WeldRecord(
                sample_id=sample_id,
                weld_id=sample_id,
                image_group=sample_id,
                fold=position % 5,
                target=(
                    float(position % 2),
                    float((position // 2) % 2),
                    float((position // 4) % 2),
                ),
                defect_codes=(),
                is_normal=False,
                signal_path=signal_path,
                metadata=(),
            )
        )
    return DatasetIndex(root, ("flash", "blur", "tunnel"), tuple(records))


def _deep_config(root: Path, output: Path) -> ExperimentConfig:
    return replace(
        ExperimentConfig.for_test(),
        release_directory=root,
        output_directory=output,
        mode="full_signal",
        aggregator="mean",
        target_fs=32.0,
        window_seconds=8.0,
        stride_seconds=8.0,
        full_signal_samples=64,
        epochs=1,
    )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch train extra is not installed",
)
class DefaultBehaviourRegressionTests(unittest.TestCase):
    """The tree and second-stage extensions must not disturb ordinary runs."""

    def test_defaults_are_inert_for_the_deep_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index = _synthetic_index(root)
            first = root / "run-a"
            second = root / "run-b"

            run_cross_validation(index, _deep_config(root, first))
            run_cross_validation(
                index,
                replace(
                    _deep_config(root, second),
                    tree=TreeConfig(),
                    second_stage=SecondStageConfig(),
                ),
            )

            for name in ("oof_predictions.csv", "fold_metrics.json", "training_summary.json"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    f"{name} changed when the new tables are present at their defaults",
                )

    def test_deep_fold_reports_gain_no_second_stage_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index = _synthetic_index(root)
            output = root / "run"
            run_cross_validation(index, _deep_config(root, output))

            reports = json.loads((output / "fold_metrics.json").read_text(encoding="utf-8"))
            for report in reports:
                self.assertNotIn("second_stage", report)
                self.assertNotIn("alpha", report)
                for key in ("macro_pr_auc", "macro_f1", "macro_recall", "per_code", "threshold"):
                    self.assertIn(key, report)

    def test_run_config_gains_only_the_two_new_tables(self) -> None:
        config = _deep_config(Path("."), Path("out"))
        payload = _json_config(config)

        self.assertEqual(
            payload["tree"],
            {
                "learner": "random_forest",
                "n_estimators": 300,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "max_depth": None,
                "criterion": "gini",
                "class_weight": "balanced",
                "tune": False,
            },
        )
        self.assertEqual(
            payload["second_stage"],
            {
                "enabled": False,
                "feature_source": "stats",
                "representation_mode": "window_mil",
                "inner_folds": 4,
                "encoder_epochs": 0,
                "embedding_projection": "none",
                "embedding_components": 8,
            },
        )
        self.assertEqual(payload["mode"], "full_signal")
        self.assertEqual(payload["aggregator"], "mean")
        self.assertNotIn("fusion", payload)

    def test_committed_baseline_config_still_loads_with_defaults(self) -> None:
        config = load_experiment_config(
            CORE_ROOT / "configs" / "moderntcn_mil_weld_independent_v0_2_1.toml"
        )

        self.assertEqual(config.tree, TreeConfig())
        self.assertEqual(config.second_stage, SecondStageConfig())

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch train extra is not installed",
    )
    def test_second_stage_inner_encoder_weights_exclude_inner_validation_labels(self) -> None:
        records = tuple(
            SimpleNamespace(
                sample_id=f"f{fold}-w{position}",
                fold=fold,
                target=(
                    float(fold in {1, 3}),
                    float(position % 2 == 0),
                    float((fold == 2 and position == 0) or (fold == 4 and position == 1)),
                ),
            )
            for fold in range(5)
            for position in range(3)
        )
        train_records = tuple(record for record in records if record.fold != 0)
        valid_records = tuple(record for record in records if record.fold == 0)
        config = replace(
            ExperimentConfig.for_test(),
            mode="second_stage",
            aggregator="mean",
            second_stage=SecondStageConfig(
                enabled=True,
                feature_source="embedding",
                representation_mode="full_signal",
                inner_folds=4,
                encoder_epochs=1,
            ),
        )
        observed: list[tuple[tuple[str, ...], np.ndarray]] = []

        def fake_run_deep_records(
            _index,
            subset_train,
            subset_valid,
            _config,
            *,
            class_weights,
            **_kwargs,
        ):
            observed.append(
                (
                    tuple(record.sample_id for record in subset_train),
                    np.asarray(class_weights, dtype=np.float64),
                )
            )
            return np.zeros((len(subset_valid), 3)), {
                "weld_representations": np.zeros((len(subset_valid), 1, 4)),
                "sample_ids": tuple(record.sample_id for record in subset_valid),
            }

        with (
            patch.object(cross_validation, "_run_deep_records", fake_run_deep_records),
            patch.object(
                cross_validation,
                "fit_predict_forest_ovr",
                return_value=np.zeros((len(valid_records), 3)),
            ),
        ):
            outer_weights = cross_validation.compute_positive_class_weights(
                np.asarray([record.target for record in train_records])
            )
            cross_validation._run_second_stage(
                object(),
                train_records,
                valid_records,
                config,
                fold=0,
                class_weights=outer_weights,
                torch=None,
                nn=None,
                DataLoader=None,
            )

        self.assertEqual(len(observed), 5)
        for training_ids, weights in observed[:-1]:
            training_subset = tuple(
                record for record in train_records if record.sample_id in training_ids
            )
            expected = cross_validation.compute_positive_class_weights(
                np.asarray([record.target for record in training_subset])
            )
            np.testing.assert_allclose(weights, expected)
        np.testing.assert_allclose(observed[-1][1], outer_weights)


if __name__ == "__main__":
    unittest.main()
