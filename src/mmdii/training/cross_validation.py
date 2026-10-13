"""Reproducible weld-level cross-validation runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from mmdii.data.training_dataset import (
    DatasetIndex,
    FoldNormalizer,
    FullSignalSpec,
    WeldWindowDataset,
    WindowSpec,
    collate_weld_batch,
)
from mmdii.evaluation.multilabel import (
    compute_positive_class_weights,
    evaluate_multilabel,
)
from mmdii.models.statistical import (
    LEARNERS,
    STATISTIC_NAMES,
    extract_statistical_features,
    fit_predict_forest_ovr,
    fit_predict_logistic_ovr,
)
from mmdii.training.strategy import (
    AugmentationConfig,
    RawWindowAugmenter,
    TrainingStrategy,
    TrainingStrategyConfig,
)
from mmdii.training.second_stage import (
    FEATURE_SOURCES,
    run_second_stage_fold,
)
from mmdii.reporting import (
    ProgressReporter,
    format_eta,
    get_logger,
    log_banner,
    log_dataset,
    log_epoch,
    log_fold_summary,
    log_results_table,
)


_LOGGER = get_logger("mmdii.training")


@dataclass(frozen=True)
class ModelConfig:
    hidden_channels: int = 32
    embedding_dim: int = 64
    kernel_size: int = 31
    block_count: int = 2
    dropout: float = 0.1
    top_k: int = 3
    attention_dim: int = 64
    encoder_chunk_size: int = 32


@dataclass(frozen=True)
class RepresentationConfig:
    name: str = "raw"
    encoder: str = "modern_tcn"
    fusion: str = "none"
    output_time_bins: int = 256
    normalization: str = "per_channel_zscore"
    stft_n_fft: int | None = None
    stft_hop_length: int | None = None
    cwt_wavelet: str = "morl"
    cwt_frequency_bins: int = 48
    cwt_min_frequency_hz: float = 30.0
    cwt_max_frequency_hz: float | None = None
    dwt_wavelet: str = "db4"
    dwt_level: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    scheduler: str = "none"
    warmup_epochs: int = 0
    gradient_clip_norm: float = 0.0
    amp: bool = False


@dataclass(frozen=True)
class AugmentationExperimentConfig:
    enabled: bool = False
    amplitude_scale: float = 0.05
    noise_std: float = 0.01
    time_mask_ratio: float = 0.10
    max_time_masks: int = 1


@dataclass(frozen=True)
class TreeConfig:
    """Settings for the non-neural tree ensemble over tabular features.

    The learner stays a config value rather than a separate ``mode`` so that
    the ensemble family can be swept as an ablation without changing the
    training dispatch, the output schema, or the tabular feature pipeline.
    """

    learner: str = "random_forest"
    n_estimators: int = 300
    min_samples_leaf: int = 1


@dataclass(frozen=True)
class SecondStageConfig:
    """Settings for fitting a tree on weld-level features instead of the deep head.

    Disabled by default; when ``enabled`` is false and the mode is anything
    other than ``second_stage`` the training pipeline behaves exactly as before.
    """

    enabled: bool = False
    feature_source: str = "stats"
    representation_mode: str = "window_mil"
    inner_folds: int = 4
    encoder_epochs: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path | None
    release_directory: Path
    output_directory: Path
    target_codes: tuple[str, ...]
    mode: str
    aggregator: str
    seed: int
    fold_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    target_fs: float
    window_seconds: float
    stride_seconds: float
    full_signal_samples: int
    fold_scheme: str = "weld_independent"
    optimizer: str = "adamw"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: float = 0.0
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationExperimentConfig = field(default_factory=AugmentationExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    second_stage: SecondStageConfig = field(default_factory=SecondStageConfig)
    run_folds: tuple[int, ...] = ()
    pipeline_backend: str = "cpu"
    preprocess_cache_directory: Path | None = None
    gpu_cache_gib: float = 4.0

    @classmethod
    def for_test(cls) -> "ExperimentConfig":
        return cls(
            config_path=None,
            release_directory=Path("release"),
            output_directory=Path("output"),
            target_codes=("flash", "blur", "tunnel"),
            mode="window_mil",
            aggregator="gated_attention",
            seed=7,
            fold_count=5,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
            device="cpu",
            target_fs=5400.0,
            window_seconds=2.0,
            stride_seconds=1.0,
            full_signal_samples=256,
            fold_scheme="weld_independent",
            optimizer="adamw",
            early_stopping_patience=0,
            early_stopping_min_delta=0.0,
            gradient_clip_norm=0.0,
            training=TrainingConfig(),
            augmentation=AugmentationExperimentConfig(),
            pipeline_backend="cpu",
            preprocess_cache_directory=None,
            gpu_cache_gib=4.0,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        experiment = payload["experiment"]
        model_payload = payload.get("model", {})
        representation_payload = payload.get("representation", {})
        training_payload = payload.get("training", {})
        augmentation_payload = payload.get("augmentation", {})
        tree_payload = payload.get("tree", {})
        second_stage_payload = payload.get("second_stage", {})
        target_codes = tuple(experiment["target_codes"])
        config = ExperimentConfig(
            config_path=config_path,
            release_directory=(config_path.parent / experiment["release_directory"]).resolve(),
            output_directory=(config_path.parent / experiment["output_directory"]).resolve(),
            target_codes=target_codes,
            mode=experiment["mode"],
            aggregator=experiment["aggregator"],
            seed=int(experiment["seed"]),
            fold_count=int(experiment["fold_count"]),
            epochs=int(experiment["epochs"]),
            batch_size=int(experiment["batch_size"]),
            learning_rate=float(experiment["learning_rate"]),
            weight_decay=float(experiment["weight_decay"]),
            device=experiment["device"],
            target_fs=float(experiment["target_fs"]),
            window_seconds=float(experiment["window_seconds"]),
            stride_seconds=float(experiment["stride_seconds"]),
            full_signal_samples=int(experiment["full_signal_samples"]),
            fold_scheme=str(experiment.get("fold_scheme", "image_group")),
            optimizer=str(experiment.get("optimizer", "adamw")),
            early_stopping_patience=int(experiment.get("early_stopping_patience", 0)),
            early_stopping_min_delta=float(experiment.get("early_stopping_min_delta", 0.0)),
            gradient_clip_norm=float(experiment.get("gradient_clip_norm", 0.0)),
            training=TrainingConfig(
                scheduler=str(training_payload.get("scheduler", "none")),
                warmup_epochs=int(training_payload.get("warmup_epochs", 0)),
                gradient_clip_norm=float(training_payload.get("gradient_clip_norm", experiment.get("gradient_clip_norm", 0.0))),
                amp=bool(training_payload.get("amp", False)),
            ),
            augmentation=AugmentationExperimentConfig(
                enabled=bool(augmentation_payload.get("enabled", False)),
                amplitude_scale=float(augmentation_payload.get("amplitude_scale", 0.05)),
                noise_std=float(augmentation_payload.get("noise_std", 0.01)),
                time_mask_ratio=float(augmentation_payload.get("time_mask_ratio", 0.10)),
                max_time_masks=int(augmentation_payload.get("max_time_masks", 1)),
            ),
            pipeline_backend=str(experiment.get("pipeline_backend", "cpu")),
            preprocess_cache_directory=(
                None if experiment.get("preprocess_cache_directory") in (None, "")
                else (config_path.parent / str(experiment["preprocess_cache_directory"])).resolve()
            ),
            gpu_cache_gib=float(experiment.get("gpu_cache_gib", 4.0)),
            model=ModelConfig(
                **{
                    field_name: model_payload.get(field_name, getattr(ModelConfig(), field_name))
                    for field_name in ModelConfig.__dataclass_fields__
                }
            ),
            representation=RepresentationConfig(
                **{
                    field_name: representation_payload.get(
                        field_name, getattr(RepresentationConfig(), field_name)
                    )
                    for field_name in RepresentationConfig.__dataclass_fields__
                }
            ),
            tree=TreeConfig(
                learner=str(tree_payload.get("learner", TreeConfig().learner)),
                n_estimators=int(tree_payload.get("n_estimators", TreeConfig().n_estimators)),
                min_samples_leaf=int(
                    tree_payload.get("min_samples_leaf", TreeConfig().min_samples_leaf)
                ),
            ),
            second_stage=SecondStageConfig(
                enabled=bool(second_stage_payload.get("enabled", False)),
                feature_source=str(
                    second_stage_payload.get(
                        "feature_source", SecondStageConfig().feature_source
                    )
                ),
                representation_mode=str(
                    second_stage_payload.get(
                        "representation_mode", SecondStageConfig().representation_mode
                    )
                ),
                inner_folds=int(
                    second_stage_payload.get("inner_folds", SecondStageConfig().inner_folds)
                ),
                encoder_epochs=int(
                    second_stage_payload.get(
                        "encoder_epochs", SecondStageConfig().encoder_epochs
                    )
                ),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid experiment configuration: {error}") from error
    _validate_config(config)
    return config


def run_cross_validation(
    index: DatasetIndex | None, config: ExperimentConfig
) -> dict[str, object]:
    """Run one model per held-out weld fold and publish weld-level OOF output."""

    if config.mode not in {"statistical", "random_forest"} and not _mode_is_tabular_only(config):
        torch, nn, DataLoader = _require_torch()
    else:
        torch = nn = DataLoader = None
    if index is None:
        raise ValueError("A DatasetIndex is required for training.")
    _validate_index(index, config)
    _seed_everything(config.seed, torch)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    oof_rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []

    folds = config.run_folds or tuple(range(config.fold_count))
    log_banner(
        _LOGGER,
        experiment=config.output_directory.name,
        mode=config.mode,
        aggregator=config.aggregator,
        seed=config.seed,
        folds=folds,
        epochs=config.epochs,
        device=config.device,
        representation=config.representation.name,
    )
    log_dataset(
        _LOGGER,
        sample_count=len(index.records),
        fold_count=config.fold_count,
        target_codes=config.target_codes,
        representation=config.representation.name,
    )
    for fold in folds:
        train_records = tuple(record for record in index.records if record.fold != fold)
        valid_records = tuple(record for record in index.records if record.fold == fold)
        if not train_records or not valid_records:
            raise ValueError(f"Fold {fold} has no train or validation records.")
        _LOGGER.info(
            "fold %s/%s start train=%d valid=%d",
            fold,
            len(folds),
            len(train_records),
            len(valid_records),
        )
        train_targets = np.asarray([record.target for record in train_records], dtype=np.float64)
        class_weights = compute_positive_class_weights(train_targets)
        training_info: dict[str, object] = {}
        if config.mode in {"statistical", "random_forest"}:
            probabilities = _run_statistical_fold(
                index, train_records, valid_records, config, fold=fold
            )
        elif config.mode == "second_stage":
            if config.second_stage.feature_source == "stats":
                probabilities = _run_statistical_fold(
                    index, train_records, valid_records, config, fold=fold
                )
                training_info["second_stage"] = {
                    "feature_source": "stats",
                    "feature_count": len(STATISTIC_NAMES) * 3,
                    "inner_folds": 0,
                }
            else:
                probabilities, training_info = _run_second_stage(
                    index,
                    train_records,
                    valid_records,
                    config,
                    fold=fold,
                    class_weights=class_weights,
                    torch=torch,
                    nn=nn,
                    DataLoader=DataLoader,
                )
        else:
            spec_kind = "full_signal" if config.mode == "full_signal" else "window"
            aggregator = "mean" if spec_kind == "full_signal" else config.aggregator
            probabilities, training_info = _run_deep_records(
                index,
                train_records,
                valid_records,
                config,
                fold=fold,
                class_weights=class_weights,
                torch=torch,
                nn=nn,
                DataLoader=DataLoader,
                spec_kind=spec_kind,
                aggregator=aggregator,
            )
        truth = np.asarray([record.target for record in valid_records], dtype=np.float64)
        metrics = evaluate_multilabel(
            truth,
            probabilities,
            target_codes=config.target_codes,
        )
        metrics.update(training_info)
        metrics["fold"] = fold
        fold_reports.append(metrics)
        log_fold_summary(_LOGGER, fold=fold, metrics=metrics)
        for record, row_probabilities in zip(valid_records, probabilities, strict=True):
            oof_rows.append(
                {
                    "sample_id": record.sample_id,
                    "weld_id": record.weld_id,
                    "image_group": record.image_group,
                    "fold": fold,
                    "target_codes_json": json.dumps(record.defect_codes, separators=(",", ":")),
                    **{
                        f"prob_{code}": float(row_probabilities[position])
                        for position, code in enumerate(config.target_codes)
                    },
                }
            )

    oof_rows.sort(key=lambda row: str(row["sample_id"]))
    _write_oof(config.output_directory / "oof_predictions.csv", oof_rows, config.target_codes)
    summary = {
        "mode": config.mode,
        "aggregator": config.aggregator,
        "fold_count": config.fold_count,
        "completed_fold_count": len(folds),
        "run_folds": list(folds),
        "fold_scheme": config.fold_scheme,
        "sample_count": len(oof_rows),
        "fold_metrics": fold_reports,
    }
    (config.output_directory / "run_config.json").write_text(
        json.dumps(_json_config(config), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_directory / "fold_metrics.json").write_text(
        json.dumps(fold_reports, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_directory / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_results_table(
        _LOGGER,
        rows=fold_reports,
        columns=("fold", "macro_f1", "macro_recall", "macro_pr_auc", "epochs_ran"),
    )
    return summary


def _run_statistical_fold(index: DatasetIndex, train_records: tuple[Any, ...], valid_records: tuple[Any, ...], config: ExperimentConfig, *, fold: int | None = None) -> np.ndarray:
    total = len(train_records) + len(valid_records)
    progress = ProgressReporter(
        _LOGGER,
        total=total,
        label=f"fold {fold} stat features",
        every_steps=max(1, total // 10),
        every_seconds=15.0,
    )
    train_features_list: list[np.ndarray] = []
    for record in train_records:
        train_features_list.append(
            extract_statistical_features(index.load_signal(record)[0])
        )
        progress.update(len(train_features_list))
    valid_features_list: list[np.ndarray] = []
    for record in valid_records:
        valid_features_list.append(
            extract_statistical_features(index.load_signal(record)[0])
        )
        progress.update(len(train_features_list) + len(valid_features_list))
    progress.close()
    train_features = np.stack(train_features_list)
    valid_features = np.stack(valid_features_list)
    if config.mode == "statistical":
        return fit_predict_logistic_ovr(
            train_features,
            np.asarray([record.target for record in train_records], dtype=np.float64),
            valid_features,
            target_codes=config.target_codes,
            random_state=config.seed,
        )
    return fit_predict_forest_ovr(
        train_features,
        np.asarray([record.target for record in train_records], dtype=np.float64),
        valid_features,
        target_codes=config.target_codes,
        random_state=config.seed,
        learner=config.tree.learner,
        n_estimators=config.tree.n_estimators,
        min_samples_leaf=config.tree.min_samples_leaf,
    )


def _build_deep_model(config: ExperimentConfig, aggregator: str, torch: Any) -> Any:
    from mmdii.models.mil import WeldMIL
    from mmdii.models.modern_tcn import ModernTCNSmall

    raw_encoder = ModernTCNSmall(
        input_channels=3,
        hidden_channels=config.model.hidden_channels,
        embedding_dim=config.model.embedding_dim,
        kernel_size=config.model.kernel_size,
        block_count=config.model.block_count,
        dropout=config.model.dropout,
    )
    if config.representation.name == "raw":
        encoder = raw_encoder
    else:
        from mmdii.models.encoders2d import RawTimeFrequencyFusion, build_2d_encoder

        time_frequency_encoder = build_2d_encoder(
            config.representation.encoder,
            input_channels=3,
            embedding_dim=config.model.embedding_dim,
        )
        encoder = (
            time_frequency_encoder
            if config.representation.fusion == "none"
            else RawTimeFrequencyFusion(
                raw_encoder=raw_encoder,
                time_frequency_encoder=time_frequency_encoder,
                raw_embedding_dim=config.model.embedding_dim,
                time_frequency_embedding_dim=config.model.embedding_dim,
                embedding_dim=config.model.embedding_dim,
                dropout=config.model.dropout,
            )
        )
    head = WeldMIL(
        embedding_dim=config.model.embedding_dim,
        num_targets=len(config.target_codes),
        mode=aggregator,
        top_k=config.model.top_k,
        attention_dim=config.model.attention_dim,
    )

    class WeldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head

        def window_embeddings(
            self,
            windows: Any,
            window_mask: Any,
            sample_mask: Any,
            representations: Any | None = None,
        ) -> Any:
            batch_size, window_count, channels, sample_count = windows.shape
            flat_windows = windows.reshape(batch_size * window_count, channels, sample_count)
            flat_sample_mask = sample_mask.reshape(batch_size * window_count, sample_count)
            valid_windows = window_mask.reshape(batch_size * window_count)
            selected_windows = flat_windows[valid_windows]
            selected_sample_mask = flat_sample_mask[valid_windows]
            selected_representations = None
            if representations is not None:
                selected_representations = representations.reshape(
                    batch_size * window_count, *representations.shape[2:]
                )[valid_windows]
            chunks = []
            for start in range(0, int(valid_windows.sum()), config.model.encoder_chunk_size):
                stop = start + config.model.encoder_chunk_size
                if config.representation.name == "raw":
                    embedding = self.encoder(
                        selected_windows[start:stop],
                        sample_mask=selected_sample_mask[start:stop],
                    )
                elif config.representation.fusion == "none":
                    if selected_representations is None:
                        raise ValueError("Time-frequency representations are required.")
                    embedding = self.encoder(selected_representations[start:stop])
                else:
                    if selected_representations is None:
                        raise ValueError("Time-frequency representations are required for fusion.")
                    embedding = self.encoder(
                        selected_windows[start:stop],
                        selected_representations[start:stop],
                        sample_mask=selected_sample_mask[start:stop],
                    )
                chunks.append(embedding)
            valid_embeddings = torch.cat(chunks, dim=0)
            embeddings = windows.new_zeros(
                (batch_size * window_count, valid_embeddings.shape[1])
            )
            embeddings[valid_windows] = valid_embeddings
            return embeddings.reshape(batch_size, window_count, -1)

        def forward(
            self,
            windows: Any,
            window_mask: Any,
            sample_mask: Any,
            representations: Any | None = None,
        ) -> Any:
            embeddings = self.window_embeddings(
                windows, window_mask, sample_mask, representations
            )
            return self.head(embeddings, window_mask)

        def forward_with_representation(
            self,
            windows: Any,
            window_mask: Any,
            sample_mask: Any,
            representations: Any | None = None,
        ) -> tuple[Any, Any]:
            """One forward pass returning both the head output and the pooled weld vector."""

            embeddings = self.window_embeddings(
                windows, window_mask, sample_mask, representations
            )
            pooled = self.head.pool(embeddings, window_mask)
            return self.head(embeddings, window_mask), pooled

    return WeldModel()


class _TimeFrequencyDataset:
    """Lazily transform an existing weld-window dataset into 2-D inputs."""

    def __init__(
        self,
        source: Any,
        config: ExperimentConfig,
        normalizer: Any,
        precomputed: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.normalizer = normalizer
        # The training fold is transformed once while fitting the fold-only
        # normalizer.  Reuse those arrays instead of recomputing expensive
        # CWT/SWT transforms when DataLoader first accesses each item.
        self._precomputed = precomputed or {}
        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, position: int) -> dict[str, object]:
        if position in self._cache:
            return self._cache[position]
        item = dict(self.source[position])
        if position in self._precomputed:
            transformed, time_masks = self._precomputed[position]
        else:
            from mmdii.data.time_frequency import transform_representation

            windows = np.asarray(item["windows"], dtype=np.float64)
            masks = np.asarray(item["sample_mask"], dtype=bool)
            transformed = []
            time_masks = []
            for window, mask in zip(windows, masks, strict=True):
                representation, time_mask = transform_representation(
                    window,
                    self.config.representation.name,
                    sample_mask=mask,
                    target_fs=self.config.target_fs,
                    output_time_bins=self.config.representation.output_time_bins,
                    **_representation_transform_parameters(self.config),
                )
                transformed.append(representation)
                time_masks.append(time_mask)
            transformed = np.stack(transformed)
            time_masks = np.stack(time_masks)
        normalized = self.normalizer.transform(transformed)
        normalized *= np.asarray(time_masks, dtype=np.float32)[:, np.newaxis, np.newaxis, :]
        item["representations"] = normalized
        self._cache[position] = item
        return item


def _time_frequency_datasets(
    train_dataset: Any, valid_dataset: Any, config: ExperimentConfig
) -> tuple[Any, Any]:
    from mmdii.data.time_frequency import RepresentationNormalizer, transform_representation

    gpu_stft_mode = os.environ.get("MMDII_GPU_STFT", "auto").lower()
    if gpu_stft_mode not in {"auto", "0", "1"}:
        raise ValueError("MMDII_GPU_STFT must be auto, 0, or 1.")
    gpu_stft = False
    gpu_device = None
    if config.pipeline_backend == "cuda" and gpu_stft_mode != "0":
        torch, _, _ = _require_torch()
        gpu_device = _device(config.device, torch)
        gpu_stft = gpu_device.type == "cuda" and config.representation.name.startswith("stft_")
        if gpu_stft_mode == "1" and not gpu_stft:
            raise RuntimeError(
                "MMDII_GPU_STFT=1 requires a CUDA device and an STFT representation."
            )
    gpu_stft_state = [gpu_stft]
    if config.representation.name.startswith("stft_") and gpu_stft_mode == "0":
        _LOGGER.info("GPU STFT disabled by MMDII_GPU_STFT=0; using CPU preprocessing.")

    def cache_path(dataset: Any, split: str) -> Path | None:
        if config.preprocess_cache_directory is None:
            return None
        manifest = dataset.index.release_directory / "dataset_manifest.json"
        release_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        payload = {"cache_version": 1, "release_hash": release_hash,
                   "records": [record.sample_id for record in dataset.records],
                   "spec": repr(dataset.spec), "representation": asdict(config.representation), "split": split}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]
        return config.preprocess_cache_directory / key

    def transform_dataset(dataset: Any, split: str) -> tuple[list[np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
        cached_path = cache_path(dataset, split)
        if cached_path is not None and cached_path.exists() and (
            not cached_path.is_dir() or not (cached_path / "COMPLETE").exists()
        ):
            shutil.rmtree(cached_path, ignore_errors=True)
        if cached_path is not None and cached_path.is_dir() and (cached_path / "COMPLETE").exists():
            try:
                arrays = [np.load(cached_path / f"array-{i}.npy", mmap_mode="r") for i in range(len(dataset))]
                masks = [np.load(cached_path / f"mask-{i}.npy", mmap_mode="r") for i in range(len(dataset))]
                _LOGGER.info("tf transform %s cache hit records=%d", split, len(dataset))
                return arrays, {i: (arrays[i], masks[i]) for i in range(len(arrays))}
            except (OSError, ValueError):
                shutil.rmtree(cached_path, ignore_errors=True)
        def transform_item(position: int) -> tuple[int, np.ndarray, np.ndarray]:
            item = dataset[position]
            windows = np.asarray(item["windows"], dtype=np.float64)
            masks = np.asarray(item["sample_mask"], dtype=bool)
            if gpu_stft_state[0]:
                from mmdii.data.time_frequency import stft_representation_torch
                try:
                    transformed, time_masks = stft_representation_torch(
                        windows, sample_mask=masks, target_fs=config.target_fs,
                        output_time_bins=config.representation.output_time_bins,
                        device=gpu_device, **_representation_transform_parameters(config))
                    return position, transformed.detach().cpu().numpy(), time_masks.detach().cpu().numpy()
                except RuntimeError as error:
                    message = str(error).lower()
                    if "cufft" not in message and "cuda" not in message:
                        raise
                    gpu_stft_state[0] = False
                    _LOGGER.warning(
                        "GPU STFT failed for %s (%s); falling back to CPU STFT "
                        "for this run while keeping model training on CUDA.",
                        split,
                        error,
                    )
            transformed: list[np.ndarray] = []
            time_masks: list[np.ndarray] = []
            for window, mask in zip(windows, masks, strict=True):
                representation, time_mask = transform_representation(
                    window,
                    config.representation.name,
                    sample_mask=mask,
                    target_fs=config.target_fs,
                    output_time_bins=config.representation.output_time_bins,
                    **_representation_transform_parameters(config),
                )
                transformed.append(representation)
                time_masks.append(time_mask)
            return position, np.stack(transformed), np.stack(time_masks)

        arrays: list[np.ndarray] = []
        transformed_by_position: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        positions = range(len(dataset))
        progress = ProgressReporter(
            _LOGGER,
            total=len(dataset),
            label=f"tf transform {split}",
            every_steps=max(1, len(dataset) // 20),
            every_seconds=15.0,
        )
        worker_count = max(1, int(os.environ.get("MMDII_TRANSFORM_WORKERS", "1")))
        if worker_count > 1 and len(dataset) > 1 and not gpu_stft:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                transformed_items = executor.map(transform_item, positions)
                results = []
                for completed, item in enumerate(transformed_items):
                    results.append(item)
                    progress.update(completed + 1)
        else:
            results = []
            for completed, position in enumerate(positions):
                results.append(transform_item(position))
                progress.update(completed + 1)
        progress.close()
        for position, transformed_array, mask_array in sorted(results):
            arrays.append(transformed_array)
            transformed_by_position[position] = (transformed_array, mask_array)
        if cached_path is not None:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached_path.with_name(cached_path.name + f".{os.getpid()}.tmp")
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=True)
            for position, transformed_array in enumerate(arrays):
                np.save(temporary / f"array-{position}.npy", transformed_array, allow_pickle=False)
                np.save(temporary / f"mask-{position}.npy", transformed_by_position[position][1], allow_pickle=False)
            (temporary / "COMPLETE").write_text("v1\n", encoding="ascii")
            _LOGGER.info("tf transform %s cache write arrays=%d", split, len(arrays))
            if not cached_path.exists():
                os.replace(temporary, cached_path)
            else:
                shutil.rmtree(temporary, ignore_errors=True)
        return arrays, transformed_by_position

    arrays, train_precomputed = transform_dataset(train_dataset, "train")
    normalizer = RepresentationNormalizer.fit(arrays)
    _, valid_precomputed = transform_dataset(valid_dataset, "valid")
    return (
        _TimeFrequencyDataset(train_dataset, config, normalizer, train_precomputed),
        _TimeFrequencyDataset(valid_dataset, config, normalizer, valid_precomputed),
    )


def _mode_is_tabular_only(config: ExperimentConfig) -> bool:
    """True when the run can be completed without importing torch."""

    return config.mode == "second_stage" and config.second_stage.feature_source == "stats"


def _deep_spec(config: ExperimentConfig, spec_kind: str) -> WindowSpec | FullSignalSpec:
    if spec_kind == "full_signal":
        return FullSignalSpec(
            target_fs=config.target_fs, output_samples=config.full_signal_samples
        )
    if spec_kind == "window":
        return WindowSpec(
            target_fs=config.target_fs,
            window_seconds=config.window_seconds,
            stride_seconds=config.stride_seconds,
        )
    raise ValueError("spec_kind must be window or full_signal.")


def _run_deep_records(
    index: DatasetIndex,
    train_records: tuple[Any, ...],
    valid_records: tuple[Any, ...],
    config: ExperimentConfig,
    *,
    fold: int | None,
    class_weights: np.ndarray,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    spec_kind: str,
    aggregator: str,
    harvest_embeddings: bool = False,
    epochs: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the deep model for one record split and train it.

    Shared by the ordinary deep modes and by the encoder stages of the
    second-stage experiment so both use identical windowing, normalisation,
    representation and optimisation code.
    """

    if epochs is not None and epochs != config.epochs:
        config = replace(config, epochs=epochs)
    normalizer = FoldNormalizer.fit(index, train_records)
    spec = _deep_spec(config, spec_kind)
    train_dataset = WeldWindowDataset(
        index, {record.fold for record in train_records}, spec, normalizer
    )
    valid_dataset = WeldWindowDataset(
        index, {record.fold for record in valid_records}, spec, normalizer
    )
    if config.representation.name != "raw":
        train_dataset, valid_dataset = _time_frequency_datasets(
            train_dataset, valid_dataset, config
        )
    model = _build_deep_model(config, aggregator, torch)
    return _run_deep_fold(
        model,
        train_dataset,
        valid_dataset,
        class_weights,
        config,
        torch,
        nn,
        DataLoader,
        fold=fold,
        harvest_embeddings=harvest_embeddings,
    )


def _run_second_stage(
    index: DatasetIndex,
    train_records: tuple[Any, ...],
    valid_records: tuple[Any, ...],
    config: ExperimentConfig,
    *,
    fold: int | None,
    class_weights: np.ndarray,
    torch: Any,
    nn: Any,
    DataLoader: Any,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a tree on weld-level features for one outer fold."""

    from mmdii.models.mil import WeldMIL

    spec_kind = (
        "full_signal"
        if config.second_stage.representation_mode == "full_signal"
        else "window"
    )
    aggregator = "mean" if spec_kind == "full_signal" else config.aggregator
    encoder_epochs = config.second_stage.encoder_epochs or config.epochs
    representation_shared = WeldMIL(
        embedding_dim=1, num_targets=1, mode=aggregator
    ).representation_shared

    def run_deep(train_subset, valid_subset, *, harvest_embeddings=False):
        subset_class_weights = class_weights
        if len(train_subset) != len(train_records):
            # Inner OOS representations must not use labels from their held-out
            # partition, including labels that affect the weighted BCE loss.
            subset_targets = np.asarray(
                [record.target for record in train_subset], dtype=np.float64
            )
            subset_class_weights = compute_positive_class_weights(subset_targets)
        return _run_deep_records(
            index,
            train_subset,
            valid_subset,
            config,
            fold=fold,
            class_weights=subset_class_weights,
            torch=torch,
            nn=nn,
            DataLoader=DataLoader,
            spec_kind=spec_kind,
            aggregator=aggregator,
            harvest_embeddings=harvest_embeddings,
            epochs=encoder_epochs,
        )

    def fit_tree(train_features: np.ndarray, train_targets: np.ndarray, valid_features: np.ndarray) -> np.ndarray:
        return fit_predict_forest_ovr(
            train_features,
            train_targets,
            valid_features,
            target_codes=config.target_codes,
            random_state=config.seed,
            learner=config.tree.learner,
            n_estimators=config.tree.n_estimators,
            min_samples_leaf=config.tree.min_samples_leaf,
        )

    return run_second_stage_fold(
        index=index,
        train_records=train_records,
        valid_records=valid_records,
        config=config,
        fold=fold,
        run_deep=run_deep,
        fit_tree=fit_tree,
        representation_shared=representation_shared,
    )


def _run_deep_fold(model: Any, train_dataset: Any, valid_dataset: Any, class_weights: np.ndarray, config: ExperimentConfig, torch: Any, nn: Any, DataLoader: Any, *, fold: int | None = None, harvest_embeddings: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    device = _device(config.device, torch)
    gpu_dataset = None
    if (config.pipeline_backend == "cuda" and device.type == "cuda"
            and not config.augmentation.enabled):
        estimated_bytes = _estimate_gpu_dataset_bytes(train_dataset) + _estimate_gpu_dataset_bytes(valid_dataset)
        if estimated_bytes <= int(config.gpu_cache_gib * (1024 ** 3)):
            gpu_dataset = (_cache_dataset_on_gpu(train_dataset, device, torch),
                           _cache_dataset_on_gpu(valid_dataset, device, torch))
    model.to(device)
    optimizer_class = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }[config.optimizer]
    optimizer = optimizer_class(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    collate = _torch_collate if config.representation.name == "raw" else _torch_time_frequency_collate
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        range(len(train_dataset)) if gpu_dataset is not None else train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=(list if gpu_dataset is not None else collate),
        pin_memory=pin_memory and gpu_dataset is None,
    )
    valid_loader = DataLoader(
        range(len(valid_dataset)) if gpu_dataset is not None else valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=(list if gpu_dataset is not None else collate),
        pin_memory=pin_memory and gpu_dataset is None,
    )
    strategy = TrainingStrategy(
        optimizer,
        config=TrainingStrategyConfig(
            scheduler=config.training.scheduler,
            warmup_epochs=config.training.warmup_epochs,
            gradient_clip_norm=(config.training.gradient_clip_norm or config.gradient_clip_norm),
            amp=config.training.amp,
        ),
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        device=device,
    )
    augmenter = RawWindowAugmenter(
        AugmentationConfig(
            enabled=config.augmentation.enabled,
            amplitude_scale=config.augmentation.amplitude_scale,
            noise_std=config.augmentation.noise_std,
            time_mask_ratio=config.augmentation.time_mask_ratio,
            max_time_masks=config.augmentation.max_time_masks,
        ),
        seed=config.seed,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0
    epoch_history: list[dict[str, float]] = []
    training_start = time.monotonic()
    _LOGGER.info(
        "fold %s train start epochs=%d steps/epoch=%d",
        fold,
        config.epochs,
        len(train_loader),
    )
    for epoch in range(config.epochs):
        epoch_start = time.monotonic()
        epoch_reporter = ProgressReporter(
            _LOGGER,
            total=len(train_loader),
            label=f"fold {fold} train",
            every_steps=max(1, len(train_loader) // 10),
            every_seconds=30.0,
        )
        model.train()
        train_loss_total = None
        train_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if gpu_dataset is not None:
                batch = _gpu_batch(batch, gpu_dataset[0], torch)
            windows = batch["windows"].to(device, non_blocking=pin_memory and gpu_dataset is None)
            sample_mask = batch["sample_mask"].to(device, non_blocking=pin_memory and gpu_dataset is None)
            windows = augmenter(windows, sample_mask, epoch=epoch, batch_index=batch_index)
            with strategy.autocast():
                logits, _ = model(
                    windows,
                    batch["window_mask"].to(device, non_blocking=pin_memory and gpu_dataset is None),
                    sample_mask,
                    _to_device(batch.get("representations"), device, non_blocking=pin_memory and gpu_dataset is None),
                )
                loss = criterion(logits, batch["targets"].to(device, non_blocking=pin_memory and gpu_dataset is None))
            strategy.backward_step(loss, model)
            train_loss_total = loss.detach() if train_loss_total is None else train_loss_total + loss.detach()
            train_batches += 1
            epoch_reporter.update(batch_index + 1)
        epoch_reporter.close()
        strategy.epoch_step()
        epoch_history.append(strategy.epoch_stats())
        epochs_ran = epoch + 1
        epoch_loss = float((train_loss_total / max(train_batches, 1)).cpu())
        epoch_elapsed = time.monotonic() - epoch_start
        epoch_stats = epoch_history[-1]
        log_epoch(
            _LOGGER,
            fold=fold,
            epoch=epoch + 1,
            total_epochs=config.epochs,
            train_loss=epoch_loss,
            learning_rate=float(epoch_stats.get("learning_rate", 0.0)),
            gradient_norm_mean=float(epoch_stats.get("gradient_norm_mean", 0.0)),
            gradient_norm_max=float(epoch_stats.get("gradient_norm_max", 0.0)),
            elapsed=epoch_elapsed,
            eta=format_eta(epoch + 1, config.epochs, time.monotonic() - training_start),
        )
        if config.early_stopping_patience > 0:
            if epoch_loss < best_loss - config.early_stopping_min_delta:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    _LOGGER.info(
                        "fold %s early_stop epoch=%d best_loss=%.4f patience=%d monitor=train_loss",
                        fold,
                        epoch + 1,
                        best_loss,
                        config.early_stopping_patience,
                    )
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probabilities = []
    harvest_representations: list[Any] = []
    harvest_sample_ids: list[str] = []
    _LOGGER.info("fold %s validation start batches=%d", fold, len(valid_loader))
    with torch.no_grad():
        for batch in valid_loader:
            batch_indices = None
            if gpu_dataset is not None:
                batch_indices = [int(index) for index in batch]
                batch = _gpu_batch(batch_indices, gpu_dataset[1], torch)
            windows = batch["windows"].to(device, non_blocking=pin_memory and gpu_dataset is None)
            window_mask = batch["window_mask"].to(device, non_blocking=pin_memory and gpu_dataset is None)
            sample_mask = batch["sample_mask"].to(device, non_blocking=pin_memory and gpu_dataset is None)
            representations = _to_device(
                batch.get("representations"), device, non_blocking=pin_memory and gpu_dataset is None
            )
            if harvest_embeddings:
                (logits, _), pooled = model.forward_with_representation(
                    windows, window_mask, sample_mask, representations
                )
                harvest_representations.append(pooled.detach().cpu().numpy())
                if batch_indices is not None:
                    harvest_sample_ids.extend(
                        str(gpu_dataset[1]["sample_ids"][index]) for index in batch_indices
                    )
                else:
                    harvest_sample_ids.extend(str(value) for value in batch["sample_ids"])
            else:
                logits, _ = model(windows, window_mask, sample_mask, representations)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    training_info: dict[str, object] = {
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch or epochs_ran,
        "early_stopping_monitor": "train_loss" if config.early_stopping_patience > 0 else None,
        "training_history": epoch_history,
        "scheduler": config.training.scheduler,
        "amp_enabled": strategy.amp_enabled,
        "augmentation_enabled": config.augmentation.enabled,
    }
    if harvest_embeddings:
        training_info["weld_representations"] = np.concatenate(harvest_representations, axis=0)
        training_info["sample_ids"] = tuple(harvest_sample_ids)
    return np.concatenate(probabilities, axis=0), training_info


def _estimate_gpu_dataset_bytes(dataset: Any) -> int:
    total = 0
    for position in range(len(dataset)):
        item = dataset[position]
        for key in ("windows", "window_mask", "sample_mask", "targets", "representations"):
            if key in item:
                total += np.asarray(item[key]).nbytes
    return total


def _cache_dataset_on_gpu(dataset: Any, device: Any, torch: Any) -> dict[str, Any]:
    columns: dict[str, list[Any]] = {key: [] for key in ("windows", "window_mask", "sample_mask", "targets", "representations")}
    metadata = {key: [] for key in ("sample_ids", "weld_ids", "image_groups", "folds")}
    for position in range(len(dataset)):
        item = dataset[position]
        for key in columns:
            if key in item:
                columns[key].append(torch.as_tensor(np.asarray(item[key]), device=device))
        metadata["sample_ids"].append(str(item["sample_id"]))
        metadata["weld_ids"].append(str(item["weld_id"]))
        metadata["image_groups"].append(str(item["image_group"]))
        metadata["folds"].append(int(item["fold"]))
    return {**{key: value for key, value in metadata.items()},
            **{key: value for key, value in columns.items() if value}}


def _gpu_batch(indices: list[int], cached: dict[str, Any], torch: Any) -> dict[str, Any]:
    positions = [int(index) for index in indices]
    windows = [cached["windows"][index] for index in positions]
    max_windows = max(value.shape[0] for value in windows)
    batch: dict[str, Any] = {}
    for key in ("windows", "sample_mask", "representations"):
        if key not in cached:
            continue
        source = cached[key]
        tail = source[0].shape[1:]
        output = torch.zeros((len(windows), max_windows, *tail), dtype=source[0].dtype, device=source[0].device)
        for row, position in enumerate(positions):
            value = source[position]
            output[row, :value.shape[0]] = value
        batch[key] = output
    window_mask = torch.zeros((len(windows), max_windows), dtype=torch.bool, device=windows[0].device)
    targets = []
    for row, position in enumerate(positions):
        value = cached["windows"][position]
        window_mask[row, :value.shape[0]] = True
        targets.append(cached["targets"][position])
    batch["window_mask"] = window_mask
    batch["targets"] = torch.stack(targets)
    return batch


def _torch_collate(items: list[dict[str, object]]) -> dict[str, Any]:
    import torch

    batch = collate_weld_batch(items)
    return {
        **{
            key: torch.as_tensor(batch[key])
            for key in ("windows", "window_mask", "sample_mask", "targets")
        },
        "sample_ids": batch["sample_ids"],
        "weld_ids": batch["weld_ids"],
        "image_groups": batch["image_groups"],
        "folds": batch["folds"],
    }


def _torch_time_frequency_collate(items: list[dict[str, object]]) -> dict[str, Any]:
    import torch

    batch = _torch_collate(items)
    values = [np.asarray(item["representations"], dtype=np.float32) for item in items]
    max_windows = max(value.shape[0] for value in values)
    representations = np.zeros(
        (len(values), max_windows, *values[0].shape[1:]), dtype=np.float32
    )
    for index, value in enumerate(values):
        representations[index, : value.shape[0]] = value
    batch["representations"] = torch.as_tensor(representations)
    return batch


def _to_device(value: Any | None, device: Any, *, non_blocking: bool = False) -> Any | None:
    return None if value is None else value.to(device, non_blocking=non_blocking)


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "Deep training requires the MMDII-Core train extra: pip install .[train]."
        ) from error
    return torch, nn, DataLoader


def _device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _validate_config(config: ExperimentConfig) -> None:
    if config.mode not in {"statistical", "random_forest", "full_signal", "window_mil", "second_stage"}:
        raise ValueError(
            "mode must be statistical, random_forest, full_signal, window_mil or second_stage."
        )
    if config.aggregator not in {"mean", "max", "topk_mean", "gated_attention"}:
        raise ValueError("Invalid aggregator.")
    if not config.target_codes or len(config.target_codes) != len(set(config.target_codes)):
        raise ValueError("target_codes must be non-empty and unique.")
    if config.fold_count != 5 or config.epochs < 1 or config.batch_size < 1:
        raise ValueError("fold_count must be 5 and epochs/batch_size positive.")
    if config.fold_scheme not in {"weld_independent", "image_group"}:
        raise ValueError("fold_scheme must be weld_independent or image_group.")
    if config.optimizer not in {"adamw", "adam"}:
        raise ValueError("optimizer must be adamw or adam.")
    if config.early_stopping_patience < 0 or config.early_stopping_min_delta < 0:
        raise ValueError("early stopping values must be non-negative.")
    if config.gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be non-negative.")
    if config.training.scheduler not in {"none", "cosine", "one_cycle"}:
        raise ValueError("training.scheduler must be none, cosine or one_cycle.")
    if config.training.warmup_epochs < 0 or config.training.warmup_epochs >= config.epochs:
        raise ValueError("training.warmup_epochs must be in [0, epochs).")
    if config.training.gradient_clip_norm < 0:
        raise ValueError("training.gradient_clip_norm must be non-negative.")
    if config.augmentation.amplitude_scale < 0 or config.augmentation.noise_std < 0:
        raise ValueError("augmentation scales must be non-negative.")
    if not 0.0 <= config.augmentation.time_mask_ratio <= 1.0:
        raise ValueError("augmentation.time_mask_ratio must be in [0, 1].")
    if config.augmentation.max_time_masks < 0:
        raise ValueError("augmentation.max_time_masks must be non-negative.")
    if config.tree.learner not in LEARNERS:
        raise ValueError("tree.learner must be random_forest or extra_trees.")
    if config.tree.n_estimators < 1 or config.tree.min_samples_leaf < 1:
        raise ValueError("tree.n_estimators and tree.min_samples_leaf must be positive.")
    if config.second_stage.feature_source not in FEATURE_SOURCES:
        raise ValueError("second_stage.feature_source must be stats, embedding or both.")
    if config.second_stage.representation_mode not in {"window_mil", "full_signal"}:
        raise ValueError("second_stage.representation_mode must be window_mil or full_signal.")
    if not 2 <= config.second_stage.inner_folds <= config.fold_count - 1:
        raise ValueError("second_stage.inner_folds must be between 2 and fold_count - 1.")
    if config.second_stage.encoder_epochs < 0:
        raise ValueError("second_stage.encoder_epochs must be non-negative.")
    if config.mode == "second_stage" and config.second_stage.feature_source != "stats":
        if config.aggregator not in {"mean", "gated_attention"}:
            raise ValueError(
                "second_stage with a deep representation requires mean or "
                "gated_attention aggregation."
            )
    if config.pipeline_backend not in {"cpu", "cuda"}:
        raise ValueError("pipeline_backend must be cpu or cuda.")
    if config.gpu_cache_gib <= 0:
        raise ValueError("gpu_cache_gib must be positive.")
    if config.target_fs <= 0 or config.window_seconds <= 0 or config.stride_seconds <= 0:
        raise ValueError("Preprocessing values must be positive.")
    if config.stride_seconds > config.window_seconds:
        raise ValueError("stride_seconds must not exceed window_seconds.")
    if config.model.encoder_chunk_size < 1:
        raise ValueError("encoder_chunk_size must be positive.")
    representation = config.representation
    if representation.name not in {"raw", "stft_256", "stft_512", "cwt_morl", "dwt_swt_db4"}:
        raise ValueError("Invalid representation.")
    if representation.name == "raw" and representation.encoder != "modern_tcn":
        raise ValueError("raw representation requires encoder=modern_tcn.")
    if representation.name != "raw" and representation.encoder not in {
        "cnn2d", "separable_cnn2d", "resnet2d_small", "resnet2d_multiscale", "convnext2d_lite"
    }:
        raise ValueError("Invalid time-frequency encoder.")
    if representation.fusion not in {
        "none",
        "raw_plus_stft",
        "raw_plus_cwt",
        "raw_plus_dwt",
    }:
        raise ValueError("Invalid fusion mode.")
    if representation.fusion == "raw_plus_stft" and representation.name not in {"stft_256", "stft_512"}:
        raise ValueError("raw_plus_stft requires an STFT representation.")
    if representation.fusion == "raw_plus_cwt" and representation.name != "cwt_morl":
        raise ValueError("raw_plus_cwt requires cwt_morl.")
    if representation.fusion == "raw_plus_dwt" and representation.name != "dwt_swt_db4":
        raise ValueError("raw_plus_dwt requires dwt_swt_db4.")
    if representation.output_time_bins < 1:
        raise ValueError("output_time_bins must be positive.")
    if representation.normalization != "per_channel_zscore":
        raise ValueError("Only per_channel_zscore normalization is supported.")
    parameters = _representation_transform_parameters(config)
    if representation.name.startswith("stft_"):
        if parameters["n_fft"] < 2 or not 1 <= parameters["hop_length"] <= parameters["n_fft"]:
            raise ValueError("STFT n_fft/hop_length are invalid.")
    if representation.name == "cwt_morl":
        if not representation.cwt_wavelet or representation.cwt_frequency_bins < 1:
            raise ValueError("CWT wavelet and frequency bin count are invalid.")
        if not 0 < parameters["min_frequency_hz"] <= parameters["max_frequency_hz"] <= config.target_fs / 2:
            raise ValueError("CWT frequency range must lie in (0, Nyquist].")
    if representation.name == "dwt_swt_db4":
        if not representation.dwt_wavelet or representation.dwt_level < 1:
            raise ValueError("DWT wavelet and level are invalid.")
    if any(fold < 0 or fold >= config.fold_count for fold in config.run_folds):
        raise ValueError("run_folds must contain valid fold indices.")


def _validate_index(index: DatasetIndex, config: ExperimentConfig) -> None:
    if index.target_codes != config.target_codes:
        raise ValueError("Dataset target codes do not match experiment configuration.")
    if {record.fold for record in index.records} != set(range(config.fold_count)):
        raise ValueError("Dataset folds do not match experiment configuration.")


def _json_config(config: ExperimentConfig) -> dict[str, object]:
    result = asdict(config)
    result["config_path"] = None if config.config_path is None else config.config_path.as_posix()
    result["release_directory"] = config.release_directory.as_posix()
    result["output_directory"] = config.output_directory.as_posix()
    result["preprocess_cache_directory"] = (
        None if config.preprocess_cache_directory is None else config.preprocess_cache_directory.as_posix()
    )
    result["representation_parameters"] = _representation_parameters(config)
    return result


def _representation_transform_parameters(config: ExperimentConfig) -> dict[str, object]:
    """Resolve the selected representation's auditable transform arguments."""

    representation = config.representation
    if representation.name == "stft_256":
        return {
            "n_fft": 256 if representation.stft_n_fft is None else representation.stft_n_fft,
            "hop_length": 64 if representation.stft_hop_length is None else representation.stft_hop_length,
        }
    if representation.name == "stft_512":
        return {
            "n_fft": 512 if representation.stft_n_fft is None else representation.stft_n_fft,
            "hop_length": 128 if representation.stft_hop_length is None else representation.stft_hop_length,
        }
    if representation.name == "cwt_morl":
        return {
            "wavelet": representation.cwt_wavelet,
            "frequency_bins": representation.cwt_frequency_bins,
            "min_frequency_hz": representation.cwt_min_frequency_hz,
            "max_frequency_hz": (
                config.target_fs / 2
                if representation.cwt_max_frequency_hz is None
                else representation.cwt_max_frequency_hz
            ),
        }
    if representation.name == "dwt_swt_db4":
        return {"wavelet": representation.dwt_wavelet, "level": representation.dwt_level}
    return {}


def _representation_parameters(config: ExperimentConfig) -> dict[str, object]:
    parameters = _representation_transform_parameters(config)
    if config.representation.name.startswith("stft_"):
        parameters.update({"window": "hann", "spectrum": "one_sided", "value": "log1p_power"})
    elif config.representation.name == "cwt_morl":
        parameters["value"] = "log1p_magnitude"
    elif config.representation.name == "dwt_swt_db4":
        parameters.update({"transform": "stationary_wavelet", "value": "log1p_magnitude"})
    if parameters:
        parameters.update(
            {
                "output_time_bins": config.representation.output_time_bins,
                "normalization": config.representation.normalization,
            }
        )
    return parameters


def _write_oof(path: Path, rows: list[dict[str, object]], target_codes: tuple[str, ...]) -> None:
    headers = (
        "sample_id",
        "weld_id",
        "image_group",
        "fold",
        "target_codes_json",
        *(f"prob_{code}" for code in target_codes),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
