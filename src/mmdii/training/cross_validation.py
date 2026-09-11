"""Reproducible weld-level cross-validation runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import csv
import json
from pathlib import Path
import random
import tomllib
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
    extract_statistical_features,
    fit_predict_logistic_ovr,
    fit_predict_random_forest_ovr,
)


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
    model: ModelConfig = field(default_factory=ModelConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    run_folds: tuple[int, ...] = ()

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
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        experiment = payload["experiment"]
        model_payload = payload.get("model", {})
        representation_payload = payload.get("representation", {})
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
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid experiment configuration: {error}") from error
    _validate_config(config)
    return config


def run_cross_validation(
    index: DatasetIndex | None, config: ExperimentConfig
) -> dict[str, object]:
    """Run one model per held-out weld fold and publish weld-level OOF output."""

    if config.mode not in {"statistical", "random_forest"}:
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
    for fold in folds:
        train_records = tuple(record for record in index.records if record.fold != fold)
        valid_records = tuple(record for record in index.records if record.fold == fold)
        if not train_records or not valid_records:
            raise ValueError(f"Fold {fold} has no train or validation records.")
        train_targets = np.asarray([record.target for record in train_records], dtype=np.float64)
        class_weights = compute_positive_class_weights(train_targets)
        training_info: dict[str, object] = {}
        if config.mode in {"statistical", "random_forest"}:
            probabilities = _run_statistical_fold(
                index, train_records, valid_records, config
            )
        else:
            normalizer = FoldNormalizer.fit(index, train_records)
            if config.mode == "full_signal":
                spec: WindowSpec | FullSignalSpec = FullSignalSpec(
                    target_fs=config.target_fs,
                    output_samples=config.full_signal_samples,
                )
                aggregator = "mean"
            else:
                spec = WindowSpec(
                    target_fs=config.target_fs,
                    window_seconds=config.window_seconds,
                    stride_seconds=config.stride_seconds,
                )
                aggregator = config.aggregator
            train_dataset = WeldWindowDataset(index, {record.fold for record in train_records}, spec, normalizer)
            valid_dataset = WeldWindowDataset(index, {fold}, spec, normalizer)
            if config.representation.name != "raw":
                train_dataset, valid_dataset = _time_frequency_datasets(
                    train_dataset, valid_dataset, config
                )
            model = _build_deep_model(config, aggregator, torch)
            probabilities, training_info = _run_deep_fold(
                model,
                train_dataset,
                valid_dataset,
                class_weights,
                config,
                torch,
                nn,
                DataLoader,
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
    return summary


def _run_statistical_fold(index: DatasetIndex, train_records: tuple[Any, ...], valid_records: tuple[Any, ...], config: ExperimentConfig) -> np.ndarray:
    train_features = np.stack(
        [extract_statistical_features(index.load_signal(record)[0]) for record in train_records]
    )
    valid_features = np.stack(
        [extract_statistical_features(index.load_signal(record)[0]) for record in valid_records]
    )
    predictor = fit_predict_random_forest_ovr if config.mode == "random_forest" else fit_predict_logistic_ovr
    return predictor(
        train_features,
        np.asarray([record.target for record in train_records], dtype=np.float64),
        valid_features,
        target_codes=config.target_codes,
        random_state=config.seed,
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

        def forward(
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
            embeddings = embeddings.reshape(batch_size, window_count, -1)
            return self.head(embeddings, window_mask)

    return WeldModel()


class _TimeFrequencyDataset:
    """Lazily transform an existing weld-window dataset into 2-D inputs."""

    def __init__(self, source: Any, config: ExperimentConfig, normalizer: Any) -> None:
        self.source = source
        self.config = config
        self.normalizer = normalizer
        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, position: int) -> dict[str, object]:
        from mmdii.data.time_frequency import transform_representation

        if position in self._cache:
            return self._cache[position]
        item = dict(self.source[position])
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
        normalized = self.normalizer.transform(np.stack(transformed))
        normalized *= np.asarray(time_masks, dtype=np.float32)[:, np.newaxis, np.newaxis, :]
        item["representations"] = normalized
        self._cache[position] = item
        return item


def _time_frequency_datasets(
    train_dataset: Any, valid_dataset: Any, config: ExperimentConfig
) -> tuple[Any, Any]:
    from mmdii.data.time_frequency import RepresentationNormalizer, transform_representation

    arrays = []
    for position in range(len(train_dataset)):
        item = train_dataset[position]
        windows = np.asarray(item["windows"], dtype=np.float64)
        masks = np.asarray(item["sample_mask"], dtype=bool)
        arrays.append(
            np.stack(
                [
                    transform_representation(
                        window,
                        config.representation.name,
                        sample_mask=mask,
                        target_fs=config.target_fs,
                        output_time_bins=config.representation.output_time_bins,
                        **_representation_transform_parameters(config),
                    )[0]
                    for window, mask in zip(windows, masks, strict=True)
                ]
            )
        )
    normalizer = RepresentationNormalizer.fit(arrays)
    return (
        _TimeFrequencyDataset(train_dataset, config, normalizer),
        _TimeFrequencyDataset(valid_dataset, config, normalizer),
    )


def _run_deep_fold(model: Any, train_dataset: Any, valid_dataset: Any, class_weights: np.ndarray, config: ExperimentConfig, torch: Any, nn: Any, DataLoader: Any) -> tuple[np.ndarray, dict[str, object]]:
    device = _device(config.device, torch)
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0
    for epoch in range(config.epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                batch["windows"].to(device),
                batch["window_mask"].to(device),
                batch["sample_mask"].to(device),
                _to_device(batch.get("representations"), device),
            )
            loss = criterion(logits, batch["targets"].to(device))
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            optimizer.step()
            train_loss_total += float(loss.detach().cpu())
            train_batches += 1
        epochs_ran = epoch + 1
        epoch_loss = train_loss_total / max(train_batches, 1)
        if config.early_stopping_patience > 0:
            if epoch_loss < best_loss - config.early_stopping_min_delta:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in valid_loader:
            logits, _ = model(
                batch["windows"].to(device),
                batch["window_mask"].to(device),
                batch["sample_mask"].to(device),
                _to_device(batch.get("representations"), device),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities, axis=0), {
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch or epochs_ran,
        "early_stopping_monitor": "train_loss" if config.early_stopping_patience > 0 else None,
    }


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


def _to_device(value: Any | None, device: Any) -> Any | None:
    return None if value is None else value.to(device)


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
    if config.mode not in {"statistical", "random_forest", "full_signal", "window_mil"}:
        raise ValueError("mode must be statistical, random_forest, full_signal or window_mil.")
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
        "cnn2d", "separable_cnn2d", "resnet2d_small", "convnext2d_lite"
    }:
        raise ValueError("Invalid time-frequency encoder.")
    if representation.fusion not in {"none", "raw_plus_stft", "raw_plus_cwt"}:
        raise ValueError("Invalid fusion mode.")
    if representation.fusion == "raw_plus_stft" and representation.name not in {"stft_256", "stft_512"}:
        raise ValueError("raw_plus_stft requires an STFT representation.")
    if representation.fusion == "raw_plus_cwt" and representation.name != "cwt_morl":
        raise ValueError("raw_plus_cwt requires cwt_morl.")
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
