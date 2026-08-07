"""Reproducible weld-level cross-validation runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    model: ModelConfig = field(default_factory=ModelConfig)

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
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        experiment = payload["experiment"]
        model_payload = payload.get("model", {})
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
            model=ModelConfig(
                **{
                    field_name: model_payload.get(field_name, getattr(ModelConfig(), field_name))
                    for field_name in ModelConfig.__dataclass_fields__
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

    if config.mode != "statistical":
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

    for fold in range(config.fold_count):
        train_records = tuple(record for record in index.records if record.fold != fold)
        valid_records = tuple(record for record in index.records if record.fold == fold)
        if not train_records or not valid_records:
            raise ValueError(f"Fold {fold} has no train or validation records.")
        train_targets = np.asarray([record.target for record in train_records], dtype=np.float64)
        class_weights = compute_positive_class_weights(train_targets)
        if config.mode == "statistical":
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
            model = _build_deep_model(config, aggregator, torch)
            probabilities = _run_deep_fold(
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
    return fit_predict_logistic_ovr(
        train_features,
        np.asarray([record.target for record in train_records], dtype=np.float64),
        valid_features,
        target_codes=config.target_codes,
        random_state=config.seed,
    )


def _build_deep_model(config: ExperimentConfig, aggregator: str, torch: Any) -> Any:
    from mmdii.models.mil import WeldMIL
    from mmdii.models.modern_tcn import ModernTCNSmall

    encoder = ModernTCNSmall(
        input_channels=3,
        hidden_channels=config.model.hidden_channels,
        embedding_dim=config.model.embedding_dim,
        kernel_size=config.model.kernel_size,
        block_count=config.model.block_count,
        dropout=config.model.dropout,
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

        def forward(self, windows: Any, window_mask: Any, sample_mask: Any) -> Any:
            batch_size, window_count, channels, sample_count = windows.shape
            safe_mask = sample_mask.clone()
            safe_mask[~window_mask, 0] = True
            embeddings = self.encoder(
                windows.reshape(batch_size * window_count, channels, sample_count),
                sample_mask=safe_mask.reshape(batch_size * window_count, sample_count),
            )
            embeddings = embeddings.reshape(batch_size, window_count, -1)
            return self.head(embeddings, window_mask)

    return WeldModel()


def _run_deep_fold(model: Any, train_dataset: Any, valid_dataset: Any, class_weights: np.ndarray, config: ExperimentConfig, torch: Any, nn: Any, DataLoader: Any) -> np.ndarray:
    device = _device(config.device, torch)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_torch_collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_torch_collate,
    )
    for _ in range(config.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                batch["windows"].to(device),
                batch["window_mask"].to(device),
                batch["sample_mask"].to(device),
            )
            loss = criterion(logits, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in valid_loader:
            logits, _ = model(
                batch["windows"].to(device),
                batch["window_mask"].to(device),
                batch["sample_mask"].to(device),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


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
    if config.mode not in {"statistical", "full_signal", "window_mil"}:
        raise ValueError("mode must be statistical, full_signal or window_mil.")
    if config.aggregator not in {"mean", "max", "topk_mean", "gated_attention"}:
        raise ValueError("Invalid aggregator.")
    if not config.target_codes or len(config.target_codes) != len(set(config.target_codes)):
        raise ValueError("target_codes must be non-empty and unique.")
    if config.fold_count != 5 or config.epochs < 1 or config.batch_size < 1:
        raise ValueError("fold_count must be 5 and epochs/batch_size positive.")
    if config.target_fs <= 0 or config.window_seconds <= 0 or config.stride_seconds <= 0:
        raise ValueError("Preprocessing values must be positive.")
    if config.stride_seconds > config.window_seconds:
        raise ValueError("stride_seconds must not exceed window_seconds.")


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
    return result


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
