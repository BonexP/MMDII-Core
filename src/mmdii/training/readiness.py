"""Training-host checks and one-step real-data smoke training."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import math
import sys
import tempfile
from typing import Any

import numpy as np

from mmdii.data.training_dataset import (
    DatasetIndex,
    FoldNormalizer,
    WeldWindowDataset,
    WindowSpec,
    collate_weld_batch,
)
from mmdii.evaluation.multilabel import compute_positive_class_weights
from mmdii.training.cross_validation import ExperimentConfig, _build_deep_model


def inspect_environment(
    config: ExperimentConfig,
    *,
    release_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Return a complete, JSON-compatible training-host readiness report."""

    release = Path(release_directory or config.release_directory).resolve()
    output = Path(output_directory or config.output_directory).resolve()
    packages, torch, errors = _inspect_packages()
    python_supported = sys.version_info >= (3, 11)
    if not python_supported:
        errors.append("Python 3.11 or newer is required.")

    accelerator: dict[str, object] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
            accelerator = {
                "cuda_available": cuda_available,
                "device_count": device_count,
                "devices": [
                    torch.cuda.get_device_name(position) for position in range(device_count)
                ],
            }
            if config.device.startswith("cuda") and not cuda_available:
                errors.append(f"Configured device {config.device!r} is unavailable.")
        except Exception as error:
            errors.append(f"Accelerator check failed: {error}")

    dataset: dict[str, object] = {
        "valid": False,
        "release_directory": str(release),
        "sample_count": 0,
        "folds": [],
        "image_group_count": 0,
        "target_codes": list(config.target_codes),
    }
    try:
        index = DatasetIndex.from_release(release, config.target_codes)
        folds = sorted({record.fold for record in index.records})
        if folds != list(range(config.fold_count)):
            raise ValueError("Dataset folds do not match the experiment configuration.")
        dataset.update(
            valid=True,
            sample_count=len(index.records),
            folds=folds,
            image_group_count=len({record.image_group for record in index.records}),
        )
    except Exception as error:  # report all readiness failures together
        errors.append(f"Dataset check failed: {error}")

    output_report = {"path": str(output), "writable": False}
    try:
        _check_output_directory(output)
        output_report["writable"] = True
    except OSError as error:
        errors.append(f"Output check failed: {error}")

    return {
        "ok": not errors,
        "python": {
            "version": ".".join(str(part) for part in sys.version_info[:3]),
            "supported": python_supported,
        },
        "packages": packages,
        "accelerator": accelerator,
        "dataset": dataset,
        "output": output_report,
        "errors": errors,
    }


def run_real_data_smoke(
    config: ExperimentConfig,
    *,
    release_directory: str | Path | None = None,
    fold: int = 0,
    batch_size: int = 1,
    device_override: str | None = None,
) -> dict[str, object]:
    """Run one optimizer step on a real Dataset v0.2 batch."""

    if fold not in range(config.fold_count):
        raise ValueError(f"fold must be between 0 and {config.fold_count - 1}.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    torch = _require_torch()
    release = Path(release_directory or config.release_directory).resolve()
    index = DatasetIndex.from_release(release, config.target_codes)
    train_records = tuple(record for record in index.records if record.fold != fold)
    train_folds = {record.fold for record in train_records}
    if not train_records or train_folds != set(range(config.fold_count)) - {fold}:
        raise ValueError("The selected fold does not leave a complete training partition.")
    class_weights = compute_positive_class_weights(
        np.asarray([record.target for record in train_records], dtype=np.float64)
    )
    normalizer = FoldNormalizer.fit(index, train_records)
    dataset = WeldWindowDataset(
        index,
        train_folds,
        WindowSpec(
            target_fs=config.target_fs,
            window_seconds=config.window_seconds,
            stride_seconds=config.stride_seconds,
        ),
        normalizer,
    )
    return _execute_smoke_batch(
        torch=torch,
        config=config,
        dataset=dataset,
        class_weights=class_weights,
        batch_size=batch_size,
        held_out_fold=fold,
        requested_device=device_override or config.device,
    )


def _inspect_packages() -> tuple[dict[str, dict[str, object]], Any | None, list[str]]:
    packages: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    torch = None
    for name, module_name in (("scipy", "scipy"), ("sklearn", "sklearn"), ("torch", "torch")):
        try:
            module = import_module(module_name)
        except ImportError:
            packages[name] = {"available": False, "version": None}
            errors.append(f"Missing package: {name}")
            continue
        except Exception as error:
            packages[name] = {"available": False, "version": None}
            errors.append(f"{name} import failed: {error}")
            continue
        packages[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
        if name == "torch":
            torch = module
    return packages, torch, errors


def _check_output_directory(path: Path) -> None:
    target = path.resolve()
    if target.exists() and not target.is_dir():
        raise OSError(f"Output path is not a directory: {target}")
    existing = target
    while not existing.exists():
        if existing.parent == existing:
            raise OSError(f"No existing parent for output path: {target}")
        existing = existing.parent
    if not existing.is_dir():
        raise OSError(f"Output parent is not a directory: {existing}")
    with tempfile.NamedTemporaryFile(prefix="mmdii-write-check-", dir=existing):
        pass


def _require_torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "Real-data smoke training requires the MMDII-Core train extra: "
            "pip install -e .[train]."
        ) from error


def _execute_smoke_batch(
    *,
    torch: Any,
    config: ExperimentConfig,
    dataset: WeldWindowDataset,
    class_weights: np.ndarray,
    batch_size: int,
    held_out_fold: int,
    requested_device: str,
) -> dict[str, object]:
    if len(dataset) == 0:
        raise ValueError("The smoke-training partition is empty.")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    batch = collate_weld_batch(
        dataset[position] for position in range(min(batch_size, len(dataset)))
    )
    tensors = {
        name: torch.as_tensor(batch[name], device=device)
        for name in ("windows", "window_mask", "sample_mask", "targets")
    }
    model = _build_deep_model(config, config.aggregator, torch).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    tracked_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    before = tracked_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(
        tensors["windows"], tensors["window_mask"], tensors["sample_mask"]
    )
    loss = criterion(logits, tensors["targets"])
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
        raise RuntimeError("Smoke training produced a non-finite logit or loss.")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("Smoke training produced missing or non-finite gradients.")
    gradient_norm = math.sqrt(
        sum(float(gradient.detach().square().sum().item()) for gradient in gradients)
    )
    optimizer.step()
    parameter_delta = float((tracked_parameter.detach() - before).abs().max().item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0 or parameter_delta <= 0.0:
        raise RuntimeError("Smoke training did not produce a finite parameter update.")

    return {
        "ok": True,
        "held_out_fold": held_out_fold,
        "training_folds": sorted({int(record.fold) for record in dataset.records}),
        "sample_ids": list(batch["sample_ids"]),
        "weld_ids": list(batch["weld_ids"]),
        "device": str(device),
        "aggregator": config.aggregator,
        "window_shape": list(tensors["windows"].shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().item()),
        "gradient_norm": gradient_norm,
        "parameter_max_delta": parameter_delta,
    }
