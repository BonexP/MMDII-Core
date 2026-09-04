"""Training-host entry point for the MMDII baseline experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mmdii.data.training_dataset import DatasetIndex
from mmdii.training.cross_validation import (
    ExperimentConfig,
    load_experiment_config,
    run_cross_validation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("statistical", "random_forest", "full_signal", "window_mil"))
    parser.add_argument(
        "--aggregator",
        choices=("mean", "max", "topk_mean", "gated_attention"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optimizer", choices=("adamw", "adam"))
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    overrides: dict[str, object] = {}
    if args.release_dir is not None:
        overrides["release_directory"] = args.release_dir.resolve()
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.aggregator is not None:
        overrides["aggregator"] = args.aggregator
    for argument_name, config_name in (
        ("seed", "seed"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("optimizer", "optimizer"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("early_stopping_min_delta", "early_stopping_min_delta"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("device", "device"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            overrides[config_name] = value
    if overrides:
        config = replace(config, **overrides)
    index = DatasetIndex.from_release(config.release_directory, config.target_codes)
    summary = run_cross_validation(index, config)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
