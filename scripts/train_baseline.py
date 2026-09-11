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
    RepresentationConfig,
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
    parser.add_argument("--fold-scheme", choices=("weld_independent", "image_group"))
    parser.add_argument(
        "--representation",
        choices=("raw", "stft_256", "stft_512", "cwt_morl", "dwt_swt_db4"),
    )
    parser.add_argument(
        "--encoder",
        choices=("modern_tcn", "cnn2d", "separable_cnn2d", "resnet2d_small", "convnext2d_lite"),
    )
    parser.add_argument("--fusion", choices=("none", "raw_plus_stft", "raw_plus_cwt"))
    parser.add_argument("--output-time-bins", type=int)
    parser.add_argument("--representation-normalization", choices=("per_channel_zscore",))
    parser.add_argument("--stft-n-fft", type=int)
    parser.add_argument("--stft-hop-length", type=int)
    parser.add_argument("--cwt-wavelet")
    parser.add_argument("--cwt-frequency-bins", type=int)
    parser.add_argument("--cwt-min-frequency-hz", type=float)
    parser.add_argument("--cwt-max-frequency-hz", type=float)
    parser.add_argument("--dwt-wavelet")
    parser.add_argument("--dwt-level", type=int)
    parser.add_argument("--fold", type=int, action="append", dest="run_folds")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    overrides: dict[str, object] = {}
    if args.release_dir is not None:
        overrides["release_directory"] = args.release_dir.resolve()
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.fold_scheme is not None:
        overrides["fold_scheme"] = args.fold_scheme
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.aggregator is not None:
        overrides["aggregator"] = args.aggregator
    representation_overrides = {}
    for argument_name, config_name in (
        ("representation", "name"),
        ("encoder", "encoder"),
        ("fusion", "fusion"),
        ("output_time_bins", "output_time_bins"),
        ("representation_normalization", "normalization"),
        ("stft_n_fft", "stft_n_fft"),
        ("stft_hop_length", "stft_hop_length"),
        ("cwt_wavelet", "cwt_wavelet"),
        ("cwt_frequency_bins", "cwt_frequency_bins"),
        ("cwt_min_frequency_hz", "cwt_min_frequency_hz"),
        ("cwt_max_frequency_hz", "cwt_max_frequency_hz"),
        ("dwt_wavelet", "dwt_wavelet"),
        ("dwt_level", "dwt_level"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            representation_overrides[config_name] = value
    if representation_overrides:
        overrides["representation"] = RepresentationConfig(
            **{
                field_name: representation_overrides.get(
                    field_name, getattr(config.representation, field_name)
                )
                for field_name in RepresentationConfig.__dataclass_fields__
            }
        )
    if args.run_folds is not None:
        overrides["run_folds"] = tuple(args.run_folds)
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
    index = DatasetIndex.from_release(
        config.release_directory, config.target_codes, fold_scheme=config.fold_scheme
    )
    summary = run_cross_validation(index, config)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
