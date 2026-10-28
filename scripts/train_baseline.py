"""Training-host entry point for the MMDII baseline experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mmdii.data.training_dataset import DatasetIndex
from mmdii.reporting import configure_logging
from mmdii.training.cross_validation import (
    ExperimentConfig,
    RepresentationConfig,
    SecondStageConfig,
    TreeConfig,
    load_experiment_config,
    run_cross_validation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "statistical",
            "random_forest",
            "gradient_boosting",
            "rbf_svm",
            "full_signal",
            "window_mil",
            "second_stage",
        ),
    )
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
    parser.add_argument("--pipeline-backend", choices=("cpu", "cuda"))
    parser.add_argument("--preprocess-cache-directory", type=Path)
    parser.add_argument("--gpu-cache-gib", type=float)
    parser.add_argument("--fold-scheme", choices=("weld_independent", "image_group"))
    parser.add_argument(
        "--representation",
        choices=("raw", "stft_256", "stft_512", "cwt_morl", "dwt_swt_db4"),
    )
    parser.add_argument(
        "--encoder",
        choices=("modern_tcn", "cnn2d", "separable_cnn2d", "resnet2d_small", "resnet2d_multiscale", "convnext2d_lite", "resnet2d_wide", "resnet2d_separable", "convnext2d_deep"),
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "raw_plus_stft", "raw_plus_cwt", "raw_plus_dwt"),
    )
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
    parser.add_argument(
        "--tree-learner",
        choices=("random_forest", "extra_trees"),
        help="Tree ensemble used by random_forest or second_stage mode.",
    )
    parser.add_argument("--tree-n-estimators", type=int)
    parser.add_argument("--tree-min-samples-leaf", type=int)
    parser.add_argument("--tree-max-features", choices=("sqrt", "all"))
    parser.add_argument("--tree-max-depth", type=int)
    parser.add_argument("--tree-criterion", choices=("gini", "log_loss"))
    parser.add_argument("--tree-class-weight", choices=("balanced", "none"))
    parser.add_argument(
        "--tree-bootstrap",
        choices=("true", "false", "default"),
        help="Explicit ensemble bootstrap setting; default delegates to the learner default.",
    )
    parser.add_argument(
        "--tree-feature-set",
        choices=("stats18", "stats15_no_ptp", "stats12_no_ptp_rms"),
    )
    parser.add_argument("--tree-tune", action="store_true")
    parser.add_argument(
        "--second-stage-source",
        choices=("stats", "embedding", "both"),
        help="Weld-level features the second-stage tree is fitted on.",
    )
    parser.add_argument(
        "--second-stage-mode",
        choices=("window_mil", "full_signal"),
        help="Encoder input layout used to build weld-level representations.",
    )
    parser.add_argument("--second-stage-inner-folds", type=int)
    parser.add_argument("--second-stage-encoder-epochs", type=int)
    parser.add_argument("--second-stage-projection", choices=("none", "pca"))
    parser.add_argument("--second-stage-components", type=int)
    parser.add_argument("--fold", type=int, action="append", dest="run_folds")
    args = parser.parse_args(argv)
    configure_logging()
    config = load_experiment_config(args.config)
    overrides: dict[str, object] = {}
    if args.release_dir is not None:
        overrides["release_directory"] = args.release_dir.resolve()
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.fold_scheme is not None:
        overrides["fold_scheme"] = args.fold_scheme
    if args.pipeline_backend is not None:
        overrides["pipeline_backend"] = args.pipeline_backend
    if args.preprocess_cache_directory is not None:
        overrides["preprocess_cache_directory"] = args.preprocess_cache_directory.resolve()
    if args.gpu_cache_gib is not None:
        overrides["gpu_cache_gib"] = args.gpu_cache_gib
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
    tree_overrides = {
        "learner": args.tree_learner,
        "n_estimators": args.tree_n_estimators,
        "min_samples_leaf": args.tree_min_samples_leaf,
        "max_features": args.tree_max_features,
        "max_depth": args.tree_max_depth,
        "criterion": args.tree_criterion,
        "class_weight": args.tree_class_weight,
        "bootstrap": (
            {"true": True, "false": False, "default": None}.get(args.tree_bootstrap)
            if args.tree_bootstrap is not None
            else None
        ),
        "feature_set": args.tree_feature_set,
    }
    tree_overrides = {key: value for key, value in tree_overrides.items() if value is not None}
    if args.tree_bootstrap == "default":
        tree_overrides["bootstrap"] = None
    if tree_overrides.get("max_depth") == 0:
        tree_overrides["max_depth"] = None
    if args.tree_tune:
        tree_overrides["tune"] = True
    if tree_overrides:
        overrides["tree"] = TreeConfig(
            **{
                field_name: tree_overrides.get(
                    field_name, getattr(config.tree, field_name)
                )
                for field_name in TreeConfig.__dataclass_fields__
            }
        )
    second_stage_overrides = {
        "feature_source": args.second_stage_source,
        "representation_mode": args.second_stage_mode,
        "inner_folds": args.second_stage_inner_folds,
        "encoder_epochs": args.second_stage_encoder_epochs,
        "embedding_projection": args.second_stage_projection,
        "embedding_components": args.second_stage_components,
    }
    second_stage_overrides = {
        key: value for key, value in second_stage_overrides.items() if value is not None
    }
    if args.mode == "second_stage":
        second_stage_overrides.setdefault("enabled", True)
    if second_stage_overrides:
        overrides["second_stage"] = SecondStageConfig(
            **{
                field_name: second_stage_overrides.get(
                    field_name, getattr(config.second_stage, field_name)
                )
                for field_name in SecondStageConfig.__dataclass_fields__
            }
        )
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
    if args.gradient_clip_norm is not None:
        overrides["training"] = replace(
            config.training, gradient_clip_norm=args.gradient_clip_norm
        )
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
