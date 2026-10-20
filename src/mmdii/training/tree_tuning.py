"""Leak-safe inner-fold tuning for statistical tree controls."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from mmdii.evaluation.multilabel import evaluate_multilabel
from mmdii.models.statistical import fit_predict_forest_ovr


TUNING_INNER_FOLDS = 4
TUNING_N_ESTIMATORS = 500


def _candidate_grid() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "n_estimators": TUNING_N_ESTIMATORS,
            "max_depth": None,
            "criterion": "gini",
        }
        for min_samples_leaf in (4, 2, 1)
        for max_features in ("sqrt", "all")
        for class_weight in ("balanced", None)
    )


def _inner_partitions(fold_ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    folds = tuple(sorted({int(value) for value in fold_ids}))
    if len(folds) < TUNING_INNER_FOLDS:
        raise ValueError(
            f"Tree tuning requires at least {TUNING_INNER_FOLDS} distinct training folds."
        )
    groups: list[list[int]] = [[] for _ in range(TUNING_INNER_FOLDS)]
    for position, fold in enumerate(folds):
        groups[position % TUNING_INNER_FOLDS].append(fold)
    return tuple(tuple(group) for group in groups)


def tune_and_predict_forest_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    *,
    train_fold_ids: Iterable[int],
    target_codes: Iterable[str],
    random_state: int,
    learner: str = "random_forest",
) -> tuple[np.ndarray, dict[str, object]]:
    """Select one tree configuration on inner folds, then fit the outer model.

    ``train_fold_ids`` identify only the outer-training records.  Each candidate
    is scored on held-out groups of those records, and the selected candidate is
    refit on all outer-training records before predicting ``valid_features``.
    """

    train_x = np.asarray(train_features, dtype=np.float64)
    train_y = np.asarray(train_targets, dtype=np.float64)
    valid_x = np.asarray(valid_features, dtype=np.float64)
    folds = np.asarray(tuple(int(value) for value in train_fold_ids), dtype=np.int64)
    codes = tuple(target_codes)
    if folds.shape != (train_x.shape[0],):
        raise ValueError("train_fold_ids must match the training feature rows.")
    if train_x.ndim != 2 or train_y.ndim != 2 or valid_x.ndim != 2:
        raise ValueError("Tree tuning expects two-dimensional feature and target matrices.")
    if train_y.shape[1] != len(codes):
        raise ValueError("Target columns do not match target_codes.")
    partitions = _inner_partitions(folds)
    for group in partitions:
        validation = np.isin(folds, group)
        fitting = ~validation
        if not fitting.any() or not validation.any():
            raise ValueError("Tree tuning inner partition has an empty side.")
        for name, mask in (("training", fitting), ("validation", validation)):
            if np.any(train_y[mask].sum(axis=0) == 0) or np.any(
                train_y[mask].sum(axis=0) == mask.sum()
            ):
                raise ValueError(
                    f"Tree tuning inner {name} partition lacks a positive or negative class."
                )

    candidate_reports: list[dict[str, object]] = []
    best_score = float("-inf")
    best_params: dict[str, object] | None = None
    for candidate_index, params in enumerate(_candidate_grid()):
        scores: list[float] = []
        for group in partitions:
            validation = np.isin(folds, group)
            probabilities = fit_predict_forest_ovr(
                train_x[~validation],
                train_y[~validation],
                train_x[validation],
                target_codes=codes,
                random_state=random_state,
                learner=learner,
                **params,
            )
            metrics = evaluate_multilabel(
                train_y[validation],
                probabilities,
                target_codes=codes,
            )
            score = metrics["macro_pr_auc"]
            if score is None:
                raise ValueError("Tree tuning received an undefined inner macro PR-AUC.")
            scores.append(float(score))
        mean_score = float(np.mean(scores))
        candidate_reports.append(
            {
                "candidate_index": candidate_index,
                "params": params,
                "inner_macro_pr_auc": scores,
                "mean_inner_macro_pr_auc": mean_score,
            }
        )
        simpler = lambda values: (
            -int(values["min_samples_leaf"]),
            0 if values["max_features"] == "sqrt" else 1,
            0 if values["class_weight"] == "balanced" else 1,
        )
        if (
            mean_score > best_score + 1e-6
            or (
                best_params is not None
                and abs(mean_score - best_score) < 1e-6
                and simpler(params) < simpler(best_params)
            )
        ):
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Tree tuning did not evaluate any candidate.")
    probabilities = fit_predict_forest_ovr(
        train_x,
        train_y,
        valid_x,
        target_codes=codes,
        random_state=random_state,
        learner=learner,
        **best_params,
    )
    diagnostics = {
        "inner_folds": TUNING_INNER_FOLDS,
        "inner_partitions": [list(group) for group in partitions],
        "candidate_count": len(candidate_reports),
        "candidates": candidate_reports,
        "selected_params": best_params,
        "selected_mean_inner_macro_pr_auc": best_score,
        "inner_training_positive_counts": [
            train_y[~np.isin(folds, group)].sum(axis=0).astype(int).tolist()
            for group in partitions
        ],
        "inner_validation_positive_counts": [
            train_y[np.isin(folds, group)].sum(axis=0).astype(int).tolist()
            for group in partitions
        ],
    }
    return probabilities, diagnostics
