"""Fold-aligned late fusion of two archived weld-level OOF prediction sets.

The whole point of this module is that fusion weights may only be chosen from
data the held-out fold never contributed to.  Every selection helper here
therefore takes the fold assignment and excludes the fold being predicted.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class OofFrames:
    """One archived run's weld-level out-of-fold predictions, sorted by sample."""

    run_directory: Path
    target_codes: tuple[str, ...]
    sample_ids: tuple[str, ...]
    weld_ids: tuple[str, ...]
    image_groups: tuple[str, ...]
    folds: np.ndarray
    truth: np.ndarray
    probabilities: np.ndarray


def load_oof(run_directory: str | Path, target_codes: Iterable[str]) -> OofFrames:
    """Read ``oof_predictions.csv`` from one run directory."""

    directory = Path(run_directory)
    codes = tuple(target_codes)
    path = directory / "oof_predictions.csv"
    if not path.is_file():
        raise ValueError(f"Missing OOF predictions: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"OOF predictions are empty: {path}")
    missing = [f"prob_{code}" for code in codes if f"prob_{code}" not in rows[0]]
    if missing:
        raise ValueError(f"OOF predictions lack columns: {', '.join(missing)}")

    rows.sort(key=lambda row: row["sample_id"])
    sample_ids = tuple(row["sample_id"] for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"OOF sample_id values are not unique: {path}")

    truth = np.zeros((len(rows), len(codes)), dtype=np.int64)
    probabilities = np.zeros((len(rows), len(codes)), dtype=np.float64)
    folds = np.zeros(len(rows), dtype=np.int64)
    for position, row in enumerate(rows):
        defect_codes = json.loads(row["target_codes_json"])
        truth[position] = [int(code in defect_codes) for code in codes]
        probabilities[position] = [float(row[f"prob_{code}"]) for code in codes]
        folds[position] = int(row["fold"])
    if not np.isfinite(probabilities).all():
        raise ValueError(f"OOF probabilities are not finite: {path}")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(f"OOF probabilities fall outside [0, 1]: {path}")

    return OofFrames(
        run_directory=directory,
        target_codes=codes,
        sample_ids=sample_ids,
        weld_ids=tuple(row["weld_id"] for row in rows),
        image_groups=tuple(row["image_group"] for row in rows),
        folds=folds,
        truth=truth,
        probabilities=probabilities,
    )


def align_oof(tree: OofFrames, deep: OofFrames) -> tuple[OofFrames, OofFrames]:
    """Return the two frames restricted to their common welds, with matched folds.

    Fusion of two runs is only meaningful when they cover the same welds under
    the same fold assignment and the same labels.  Any mismatch is an error
    rather than something to paper over.
    """

    if tree.target_codes != deep.target_codes:
        raise ValueError("OOF frames use different target codes.")
    if tree.sample_ids != deep.sample_ids:
        only_tree = sorted(set(tree.sample_ids) - set(deep.sample_ids))
        only_deep = sorted(set(deep.sample_ids) - set(tree.sample_ids))
        raise ValueError(
            "OOF frames cover different welds "
            f"(tree only: {only_tree[:3]}, deep only: {only_deep[:3]})."
        )
    if not np.array_equal(tree.folds, deep.folds):
        mismatched = int(np.sum(tree.folds != deep.folds))
        raise ValueError(f"OOF frames disagree on the fold of {mismatched} welds.")
    if not np.array_equal(tree.truth, deep.truth):
        mismatched = int(np.sum(np.any(tree.truth != deep.truth, axis=1)))
        raise ValueError(f"OOF frames disagree on labels for {mismatched} welds.")
    if not np.array_equal(tree.weld_ids, deep.weld_ids):
        raise ValueError("OOF frames disagree on weld identifiers.")
    return tree, deep


def _per_class_average_precision(truth: np.ndarray, scores: np.ndarray) -> list[float | None]:
    from sklearn.metrics import average_precision_score

    values: list[float | None] = []
    for position in range(truth.shape[1]):
        if truth[:, position].sum() == 0:
            values.append(None)
        else:
            values.append(float(average_precision_score(truth[:, position], scores[:, position])))
    return values


def macro_average_precision(truth: np.ndarray, scores: np.ndarray) -> float | None:
    """Macro PR-AUC over codes that have at least one positive in ``truth``."""

    per_class = _per_class_average_precision(truth, scores)
    defined = [value for value in per_class if value is not None]
    return float(np.mean(defined)) if defined else None


def fold_mean_macro_average_precision(
    truth: np.ndarray, scores: np.ndarray, folds: np.ndarray
) -> float | None:
    """Mean over folds of each fold's macro PR-AUC (the project's other convention)."""

    per_fold = []
    for fold in sorted(set(int(value) for value in folds)):
        valid = folds == fold
        value = macro_average_precision(truth[valid], scores[valid])
        if value is not None:
            per_fold.append(value)
    return float(np.mean(per_fold)) if per_fold else None


def brier_scores(truth: np.ndarray, scores: np.ndarray) -> tuple[float, list[float | None]]:
    per_class: list[float | None] = []
    for position in range(truth.shape[1]):
        if truth[:, position].sum() == 0:
            per_class.append(None)
        else:
            per_class.append(float(np.mean((scores[:, position] - truth[:, position]) ** 2)))
    defined = [value for value in per_class if value is not None]
    return (float(np.mean(defined)) if defined else None), per_class


def fuse(tree_probabilities: np.ndarray, deep_probabilities: np.ndarray, alpha) -> np.ndarray:
    """Weighted average placing ``alpha`` on the tree model and ``1 - alpha`` on the deep model."""

    tree = np.asarray(tree_probabilities, dtype=np.float64)
    deep = np.asarray(deep_probabilities, dtype=np.float64)
    if tree.shape != deep.shape or tree.ndim != 2:
        raise ValueError("Fusion inputs must be matching two-dimensional arrays.")
    weights = np.asarray(alpha, dtype=np.float64)
    if weights.ndim == 0:
        if not 0.0 <= float(weights) <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        combined = float(weights) * tree + (1.0 - float(weights)) * deep
    elif weights.shape == (tree.shape[1],):
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("Per-class alpha values must lie in [0, 1].")
        combined = weights[np.newaxis, :] * tree + (1.0 - weights[np.newaxis, :]) * deep
    else:
        raise ValueError("alpha must be a scalar or one value per target code.")
    return np.clip(combined, 0.0, 1.0)


def alpha_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("alpha_grid must lie in (0, 1].")
    count = int(round(1.0 / step))
    return np.round(np.linspace(0.0, 1.0, count + 1), 6)


def select_alpha_fold_local(
    truth: np.ndarray,
    folds: np.ndarray,
    tree_probabilities: np.ndarray,
    deep_probabilities: np.ndarray,
    *,
    per_class: bool = False,
    step: float = 0.05,
) -> dict[int, float | np.ndarray]:
    """Choose alpha without ever reading the labels of the fold it will be applied to.

    ``result[f]`` is selected using only the welds whose fold is not ``f``, and
    the criterion is the macro PR-AUC of the fused scores on those training
    folds.  A held-out weld therefore contributes to neither the selection nor
    any tie-break.

    Per-class selection is available for exploration but should not be used to
    support a claim: with ten positive tunnel welds spread over five folds the
    per-class criterion is estimated from a handful of labels.
    """

    from sklearn.metrics import average_precision_score

    truth = np.asarray(truth)
    folds = np.asarray(folds)
    tree_probabilities = np.asarray(tree_probabilities, dtype=np.float64)
    deep_probabilities = np.asarray(deep_probabilities, dtype=np.float64)
    if truth.shape != tree_probabilities.shape or truth.shape != deep_probabilities.shape:
        raise ValueError("truth and both probability arrays must share a shape.")
    grid = alpha_grid(step)
    selection: dict[int, float | np.ndarray] = {}
    for fold in sorted(set(int(value) for value in folds)):
        train = folds != fold
        if not train.any():
            raise ValueError(f"Fold {fold} has no training welds to select alpha from.")
        if per_class:
            chosen = np.full(truth.shape[1], 0.5, dtype=np.float64)
            for position in range(truth.shape[1]):
                positive = truth[train, position] == 1
                if not positive.any():
                    continue
                best_value, best_alpha = -np.inf, 0.5
                for candidate in grid:
                    combined = (
                        candidate * tree_probabilities[train, position]
                        + (1.0 - candidate) * deep_probabilities[train, position]
                    )
                    value = float(
                        average_precision_score(truth[train, position], combined)
                    )
                    if value > best_value:
                        best_value, best_alpha = value, float(candidate)
                chosen[position] = best_alpha
            selection[fold] = chosen
        else:
            best_value, best_alpha = -np.inf, 0.5
            for candidate in grid:
                combined = fuse(
                    tree_probabilities[train], deep_probabilities[train], float(candidate)
                )
                value = macro_average_precision(truth[train], combined)
                if value is not None and value > best_value:
                    best_value, best_alpha = value, float(candidate)
            selection[fold] = best_alpha
    return selection


def apply_fold_local_alpha(
    tree_probabilities: np.ndarray,
    deep_probabilities: np.ndarray,
    folds: np.ndarray,
    selection: dict[int, float | np.ndarray],
) -> np.ndarray:
    """Apply each fold's own alpha to that fold's weld rows."""

    combined = np.zeros_like(tree_probabilities, dtype=np.float64)
    for fold, alpha in selection.items():
        valid = folds == fold
        combined[valid] = fuse(tree_probabilities[valid], deep_probabilities[valid], alpha)
    return combined


def clustered_bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    groups: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 7,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap that resamples whole image groups, not single welds.

    Welds photographed in the same image are not independent, so resampling
    welds alone would understate the interval.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    labels = np.asarray(list(groups))
    unique = np.unique(labels)
    members = {group: np.flatnonzero(labels == group) for group in unique}
    generator = np.random.default_rng(seed)
    samples = []
    observed = statistic(np.arange(len(labels)))
    for _ in range(n_boot):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[group] for group in drawn])
        value = statistic(index)
        if value is not None and np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"observed": observed, "low": None, "high": None, "n_boot": 0}
    tail = (1.0 - confidence) / 2.0
    return {
        "observed": observed,
        "low": float(np.quantile(samples, tail)),
        "high": float(np.quantile(samples, 1.0 - tail)),
        "n_boot": len(samples),
    }


def paired_bootstrap_ci(
    truth: np.ndarray,
    fused_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    groups: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 7,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Clustered bootstrap interval for the pooled macro PR-AUC difference."""

    def difference(index: np.ndarray) -> float | None:
        reference = macro_average_precision(truth[index], reference_probabilities[index])
        candidate = macro_average_precision(truth[index], fused_probabilities[index])
        if reference is None or candidate is None:
            return None
        return candidate - reference

    return clustered_bootstrap_ci(
        difference, groups, n_boot=n_boot, seed=seed, confidence=confidence
    )


def evaluate_fusion(
    tree: OofFrames,
    deep: OofFrames,
    fused_probabilities: np.ndarray,
    *,
    alpha: dict[int, float | np.ndarray] | None = None,
    n_boot: int = 2000,
    seed: int = 7,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Report every candidate under both metric conventions plus paired uncertainty."""

    truth = tree.truth
    folds = tree.folds
    report: dict[str, object] = {
        "target_codes": list(tree.target_codes),
        "sample_count": int(len(truth)),
        "fold_count": int(len(set(int(value) for value in folds))),
        "positive_counts": {
            code: int(truth[:, position].sum())
            for position, code in enumerate(tree.target_codes)
        },
        "alpha": {
            str(fold): (float(value) if np.ndim(value) == 0 else [float(x) for x in value])
            for fold, value in (alpha or {}).items()
        },
        "candidates": {},
    }
    for name, scores in (
        ("tree", tree.probabilities),
        ("deep", deep.probabilities),
        ("fused", fused_probabilities),
    ):
        per_class = _per_class_average_precision(truth, scores)
        brier_macro, brier_per_class = brier_scores(truth, scores)
        report["candidates"][name] = {
            "pooled_macro_pr_auc": macro_average_precision(truth, scores),
            "fold_mean_macro_pr_auc": fold_mean_macro_average_precision(truth, scores, folds),
            "per_code_pr_auc": {
                code: per_class[position] for position, code in enumerate(tree.target_codes)
            },
            "brier_macro": brier_macro,
            "brier_per_code": {
                code: brier_per_class[position] for position, code in enumerate(tree.target_codes)
            },
        }
    report["fused_minus_tree"] = paired_bootstrap_ci(
        truth,
        fused_probabilities,
        tree.probabilities,
        tree.image_groups,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )
    report["fused_minus_deep"] = paired_bootstrap_ci(
        truth,
        fused_probabilities,
        deep.probabilities,
        tree.image_groups,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )
    return report
