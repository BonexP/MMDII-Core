"""Weld-level multi-label losses and evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_positive_class_weights(targets: np.ndarray) -> np.ndarray:
    """Return negative/positive ratios for one training fold."""

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("targets must be a non-empty two-dimensional array.")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("targets must contain only zero and one.")
    positives = values.sum(axis=0)
    negatives = values.shape[0] - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("Training target contains only one class.")
    return negatives / positives


def weighted_bce_loss(
    logits: object, targets: object, positive_class_weights: np.ndarray
) -> object:
    """Create a PyTorch weighted BCE loss without making torch a data dependency."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Weighted BCE training requires the MMDII-Core train extra."
        ) from error
    device = getattr(logits, "device", None)
    weights = torch.as_tensor(
        positive_class_weights, dtype=torch.float32, device=device
    )
    return nn.BCEWithLogitsLoss(pos_weight=weights)(logits, targets)


def evaluate_multilabel(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_codes: Iterable[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Return per-code PR-AUC, recall, F1 and explicit undefined-code fields."""

    labels = tuple(target_codes)
    values = np.asarray(truth, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 2
        or scores.shape != values.shape
        or values.shape[1] != len(labels)
        or not labels
        or len(labels) != len(set(labels))
        or not 0.0 <= threshold <= 1.0
        or not np.isfinite(values).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Metric inputs have invalid shapes or values.")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("truth must contain only zero and one.")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("probabilities must be between zero and one.")
    try:
        from sklearn.metrics import average_precision_score, f1_score, recall_score
    except ImportError as error:
        raise RuntimeError(
            "Metric evaluation requires the MMDII-Core train extra."
        ) from error

    per_code: dict[str, dict[str, float | int | None]] = {}
    valid = []
    for position, code in enumerate(labels):
        binary_truth = values[:, position].astype(np.int64)
        positive_count = int(binary_truth.sum())
        if positive_count == 0:
            per_code[code] = {
                "pr_auc": None,
                "recall": None,
                "f1": None,
                "valid_positive_count": 0,
            }
            continue
        predicted = (scores[:, position] >= threshold).astype(np.int64)
        metrics = {
            "pr_auc": float(average_precision_score(binary_truth, scores[:, position])),
            "recall": float(recall_score(binary_truth, predicted, zero_division=0)),
            "f1": float(f1_score(binary_truth, predicted, zero_division=0)),
            "valid_positive_count": positive_count,
        }
        per_code[code] = metrics
        valid.append(metrics)
    return {
        "per_code": per_code,
        "valid_code_count": len(valid),
        "macro_pr_auc": (
            float(np.mean([metric["pr_auc"] for metric in valid])) if valid else None
        ),
        "macro_recall": (
            float(np.mean([metric["recall"] for metric in valid])) if valid else None
        ),
        "macro_f1": (
            float(np.mean([metric["f1"] for metric in valid])) if valid else None
        ),
        "threshold": float(threshold),
    }
