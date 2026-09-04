"""Calibrate per-class thresholds from non-held-out OOF folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("OOF file is empty.")
    probability_columns = tuple(sorted(column for column in rows[0] if column.startswith("prob_")))
    if not probability_columns:
        raise ValueError("OOF file has no prob_* columns.")
    return rows, tuple(column.removeprefix("prob_") for column in probability_columns)


def _truth(rows: list[dict[str, str]], codes: tuple[str, ...]) -> np.ndarray:
    values = np.zeros((len(rows), len(codes)), dtype=np.int64)
    for row_index, row in enumerate(rows):
        defects = json.loads(row["target_codes_json"])
        values[row_index] = [int(code in defects) for code in codes]
    return values


def _best_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    if truth.sum() == 0:
        return 0.5
    candidates = np.unique(np.concatenate(([0.5, 0.0, 1.0], scores)))
    best = (0.5, -1.0)
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int(np.logical_and(predicted, truth == 1).sum())
        false_positive = int(np.logical_and(predicted, truth == 0).sum())
        false_negative = int(np.logical_and(~predicted, truth == 1).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        if f1 > best[1] or (f1 == best[1] and abs(threshold - 0.5) < abs(best[0] - 0.5)):
            best = (float(threshold), f1)
    return best[0]


def _average_precision(truth: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(truth.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_truth = truth[order].astype(np.int64)
    precision_at_rank = np.cumsum(ordered_truth) / np.arange(1, len(truth) + 1)
    return float((precision_at_rank * ordered_truth).sum() / positive_count)


def calibrate(path: Path) -> dict[str, Any]:
    rows, codes = _load(path)
    truth = _truth(rows, codes)
    scores = np.asarray([[float(row[f"prob_{code}"]) for code in codes] for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    reports = []
    pooled_truth = []
    pooled_pred = []
    pooled_scores = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        thresholds = [_best_threshold(truth[train, position], scores[train, position]) for position in range(len(codes))]
        predictions = scores[valid] >= np.asarray(thresholds)
        reports.append({"fold": int(fold), "thresholds": dict(zip(codes, thresholds, strict=True)), "valid_count": int(valid.sum())})
        pooled_truth.append(truth[valid])
        pooled_pred.append(predictions)
        pooled_scores.append(scores[valid])
    truth_all = np.concatenate(pooled_truth)
    pred_all = np.concatenate(pooled_pred)
    score_all = np.concatenate(pooled_scores)
    per_code = {}
    for position, code in enumerate(codes):
        positive = truth_all[:, position] == 1
        tp = int(np.logical_and(pred_all[:, position], positive).sum())
        fp = int(np.logical_and(pred_all[:, position], ~positive).sum())
        fn = int(np.logical_and(~pred_all[:, position], positive).sum())
        per_code[code] = {
            "positive_count": int(positive.sum()),
            "pr_auc": _average_precision(positive, score_all[:, position]),
            "recall": None if positive.sum() == 0 else tp / (tp + fn),
            "f1": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
        }
    return {"oof_path": str(path.resolve()), "fold_count": len(reports), "sample_count": len(rows), "target_codes": list(codes), "folds": reports, "pooled_per_code": per_code}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate(args.oof.resolve())
    payload = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
