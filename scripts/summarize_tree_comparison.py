"""Compare tree-fusion runs under both metric conventions with paired uncertainty.

Reports, for every discovered ``seed-<n>-<config>`` run:

* pooled out-of-fold macro PR-AUC and the fold-mean macro PR-AUC (the project
  uses both, and they can disagree, so neither is allowed to stand alone);
* per-class PR-AUC **with the positive count**, left null when a class has no
  positive weld rather than filled with a zero;
* macro and per-class Brier scores;
* the pooled macro PR-AUC difference against a reference run of the same seed,
  with a percentile bootstrap interval that resamples whole image groups.

A run is only called better than the reference when the mean gain across seeds
clears ``--min-effect`` (default 0.03, the measured rerun spread), the seeds
agree in direction, and the paired interval excludes zero.  Anything smaller is
reported as indistinguishable from noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any, Sequence

import numpy as np

from mmdii.fusion.late_fusion import (
    clustered_bootstrap_ci,
    fold_mean_macro_average_precision,
    load_oof,
    macro_average_precision,
)
from mmdii.reporting import configure_logging, get_logger


_LOGGER = get_logger("mmdii.tree_report")

RUN_NAME = re.compile(r"^seed-(?P<seed>\d+)-(?P<config>.+)$")
DEFAULT_ROOTS = (
    "outputs/time-frequency-v0-2-1",
    "outputs/time-frequency-gpu-focused-v0-2-1",
)


def _discover(roots: Sequence[Path]) -> dict[tuple[int, str], Path]:
    runs: dict[tuple[int, str], Path] = {}
    for root in roots:
        if not root.is_dir():
            _LOGGER.info("skip missing root %s", root)
            continue
        for entry in sorted(root.iterdir()):
            match = RUN_NAME.match(entry.name)
            if not match or not (entry / "oof_predictions.csv").is_file():
                continue
            runs[(int(match.group("seed")), match.group("config"))] = entry
    return runs


def _run_metrics(
    frame: Any,
    target_codes: Sequence[str],
    *,
    ranking_scores: Any | None = None,
    reference: Any | None,
    reference_ranking_scores: Any | None = None,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    from mmdii.fusion.late_fusion import brier_scores, paired_bootstrap_ci

    truth = frame.truth
    probabilities = frame.probabilities
    ranking = probabilities if ranking_scores is None else ranking_scores
    folds = frame.folds
    probability_metrics = _threshold_metrics(truth, probabilities, target_codes)
    brier_macro, brier_per_class = brier_scores(truth, probabilities)
    payload: dict[str, Any] = {
        "run": frame.run_directory.name,
        "pooled_macro_pr_auc": macro_average_precision(truth, ranking),
        "fold_mean_macro_pr_auc": fold_mean_macro_average_precision(truth, ranking, folds),
        "positive_counts": {
            code: int(truth[:, position].sum())
            for position, code in enumerate(target_codes)
        },
        "per_code_pr_auc": _per_code(truth, ranking, target_codes),
        **probability_metrics,
        "brier_macro": brier_macro,
        "brier_per_code": {
            code: brier_per_class[position] for position, code in enumerate(target_codes)
        },
    }
    if reference is not None:
        if (
            truth.shape != reference.truth.shape
            or not (truth == reference.truth).all()
            or not (folds == reference.folds).all()
        ):
            raise ValueError(
                f"{frame.run_directory.name} and {reference.run_directory.name} are not "
                "fold-aligned with identical labels."
            )
        payload["paired_vs_reference"] = {
            "reference": reference.run_directory.name,
            **paired_bootstrap_ci(
                truth,
                ranking,
                reference.probabilities if reference_ranking_scores is None else reference_ranking_scores,
                frame.image_groups,
                n_boot=n_boot,
                seed=bootstrap_seed,
            ),
        }
    return payload


def _threshold_metrics(
    truth: Any, probabilities: Any, target_codes: Sequence[str]
) -> dict[str, Any]:
    """Report fixed-0.5 F1/recall from probabilities, excluding undefined codes."""

    truth_values = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(probabilities, dtype=np.float64) >= 0.5
    per_code: dict[str, dict[str, float | None]] = {}
    valid = []
    for position in range(truth_values.shape[1]):
        labels = truth_values[:, position].astype(bool)
        positive_count = int(labels.sum())
        if positive_count == 0:
            per_code[target_codes[position]] = {"f1": None, "recall": None}
            continue
        selected = predicted[:, position]
        true_positive = int(np.logical_and(selected, labels).sum())
        false_positive = int(np.logical_and(selected, ~labels).sum())
        false_negative = positive_count - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        metrics = {
            "f1": 0.0 if denominator == 0 else 2 * true_positive / denominator,
            "recall": true_positive / positive_count,
        }
        per_code[target_codes[position]] = metrics
        valid.append(metrics)
    return {
        "threshold": 0.5,
        "per_code_threshold_metrics": per_code,
        "macro_f1": (
            float(np.mean([item["f1"] for item in valid])) if valid else None
        ),
        "macro_recall": (
            float(np.mean([item["recall"] for item in valid])) if valid else None
        ),
    }


def _load_ranking_scores(directory: Path, target_codes: Sequence[str]) -> Any | None:
    """Read optional score_* columns without changing the shared fusion frame."""

    path = directory / "oof_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    required = [f"score_{code}" for code in target_codes]
    present = [column for column in rows[0] if column.startswith("score_")]
    if not present:
        return None
    if set(present) != set(required):
        raise ValueError(f"score_* columns do not match target codes in {path}")
    rows.sort(key=lambda row: row["sample_id"])
    try:
        values = [[float(row[column]) for column in required] for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"OOF ranking scores are invalid: {path}") from error
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError(f"OOF ranking scores are not finite: {path}")
    return np.asarray(values, dtype=float)


def _per_code(truth, probabilities, target_codes) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score

    values: dict[str, float | None] = {}
    for position, code in enumerate(target_codes):
        if truth[:, position].sum() == 0:
            values[code] = None
        else:
            values[code] = float(
                average_precision_score(truth[:, position], probabilities[:, position])
            )
    return values


def _verdict(delta: float | None, interval_low: float | None, min_effect: float, direction_consistent: bool) -> str:
    if delta is None or interval_low is None:
        return "undefined"
    if delta < min_effect:
        return "indistinguishable_from_noise"
    if interval_low <= 0.0:
        return "interval_includes_zero"
    if not direction_consistent:
        return "seeds_disagree"
    return "accept"


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots")
    parser.add_argument(
        "--reference",
        help="Reference run directory name or config suffix, matched per seed.",
    )
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--target-codes", default="flash,blur,tunnel")
    parser.add_argument("--output", type=Path, help="Prefix for the .json and .csv outputs.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    arguments = parser.parse_args(argv)
    configure_logging()

    target_codes = tuple(part.strip() for part in arguments.target_codes.split(",") if part.strip())
    roots = tuple(arguments.roots) if arguments.roots else tuple(Path(root) for root in DEFAULT_ROOTS)
    runs = _discover(roots)
    if not runs:
        raise ValueError("No runs with oof_predictions.csv were found.")

    reference_by_seed: dict[int, Path] = {}
    if arguments.reference:
        for (seed, config), directory in runs.items():
            if directory.name == arguments.reference or config == arguments.reference:
                reference_by_seed[seed] = directory
        if not reference_by_seed:
            raise ValueError(f"No run matched --reference {arguments.reference!r}.")

    frames: dict[tuple[int, str], Any] = {}
    ranking_scores: dict[tuple[int, str], Any | None] = {}
    for key, directory in sorted(runs.items()):
        try:
            frames[key] = load_oof(directory, target_codes)
            ranking_scores[key] = _load_ranking_scores(directory, target_codes)
        except (ValueError, KeyError) as error:
            _LOGGER.info("skip %s: %s", directory.name, error)

    key_by_directory = {directory: key for key, directory in runs.items()}
    reports: dict[str, Any] = {}
    for (seed, config), frame in frames.items():
        reference = reference_by_seed.get(seed)
        reference_frame = None
        if reference is not None:
            if reference == runs[(seed, config)]:
                continue
            reference_frame = frames.get(key_by_directory[reference])
        try:
            reports.setdefault(config, {})[seed] = _run_metrics(
                frame,
                target_codes,
                ranking_scores=ranking_scores.get((seed, config)),
                reference=reference_frame,
                reference_ranking_scores=(
                    ranking_scores.get(key_by_directory[reference])
                    if reference is not None
                    else None
                ),
                n_boot=arguments.n_boot,
                bootstrap_seed=arguments.bootstrap_seed,
            )
        except (ValueError, KeyError) as error:
            _LOGGER.info("skip %s: %s", frame.run_directory.name, error)

    summary: list[dict[str, Any]] = []
    for config, per_seed in sorted(reports.items()):
        seeds = sorted(per_seed)
        pooled = [per_seed[seed]["pooled_macro_pr_auc"] for seed in seeds]
        fold_mean = [per_seed[seed]["fold_mean_macro_pr_auc"] for seed in seeds]
        deltas = [
            per_seed[seed]["paired_vs_reference"]["observed"]
            for seed in seeds
            if "paired_vs_reference" in per_seed[seed]
        ]
        lows = [
            per_seed[seed]["paired_vs_reference"]["low"]
            for seed in seeds
            if "paired_vs_reference" in per_seed[seed]
        ]
        mean_delta = mean(deltas) if deltas else None
        direction_consistent = bool(deltas) and all(delta > 0 for delta in deltas)
        summary.append(
            {
                "config": config,
                "seeds": seeds,
                "seed_count": len(seeds),
                "pooled_macro_pr_auc_mean": mean(pooled),
                "pooled_macro_pr_auc_sd": stdev(pooled) if len(pooled) > 1 else 0.0,
                "fold_mean_macro_pr_auc_mean": mean(fold_mean),
                "fold_mean_macro_pr_auc_sd": stdev(fold_mean) if len(fold_mean) > 1 else 0.0,
                "macro_f1_mean": _mean_defined(
                    [per_seed[seed]["macro_f1"] for seed in seeds]
                ),
                "macro_recall_mean": _mean_defined(
                    [per_seed[seed]["macro_recall"] for seed in seeds]
                ),
                "brier_macro_mean": _mean_defined(
                    [per_seed[seed]["brier_macro"] for seed in seeds]
                ),
                "mean_delta_vs_reference": mean_delta,
                "min_interval_low": min(lows) if lows else None,
                "direction_consistent": direction_consistent,
                "verdict": _verdict(
                    mean_delta,
                    min(lows) if lows else None,
                    arguments.min_effect,
                    direction_consistent,
                ),
                "per_seed": per_seed,
            }
        )

    print(
        f'{"config":44s} {"seeds":>5s} {"pooledAP":>17s} {"foldmeanAP":>17s} '
        f'{"delta_vs_ref":>13s} {"verdict":>28s}'
    )
    for row in summary:
        pooled = (
            f'{row["pooled_macro_pr_auc_mean"]:.4f}'
            f'±{row["pooled_macro_pr_auc_sd"]:.4f}'
        )
        fold_mean = (
            f'{row["fold_mean_macro_pr_auc_mean"]:.4f}'
            f'±{row["fold_mean_macro_pr_auc_sd"]:.4f}'
        )
        delta = (
            "n/a"
            if row["mean_delta_vs_reference"] is None
            else f'{row["mean_delta_vs_reference"]:+.4f}'
        )
        print(
            f'{row["config"]:44s} {row["seed_count"]:5d} {pooled:>17s} '
            f'{fold_mean:>17s} {delta:>13s} {row["verdict"]:>28s}'
        )

    first = summary[0]["per_seed"][summary[0]["seeds"][0]] if summary else {}
    if first:
        print(
            "\npositive counts: "
            + ", ".join(f"{code}={count}" for code, count in first["positive_counts"].items())
        )
        print(f'min effect threshold: {arguments.min_effect:.3f} (measured rerun spread 0.0315)')

    payload = {
        "roots": [str(root) for root in roots],
        "reference": arguments.reference,
        "min_effect": arguments.min_effect,
        "n_boot": arguments.n_boot,
        "bootstrap_seed": arguments.bootstrap_seed,
        "summary": summary,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with arguments.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "config",
                    "seed_count",
                    "pooled_macro_pr_auc_mean",
                    "pooled_macro_pr_auc_sd",
                    "fold_mean_macro_pr_auc_mean",
                    "fold_mean_macro_pr_auc_sd",
                    "macro_f1_mean",
                    "macro_recall_mean",
                    "brier_macro_mean",
                    "mean_delta_vs_reference",
                    "min_interval_low",
                    "verdict",
                ]
            )
            for row in summary:
                writer.writerow(
                    [
                        row["config"],
                        row["seed_count"],
                        f'{row["pooled_macro_pr_auc_mean"]:.10f}',
                        f'{row["pooled_macro_pr_auc_sd"]:.10f}',
                        f'{row["fold_mean_macro_pr_auc_mean"]:.10f}',
                        f'{row["fold_mean_macro_pr_auc_sd"]:.10f}',
                        ""
                        if row["macro_f1_mean"] is None
                        else f'{row["macro_f1_mean"]:.10f}',
                        ""
                        if row["macro_recall_mean"] is None
                        else f'{row["macro_recall_mean"]:.10f}',
                        ""
                        if row["brier_macro_mean"] is None
                        else f'{row["brier_macro_mean"]:.10f}',
                        ""
                        if row["mean_delta_vs_reference"] is None
                        else f'{row["mean_delta_vs_reference"]:.10f}',
                        "" if row["min_interval_low"] is None else f'{row["min_interval_low"]:.10f}',
                        row["verdict"],
                    ]
                )
    if arguments.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
