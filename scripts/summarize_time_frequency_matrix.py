"""Summarize time-frequency runs across folds and aligned seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any

METRICS = ("macro_pr_auc", "macro_f1", "macro_recall")
CODES = ("flash", "blur", "tunnel")
SEED_NAME = re.compile(r"^seed-(?P<seed>\d+)-(?P<configuration>.+)$")


def _mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _run_metric(summary: dict[str, Any]) -> dict[str, float]:
    folds = summary.get("fold_metrics", [])
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for code in CODES:
        for metric in ("f1", "pr_auc", "recall"):
            values[f"{code}_{metric}"] = []
    for fold in folds:
        for metric in METRICS:
            values[metric].append(float(fold[metric]))
        for code, report in fold.get("per_code", {}).items():
            if code in CODES:
                for metric in ("f1", "pr_auc", "recall"):
                    value = report.get(metric)
                    if value is not None:
                        values[f"{code}_{metric}"].append(float(value))
    return {name: _mean(items) for name, items in values.items()}


def _runs(root: Path) -> dict[tuple[str, int], dict[str, float]]:
    runs: dict[tuple[str, int], dict[str, float]] = {}
    for summary_path in sorted(root.rglob("training_summary.json")):
        match = SEED_NAME.match(summary_path.parent.name)
        if not match:
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = _run_metric(payload)
        if all(metrics[metric] == metrics[metric] for metric in METRICS):
            runs[(match.group("configuration"), int(match.group("seed")))] = metrics
    return runs


def _reference(
    runs: dict[tuple[str, int], dict[str, float]],
    seed: int,
    names: tuple[str, ...],
) -> dict[str, float] | None:
    for name in names:
        if (name, seed) in runs:
            return runs[(name, seed)]
    return None


def _best_baseline_name(
    runs: dict[tuple[str, int], dict[str, float]],
) -> str | None:
    candidates = ("random-forest", "b0-statistical")
    scores = {
        candidate: _mean(
            [
                metrics["macro_pr_auc"]
                for (configuration, _), metrics in runs.items()
                if configuration == candidate
            ]
        )
        for candidate in candidates
    }
    available = {name: score for name, score in scores.items() if score == score}
    return max(available, key=available.get) if available else None


def _build_rows(runs: dict[tuple[str, int], dict[str, float]]) -> list[dict[str, Any]]:
    configurations = sorted({configuration for configuration, _ in runs})
    metric_names = list(METRICS)
    metric_names.extend(
        f"{code}_{metric}" for code in CODES for metric in ("f1", "pr_auc", "recall")
    )
    rows: list[dict[str, Any]] = []
    best_baseline = _best_baseline_name(runs)
    for configuration in configurations:
        seeds = sorted(seed for name, seed in runs if name == configuration)
        row: dict[str, Any] = {
            "configuration": configuration,
            "seed_count": len(seeds),
            "best_baseline_reference": best_baseline,
        }
        for metric in metric_names:
            values = [runs[(configuration, seed)][metric] for seed in seeds]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
        for reference_name, reference_names in (
            ("raw_e1c", ("e1c-gated-attention",)),
            ("best_baseline", () if best_baseline is None else (best_baseline,)),
        ):
            for metric in METRICS:
                deltas = []
                for seed in seeds:
                    reference = _reference(runs, seed, reference_names)
                    if reference is not None:
                        deltas.append(runs[(configuration, seed)][metric] - reference[metric])
                row[f"delta_{reference_name}_{metric}_mean"] = _mean(deltas)
                row[f"delta_{reference_name}_{metric}_std"] = _std(deltas)
        rows.append(row)
    return rows


def _seed_rows(runs: dict[tuple[str, int], dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {"configuration": configuration, "seed": seed, **metrics}
        for (configuration, seed), metrics in sorted(runs.items())
    ]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = _runs(args.root.resolve())
    rows = _build_rows(runs)
    seed_rows = _seed_rows(runs)
    output = (args.output or args.root / "time_frequency_summary.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(METRICS)
    metric_names.extend(
        f"{code}_{metric}" for code in CODES for metric in ("f1", "pr_auc", "recall")
    )
    fields = ["configuration", "seed_count", "best_baseline_reference"] + [
        suffix for metric in metric_names for suffix in (f"{metric}_mean", f"{metric}_std")
    ] + [
        suffix
        for reference in ("raw_e1c", "best_baseline")
        for metric in METRICS
        for suffix in (f"delta_{reference}_{metric}_mean", f"delta_{reference}_{metric}_std")
    ]
    _write_csv(output, fields, rows)
    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seed_output = output.with_name(f"{output.stem}_by_seed.csv")
    seed_fields = ["configuration", "seed", *metric_names]
    _write_csv(seed_output, seed_fields, seed_rows)
    seed_json_output = seed_output.with_suffix(".json")
    seed_json_output.write_text(
        json.dumps(seed_rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output}, {json_output}, {seed_output}, and {seed_json_output} "
        f"({len(rows)} configurations, {len(runs)} seed runs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
