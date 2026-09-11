"""Validate tracked MMDII experiment outputs without loading signal data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def validate_run(
    path: Path,
    *,
    allow_partial: bool = False,
    expected_weld_count: int | None = None,
    expected_fold_count: int | None = None,
    expected_fold_scheme: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    payloads: dict[str, Any] = {}
    for name in ("training_summary", "run_config", "fold_metrics"):
        file_path = path / f"{name}.json"
        try:
            payloads[name] = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid {name}.json: {error}")
    if not (path / ".complete").exists():
        errors.append("missing .complete marker")

    rows: list[dict[str, str]] = []
    try:
        with (path / "oof_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        errors.append(f"invalid oof_predictions.csv: {error}")

    sample_ids = [row.get("sample_id", "") for row in rows]
    if not all(sample_ids):
        errors.append("OOF rows contain an empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("OOF sample_id values are not unique")
    weld_ids = [row.get("weld_id", "") for row in rows]
    unique_weld_ids = {weld_id for weld_id in weld_ids if weld_id}
    if expected_weld_count is not None and len(unique_weld_ids) != expected_weld_count:
        errors.append(
            f"expected {expected_weld_count} unique welds, found {len(unique_weld_ids)}"
        )
    fold_values: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            fold_values.add(int(row.get("fold", "")))
        except (TypeError, ValueError):
            if expected_fold_count is not None:
                errors.append(f"row {row_number} has invalid fold")
    probability_columns = sorted(column for column in (rows[0] if rows else {}) if column.startswith("prob_"))
    if not probability_columns:
        errors.append("OOF has no prob_* columns")
    for row_number, row in enumerate(rows, start=2):
        for column in probability_columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                errors.append(f"row {row_number} has invalid {column}")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"row {row_number} has out-of-range {column}")

    expected_count = None
    summary = payloads.get("training_summary")
    if isinstance(summary, dict):
        expected_count = summary.get("sample_count")
        if expected_count != len(rows):
            errors.append(f"summary sample_count={expected_count} but OOF has {len(rows)} rows")
    fold_metrics = payloads.get("fold_metrics")
    required_folds = expected_fold_count if expected_fold_count is not None else (None if allow_partial else 5)
    if isinstance(fold_metrics, list) and required_folds is not None:
        if len(fold_metrics) != required_folds:
            errors.append(f"expected {required_folds} fold reports, found {len(fold_metrics)}")
        if expected_fold_count is not None and rows and not fold_values:
            errors.append("OOF has no parseable fold values")
        if fold_values and fold_values != set(range(required_folds)) and not allow_partial:
            errors.append(f"expected OOF folds 0..{required_folds - 1}, found {sorted(fold_values)}")
    run_config = payloads.get("run_config")
    seed = release_directory = fold_scheme = None
    if isinstance(run_config, dict):
        seed = run_config.get("seed")
        release_directory = run_config.get("release_directory")
        fold_scheme = run_config.get("fold_scheme")
        if expected_fold_scheme is not None and fold_scheme != expected_fold_scheme:
            errors.append(
                f"expected fold_scheme={expected_fold_scheme}, found {fold_scheme}"
            )
    train_log = path / "train.log"
    if train_log.exists():
        try:
            if "traceback (most recent call last)" in train_log.read_text(
                encoding="utf-8", errors="replace"
            ).lower():
                errors.append("train.log contains a Python traceback")
        except OSError as error:
            errors.append(f"cannot read train.log: {error}")
    for error_log in (*path.glob("*.err"), *path.glob("error*.log")):
        try:
            if error_log.stat().st_size:
                errors.append(f"non-empty error log: {error_log.name}")
        except OSError as error:
            errors.append(f"cannot inspect {error_log.name}: {error}")
    return {
        "path": str(path.resolve()),
        "sample_count": len(rows),
        "weld_count": len(unique_weld_ids),
        "folds": sorted(fold_values),
        "expected_sample_count": expected_count,
        "probability_columns": probability_columns,
        "seed": seed,
        "release_directory": release_directory,
        "fold_scheme": fold_scheme,
        "complete": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-weld-count", type=int)
    parser.add_argument("--expected-fold-count", type=int)
    parser.add_argument("--expected-run-count", type=int)
    parser.add_argument("--expected-configurations-per-seed", type=int)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--expected-fold-scheme")
    args = parser.parse_args()
    root = args.root.resolve()
    runs = sorted(path.parent for path in root.rglob("training_summary.json"))
    results = [
        validate_run(
            path,
            allow_partial=args.allow_partial,
            expected_weld_count=args.expected_weld_count,
            expected_fold_count=args.expected_fold_count,
            expected_fold_scheme=args.expected_fold_scheme,
        )
        for path in runs
    ]
    suite_errors: list[str] = []
    if args.expected_run_count is not None and len(results) != args.expected_run_count:
        suite_errors.append(f"expected {args.expected_run_count} runs, found {len(results)}")
    releases = {item["release_directory"] for item in results if item["release_directory"]}
    if len(releases) > 1:
        suite_errors.append(f"runs use different releases: {sorted(releases)}")
    schemes = {item["fold_scheme"] for item in results if item["fold_scheme"]}
    if len(schemes) > 1:
        suite_errors.append(f"runs use different fold schemes: {sorted(schemes)}")
    observed_seeds = {int(item["seed"]) for item in results if item["seed"] is not None}
    if args.expected_seeds is not None and observed_seeds != set(args.expected_seeds):
        suite_errors.append(
            f"expected seeds {sorted(args.expected_seeds)}, found {sorted(observed_seeds)}"
        )
    if args.expected_configurations_per_seed is not None:
        for seed in sorted(observed_seeds):
            count = sum(item["seed"] == seed for item in results)
            if count != args.expected_configurations_per_seed:
                suite_errors.append(
                    f"seed {seed}: expected {args.expected_configurations_per_seed} runs, found {count}"
                )
    payload = {
        "root": str(root),
        "run_count": len(results),
        "ok": bool(results) and not suite_errors and all(item["complete"] for item in results),
        "errors": suite_errors,
        "runs": results,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(f"runs={payload['run_count']} ok={payload['ok']}")
        for error in suite_errors:
            print(f"FAIL suite: {error}")
        for item in results:
            state = "OK" if item["complete"] else "FAIL"
            print(f"{state} {item['path']} samples={item['sample_count']}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
