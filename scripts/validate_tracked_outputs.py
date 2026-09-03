"""Validate tracked MMDII experiment outputs without loading signal data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def validate_run(path: Path) -> dict[str, Any]:
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
    if isinstance(fold_metrics, list) and len(fold_metrics) != 5:
        errors.append(f"expected 5 fold reports, found {len(fold_metrics)}")
    return {
        "path": str(path.resolve()),
        "sample_count": len(rows),
        "expected_sample_count": expected_count,
        "probability_columns": probability_columns,
        "complete": not errors,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    runs = sorted(path.parent for path in root.rglob("training_summary.json"))
    results = [validate_run(path) for path in runs]
    payload = {
        "root": str(root),
        "run_count": len(results),
        "ok": bool(results) and all(item["complete"] for item in results),
        "runs": results,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(f"runs={payload['run_count']} ok={payload['ok']}")
        for item in results:
            state = "OK" if item["complete"] else "FAIL"
            print(f"{state} {item['path']} samples={item['sample_count']}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
