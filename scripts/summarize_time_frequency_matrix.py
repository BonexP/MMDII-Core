"""Summarize completed time-frequency runs by configuration and seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ("macro_pr_auc", "macro_f1", "macro_recall")


def _runs(root: Path):
    for summary_path in sorted(root.rglob("training_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        yield summary_path.parent.name, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, float]]] = {}
    for name, payload in _runs(args.root.resolve()):
        metrics = payload.get("fold_metrics", [])
        if not metrics:
            continue
        parts = name.split("-", 2)
        configuration = parts[2] if len(parts) == 3 and parts[0] == "seed" else name
        grouped.setdefault(configuration, []).append(
            {metric: mean(float(item[metric]) for item in metrics) for metric in METRICS}
        )
    output = (args.output or args.root / "time_frequency_summary.csv").resolve()
    fields = ["configuration", "seed_count"] + [
        suffix
        for metric in METRICS
        for suffix in (f"{metric}_mean", f"{metric}_std")
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, entries in sorted(grouped.items()):
            row: dict[str, object] = {"configuration": name, "seed_count": len(entries)}
            for metric in METRICS:
                values = [entry[metric] for entry in entries]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            writer.writerow(row)
    print(f"wrote {output} ({len(grouped)} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
