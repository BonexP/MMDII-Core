"""Check whether a host can train MMDII-Core on a real Dataset v0.2 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mmdii.training.cross_validation import load_experiment_config
from mmdii.training.readiness import inspect_environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    report = inspect_environment(
        config,
        release_directory=args.release_dir,
        output_directory=args.output_dir,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


def _print_report(report: dict[str, object]) -> None:
    python = report["python"]
    packages = report["packages"]
    accelerator = report["accelerator"]
    dataset = report["dataset"]
    output = report["output"]
    print(f"status: {'READY' if report['ok'] else 'NOT READY'}")
    print(f"python: {python['version']} ({'ok' if python['supported'] else 'unsupported'})")
    for name, package in packages.items():
        value = package["version"] if package["available"] else "missing"
        print(f"{name}: {value}")
    print(
        f"cuda: {'available' if accelerator['cuda_available'] else 'not available'} "
        f"({accelerator['device_count']} device(s))"
    )
    print(
        f"dataset: {'valid' if dataset['valid'] else 'invalid'} "
        f"({dataset['sample_count']} samples)"
    )
    print(f"output: {'writable' if output['writable'] else 'not writable'}")
    for error in report["errors"]:
        print(f"error: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
