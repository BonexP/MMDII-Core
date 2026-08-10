"""Run one ModernTCN-MIL optimizer step on a real Dataset v0.2 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from mmdii.training.cross_validation import load_experiment_config
from mmdii.training.readiness import run_real_data_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    try:
        summary = run_real_data_smoke(
            config,
            release_directory=args.release_dir,
            fold=args.fold,
            batch_size=args.batch_size,
            device_override=args.device,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
