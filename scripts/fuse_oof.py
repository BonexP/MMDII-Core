"""Fuse two archived weld-level OOF runs into one standard-shaped run directory.

The output directory is written in the same schema as a training run
(``oof_predictions.csv``, ``fold_metrics.json``, ``training_summary.json``,
``run_config.json`` and a ``.complete`` marker) so that
``validate_tracked_outputs.py`` and ``summarize_time_frequency_matrix.py`` pick
it up without modification.  Extra fusion diagnostics go to a separate
``fusion_details.json`` instead of reordering the standard keys.

The fusion weight for each fold is selected from the other folds only; see
``mmdii.fusion.late_fusion``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from mmdii.evaluation.multilabel import evaluate_multilabel
from mmdii.fusion.late_fusion import (
    align_oof,
    apply_fold_local_alpha,
    evaluate_fusion,
    fuse,
    load_oof,
    select_alpha_fold_local,
)
from mmdii.reporting import configure_logging, get_logger


_LOGGER = get_logger("mmdii.fuse_oof")

MODE = "late_fusion"


def _read_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    fusion = payload.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError(f"{path} has no [fusion] table.")
    return fusion


def _run_config_of(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "run_config.json"
    if not path.is_file():
        raise ValueError(f"Missing run_config.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _default_output_name(tree_run: Path, deep_run: Path) -> str:
    pattern = re.compile(r"^(seed-\d+)-(.+)$")

    def parts(name: str) -> tuple[str, str]:
        match = pattern.match(name)
        return (match.group(1), match.group(2)) if match else ("seed-unknown", name)

    seed_token, tree_tag = parts(tree_run.name)
    _, deep_tag = parts(deep_run.name)
    return f"{seed_token}-late-fusion-{tree_tag}-{deep_tag}"


def _macro_pr_auc(truth: np.ndarray, probabilities: np.ndarray, target_codes: Sequence[str]) -> float | None:
    value = evaluate_multilabel(truth, probabilities, target_codes=target_codes)["macro_pr_auc"]
    return None if value is None else float(value)


def _write_oof(path: Path, rows: list[dict[str, object]], target_codes: Sequence[str]) -> None:
    headers = ("sample_id", "weld_id", "image_group", "fold", "target_codes_json") + tuple(
        f"prob_{code}" for code in target_codes
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML file with a [fusion] table.")
    parser.add_argument("--tree-run", type=Path)
    parser.add_argument("--deep-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alpha-selection", choices=("fixed", "fold_local"))
    parser.add_argument("--alpha-grid", type=float)
    parser.add_argument("--per-class", action="store_true", default=None)
    parser.add_argument("--n-boot", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--fold", type=int, action="append", dest="run_folds")
    arguments = parser.parse_args(argv)
    configure_logging()

    settings: dict[str, Any] = {}
    if arguments.config is not None:
        settings.update(_read_config(arguments.config.resolve()))
    for name, value in (
        ("tree_run", arguments.tree_run),
        ("deep_run", arguments.deep_run),
        ("output_dir", arguments.output_dir),
        ("output_name", arguments.output_name),
        ("alpha", arguments.alpha),
        ("alpha_selection", arguments.alpha_selection),
        ("alpha_grid", arguments.alpha_grid),
        ("per_class", arguments.per_class),
        ("n_boot", arguments.n_boot),
        ("confidence", arguments.confidence),
        ("bootstrap_seed", arguments.bootstrap_seed),
    ):
        if value is not None:
            settings[name] = value

    if "tree_run" not in settings or "deep_run" not in settings:
        raise ValueError("Both tree_run and deep_run are required.")
    tree_run = Path(settings["tree_run"])
    deep_run = Path(settings["deep_run"])
    alpha_selection = str(settings.get("alpha_selection", "fold_local"))
    if alpha_selection not in {"fixed", "fold_local"}:
        raise ValueError("alpha_selection must be fixed or fold_local.")
    alpha_step = float(settings.get("alpha_grid", 0.05))
    per_class = bool(settings.get("per_class", False))
    fixed_alpha = float(settings.get("alpha", 0.5))
    n_boot = int(settings.get("n_boot", 2000))
    confidence = float(settings.get("confidence", 0.95))
    bootstrap_seed = int(settings.get("bootstrap_seed", 7))
    if alpha_selection == "fixed" and per_class:
        raise ValueError("per_class selection is only available with alpha_selection=fold_local.")

    tree_config = _run_config_of(tree_run)
    deep_config = _run_config_of(deep_run)
    target_codes = tuple(tree_config["target_codes"])
    if tuple(deep_config["target_codes"]) != target_codes:
        raise ValueError("Parent runs disagree on target_codes.")
    for key in ("release_directory", "fold_scheme", "seed"):
        if tree_config.get(key) != deep_config.get(key):
            raise ValueError(f"Parent runs disagree on {key}.")

    tree = load_oof(tree_run, target_codes)
    deep = load_oof(deep_run, target_codes)
    tree, deep = align_oof(tree, deep)

    folds = tuple(arguments.run_folds) if arguments.run_folds else tuple(
        sorted(set(int(value) for value in tree.folds))
    )
    selected = np.isin(tree.folds, folds)
    if not selected.any():
        raise ValueError("No welds fall in the requested folds.")
    truth = tree.truth[selected]
    fold_values = tree.folds[selected]
    tree_probabilities = tree.probabilities[selected]
    deep_probabilities = deep.probabilities[selected]
    sample_ids = np.asarray(tree.sample_ids)[selected]
    weld_ids = np.asarray(tree.weld_ids)[selected]
    image_groups = np.asarray(tree.image_groups)[selected]
    tree_selected = replace(
        tree,
        sample_ids=tuple(sample_ids),
        weld_ids=tuple(weld_ids),
        image_groups=tuple(image_groups),
        folds=fold_values,
        truth=truth,
        probabilities=tree_probabilities,
    )
    deep_selected = replace(tree_selected, probabilities=deep_probabilities)

    if alpha_selection == "fold_local":
        selection = select_alpha_fold_local(
            truth,
            fold_values,
            tree_probabilities,
            deep_probabilities,
            per_class=per_class,
            step=alpha_step,
        )
        fused = apply_fold_local_alpha(
            tree_probabilities, deep_probabilities, fold_values, selection
        )
    else:
        selection = {int(fold): fixed_alpha for fold in folds}
        fused = fuse(tree_probabilities, deep_probabilities, fixed_alpha)
    _LOGGER.info(
        "fuse start tree=%s deep=%s folds=%s selection=%s",
        tree_run.name,
        deep_run.name,
        ",".join(str(fold) for fold in folds),
        alpha_selection,
    )

    fold_reports: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for fold in folds:
        in_fold = fold_values == fold
        metrics = evaluate_multilabel(
            truth[in_fold], fused[in_fold], target_codes=target_codes
        )
        metrics["fold"] = int(fold)
        metrics["alpha"] = (
            float(selection[fold])
            if np.ndim(selection[fold]) == 0
            else [float(value) for value in selection[fold]]
        )
        metrics["tree_macro_pr_auc"] = _macro_pr_auc(
            truth[in_fold], tree_probabilities[in_fold], target_codes
        )
        metrics["deep_macro_pr_auc"] = _macro_pr_auc(
            truth[in_fold], deep_probabilities[in_fold], target_codes
        )
        fold_reports.append(metrics)
        for position in np.flatnonzero(in_fold):
            oof_rows.append(
                {
                    "sample_id": sample_ids[position],
                    "weld_id": weld_ids[position],
                    "image_group": image_groups[position],
                    "fold": int(fold_values[position]),
                    "target_codes_json": json.dumps(
                        [
                            code
                            for code, value in zip(target_codes, truth[position], strict=True)
                            if value
                        ],
                        separators=(",", ":"),
                    ),
                    **{
                        f"prob_{code}": float(fused[position][index])
                        for index, code in enumerate(target_codes)
                    },
                }
            )

    output_directory = settings.get("output_dir")
    if output_directory is None:
        name = str(settings.get("output_name") or _default_output_name(tree_run, deep_run))
        output_directory = tree_run.parent / name
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / ".complete").unlink(missing_ok=True)

    oof_rows.sort(key=lambda row: str(row["sample_id"]))
    _write_oof(output_directory / "oof_predictions.csv", oof_rows, target_codes)
    summary = {
        "mode": MODE,
        "aggregator": "probability_fusion",
        "fold_count": len(folds),
        "completed_fold_count": len(folds),
        "run_folds": [int(fold) for fold in folds],
        "fold_scheme": tree_config.get("fold_scheme"),
        "sample_count": len(oof_rows),
        "fold_metrics": fold_reports,
    }
    details = evaluate_fusion(
        tree_selected,
        deep_selected,
        fused,
        alpha=selection,
        n_boot=n_boot,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    details.update(
        {
            "alpha_selection": alpha_selection,
            "alpha_grid": alpha_step,
            "per_class_alpha": per_class,
            "tree_run": str(tree_run.resolve()),
            "deep_run": str(deep_run.resolve()),
            "folds": [int(fold) for fold in folds],
            "confidence": confidence,
        }
    )

    run_config = dict(tree_config)
    run_config["output_directory"] = output_directory.as_posix()
    run_config["mode"] = MODE
    run_config["fusion"] = {
        "alpha_selection": alpha_selection,
        "alpha": fixed_alpha,
        "alpha_grid": alpha_step,
        "per_class": per_class,
        "tree_run": str(tree_run.resolve()),
        "deep_run": str(deep_run.resolve()),
    }
    (output_directory / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "fold_metrics.json").write_text(
        json.dumps(fold_reports, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "fusion_details.json").write_text(
        json.dumps(details, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / ".complete").touch()
    _LOGGER.info("fuse complete output=%s", output_directory)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
