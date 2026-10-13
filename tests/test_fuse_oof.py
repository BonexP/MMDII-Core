from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(0, str(CORE_ROOT / "scripts"))

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_tracked_outputs", CORE_ROOT / "scripts" / "validate_tracked_outputs.py"
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)

import fuse_oof  # noqa: E402

CODES = ("flash", "blur", "tunnel")
WELD_COUNT = 25


def _write_parent(directory: Path, truth: np.ndarray, probabilities: np.ndarray, folds: np.ndarray) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_config.json").write_text(
        json.dumps(
            {
                "target_codes": list(CODES),
                "release_directory": "/release",
                "fold_scheme": "weld_independent",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    with (directory / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "weld_id", "image_group", "fold", "target_codes_json"] + [
            f"prob_{code}" for code in CODES
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(len(folds)):
            writer.writerow(
                {
                    "sample_id": f"s{index:03d}",
                    "weld_id": str(index),
                    "image_group": f"g{index % 6}",
                    "fold": int(folds[index]),
                    "target_codes_json": json.dumps(
                        [code for code, value in zip(CODES, truth[index]) if value]
                    ),
                    **{
                        f"prob_{code}": float(probabilities[index][position])
                        for position, code in enumerate(CODES)
                    },
                }
            )
    return directory


def _synthetic_parents(root: Path):
    generator = np.random.default_rng(11)
    folds = np.arange(WELD_COUNT) % 5
    truth = np.zeros((WELD_COUNT, 3), dtype=np.int64)
    truth[:, 0] = 1
    truth[:, 1] = (generator.random(WELD_COUNT) < 0.6).astype(np.int64)
    truth[:, 2] = (np.arange(WELD_COUNT) % 5 == 0).astype(np.int64)
    tree = np.clip(0.6 * truth + generator.random((WELD_COUNT, 3)) * 0.4, 0.0, 1.0)
    deep = np.clip(0.3 * truth + generator.random((WELD_COUNT, 3)) * 0.7, 0.0, 1.0)
    return (
        _write_parent(root / "seed-7-extra-trees", truth, tree, folds),
        _write_parent(root / "seed-7-e1c-gated-attention", truth, deep, folds),
        truth,
        tree,
        deep,
    )


class FuseOofScriptTests(unittest.TestCase):
    def tearDown(self) -> None:
        logger = logging.getLogger("mmdii")
        logger.handlers = [logging.NullHandler()]
        logger.setLevel(logging.NOTSET)
        logger.propagate = False

    def test_output_directory_passes_the_project_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, _, _, _ = _synthetic_parents(root)
            output = root / "seed-7-late-fusion-extra-trees-e1c-gated-attention"

            with contextlib.redirect_stdout(io.StringIO()):
                code = fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--output-dir",
                        str(output),
                        "--n-boot",
                        "20",
                    ]
                )

            self.assertEqual(code, 0)
            result = VALIDATOR.validate_run(
                output,
                expected_weld_count=WELD_COUNT,
                expected_fold_count=5,
                expected_fold_scheme="weld_independent",
            )
            self.assertTrue(result["complete"], result["errors"])
            self.assertEqual(result["sample_count"], WELD_COUNT)
            self.assertEqual(result["folds"], [0, 1, 2, 3, 4])

    def test_written_oof_holds_the_fused_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, truth, tree, deep = _synthetic_parents(root)
            output = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--output-dir",
                        str(output),
                        "--alpha-selection",
                        "fixed",
                        "--alpha",
                        "0.5",
                        "--n-boot",
                        "20",
                    ]
                )

            with (output / "oof_predictions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            by_sample = {row["sample_id"]: row for row in rows}
            for index in range(WELD_COUNT):
                row = by_sample[f"s{index:03d}"]
                for position, code in enumerate(CODES):
                    expected = 0.5 * tree[index][position] + 0.5 * deep[index][position]
                    self.assertAlmostEqual(float(row[f"prob_{code}"]), expected, places=10)

    def test_run_config_keeps_parent_release_scheme_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, _, _, _ = _synthetic_parents(root)
            output = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--output-dir",
                        str(output),
                        "--n-boot",
                        "20",
                    ]
                )

            run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["release_directory"], "/release")
            self.assertEqual(run_config["fold_scheme"], "weld_independent")
            self.assertEqual(run_config["seed"], 7)
            self.assertEqual(run_config["fusion"]["alpha_selection"], "fold_local")

    def test_fusion_details_are_written_outside_the_standard_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, _, _, _ = _synthetic_parents(root)
            output = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--output-dir",
                        str(output),
                        "--n-boot",
                        "20",
                    ]
                )

            details = json.loads((output / "fusion_details.json").read_text(encoding="utf-8"))
            self.assertIn("fused_minus_tree", details)
            self.assertIn("candidates", details)
            fold_metrics = json.loads((output / "fold_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(fold_metrics), 5)
            for report in fold_metrics:
                self.assertIn("macro_pr_auc", report)
                self.assertIn("per_code", report)
                self.assertIn("threshold", report)

    def test_rejects_parents_that_disagree_on_the_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, _, _, _ = _synthetic_parents(root)
            broken = json.loads((deep_run / "run_config.json").read_text(encoding="utf-8"))
            broken["release_directory"] = "/other-release"
            (deep_run / "run_config.json").write_text(json.dumps(broken), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "release_directory"):
                fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--output-dir",
                        str(root / "out"),
                        "--n-boot",
                        "20",
                    ]
                )

    def test_default_output_name_is_summariser_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tree_run, deep_run, _, _, _ = _synthetic_parents(root)

            with contextlib.redirect_stdout(io.StringIO()):
                fuse_oof.main(
                    [
                        "--tree-run",
                        str(tree_run),
                        "--deep-run",
                        str(deep_run),
                        "--n-boot",
                        "20",
                    ]
                )

            produced = root / "seed-7-late-fusion-extra-trees-e1c-gated-attention"
            self.assertTrue(produced.is_dir())
            self.assertTrue((produced / ".complete").is_file())


if __name__ == "__main__":
    unittest.main()
