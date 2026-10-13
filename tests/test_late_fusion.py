from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.fusion.late_fusion import (  # noqa: E402
    OofFrames,
    align_oof,
    alpha_grid,
    apply_fold_local_alpha,
    clustered_bootstrap_ci,
    evaluate_fusion,
    fold_mean_macro_average_precision,
    fuse,
    load_oof,
    macro_average_precision,
    paired_bootstrap_ci,
    select_alpha_fold_local,
)

CODES = ("flash", "blur", "tunnel")
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def _synthetic(n: int = 60, seed: int = 0):
    generator = np.random.default_rng(seed)
    folds = np.repeat(np.arange(5), n // 5) if n % 5 == 0 else np.arange(n) % 5
    truth = np.zeros((n, 3), dtype=np.int64)
    truth[:, 0] = 1  # flash is saturated, like the real release
    truth[:, 1] = (generator.random(n) < 0.5).astype(np.int64)
    truth[:, 2] = (np.arange(n) % 6 == 0).astype(np.int64)
    tree = np.clip(0.55 * truth + generator.random((n, 3)) * 0.45, 0.0, 1.0)
    deep = np.clip(0.35 * truth + generator.random((n, 3)) * 0.65, 0.0, 1.0)
    return folds, truth, tree, deep


def _write_run(directory: Path, target_codes, folds, truth, probabilities) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "weld_id", "image_group", "fold", "target_codes_json"] + [
            f"prob_{code}" for code in target_codes
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(len(folds)):
            writer.writerow(
                {
                    "sample_id": f"s{index:03d}",
                    "weld_id": str(index),
                    "image_group": f"g{index % 7}",
                    "fold": int(folds[index]),
                    "target_codes_json": json.dumps(
                        [code for code, value in zip(target_codes, truth[index]) if value]
                    ),
                    **{
                        f"prob_{code}": float(probabilities[index][position])
                        for position, code in enumerate(target_codes)
                    },
                }
            )
    (directory / "run_config.json").write_text(
        json.dumps(
            {
                "target_codes": list(target_codes),
                "release_directory": "/release",
                "fold_scheme": "weld_independent",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    return directory


def _frame(folds, truth, probabilities) -> OofFrames:
    return OofFrames(
        run_directory=Path("."),
        target_codes=CODES,
        sample_ids=tuple(f"s{index:03d}" for index in range(len(folds))),
        weld_ids=tuple(str(index) for index in range(len(folds))),
        image_groups=tuple(f"g{index % 7}" for index in range(len(folds))),
        folds=np.asarray(folds),
        truth=np.asarray(truth),
        probabilities=np.asarray(probabilities),
    )


class AlphaGridTests(unittest.TestCase):
    def test_grid_spans_zero_to_one(self) -> None:
        grid = alpha_grid(0.05)
        self.assertEqual(len(grid), 21)
        self.assertEqual(float(grid[0]), 0.0)
        self.assertEqual(float(grid[-1]), 1.0)

    def test_rejects_bad_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha_grid"):
            alpha_grid(0.0)


class LateFusionTests(unittest.TestCase):
    def test_fuse_is_a_convex_combination(self) -> None:
        tree = np.array([[1.0, 0.0], [0.0, 1.0]])
        deep = np.array([[0.0, 1.0], [1.0, 0.0]])

        self.assertTrue(np.allclose(fuse(tree, deep, 0.5), 0.5))
        self.assertTrue(np.allclose(fuse(tree, deep, 1.0), tree))
        self.assertTrue(np.allclose(fuse(tree, deep, 0.0), deep))

    def test_fuse_accepts_one_alpha_per_code(self) -> None:
        tree = np.array([[1.0, 1.0]])
        deep = np.array([[0.0, 0.0]])

        combined = fuse(tree, deep, np.array([1.0, 0.0]))

        self.assertTrue(np.allclose(combined, [[1.0, 0.0]]))

    def test_fuse_rejects_out_of_range_alpha(self) -> None:
        tree = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "alpha"):
            fuse(tree, tree, 1.5)
        with self.assertRaisesRegex(ValueError, "alpha"):
            fuse(tree, tree, np.array([0.5, 1.5]))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_fold_alpha_ignores_the_labels_of_the_fold_it_serves(self) -> None:
        folds, truth, tree, deep = _synthetic()
        selection = select_alpha_fold_local(truth, folds, tree, deep)

        flipped = truth.copy()
        flipped[folds == 2] = 1 - flipped[folds == 2]
        after_flip = select_alpha_fold_local(flipped, folds, tree, deep)

        self.assertEqual(after_flip[2], selection[2])

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_fold_alpha_uses_only_the_other_folds(self) -> None:
        folds, truth, tree, deep = _synthetic()
        selection = select_alpha_fold_local(truth, folds, tree, deep)

        train = folds != 0
        expected = max(
            alpha_grid(0.05),
            key=lambda candidate: macro_average_precision(
                truth[train], fuse(tree[train], deep[train], float(candidate))
            ),
        )

        self.assertAlmostEqual(float(selection[0]), float(expected), places=6)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_apply_fold_local_alpha_matches_manual_fusion(self) -> None:
        folds, truth, tree, deep = _synthetic()
        selection = select_alpha_fold_local(truth, folds, tree, deep)

        combined = apply_fold_local_alpha(tree, deep, folds, selection)

        for fold, alpha in selection.items():
            valid = folds == fold
            np.testing.assert_allclose(combined[valid], fuse(tree[valid], deep[valid], alpha))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_zero_positive_code_is_reported_as_undefined_not_zero(self) -> None:
        folds, truth, tree, _ = _synthetic()
        empty = truth.copy()
        empty[:, 2] = 0

        per_class_report = evaluate_fusion(
            _frame(folds, empty, tree), _frame(folds, empty, tree), tree, n_boot=10
        )

        self.assertIsNone(per_class_report["candidates"]["tree"]["per_code_pr_auc"]["tunnel"])
        self.assertEqual(per_class_report["positive_counts"]["tunnel"], 0)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn train extra is not installed")
    def test_pooled_and_fold_mean_are_both_reported_and_may_differ(self) -> None:
        folds, truth, tree, deep = _synthetic()
        fused = fuse(tree, deep, 0.5)

        pooled = macro_average_precision(truth, fused)
        fold_mean = fold_mean_macro_average_precision(truth, fused, folds)

        self.assertIsNotNone(pooled)
        self.assertIsNotNone(fold_mean)
        self.assertNotAlmostEqual(pooled, fold_mean, places=6)

    def test_clustered_bootstrap_is_reproducible_and_clustered(self) -> None:
        groups = [f"g{index % 5}" for index in range(40)]
        values = np.linspace(0.0, 1.0, 40)

        first = clustered_bootstrap_ci(lambda index: float(values[index].mean()), groups, n_boot=100, seed=3)
        second = clustered_bootstrap_ci(lambda index: float(values[index].mean()), groups, n_boot=100, seed=3)

        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["observed"])
        self.assertGreaterEqual(first["high"], first["observed"])

    def test_paired_bootstrap_reports_zero_width_when_models_are_identical(self) -> None:
        folds, truth, tree, _ = _synthetic()
        groups = [f"g{index}" for index in range(len(truth))]

        result = paired_bootstrap_ci(truth, tree, tree, groups, n_boot=50, seed=1)

        self.assertEqual(result["observed"], 0.0)


class AlignmentTests(unittest.TestCase):
    def test_load_oof_rejects_missing_probability_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "run"
            folds, truth, tree, _ = _synthetic()
            _write_run(directory, CODES[:2], folds, truth, tree[:, :2])

            with self.assertRaisesRegex(ValueError, "lack columns"):
                load_oof(directory, CODES)

    def test_load_oof_rejects_out_of_range_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "run"
            folds, truth, tree, _ = _synthetic()
            broken = tree.copy()
            broken[0, 0] = 1.4
            _write_run(directory, CODES, folds, truth, broken)

            with self.assertRaisesRegex(ValueError, "outside"):
                load_oof(directory, CODES)

    def test_align_rejects_a_fold_disagreement(self) -> None:
        folds, truth, tree, deep = _synthetic()
        shifted = folds.copy()
        shifted[0] = (shifted[0] + 1) % 5

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = load_oof(_write_run(root / "a", CODES, folds, truth, tree), CODES)
            right = load_oof(_write_run(root / "b", CODES, shifted, truth, deep), CODES)

            with self.assertRaisesRegex(ValueError, "fold"):
                align_oof(left, right)

    def test_align_rejects_a_label_disagreement(self) -> None:
        folds, truth, tree, deep = _synthetic()
        altered = truth.copy()
        altered[0, 1] = 1 - altered[0, 1]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = load_oof(_write_run(root / "a", CODES, folds, truth, tree), CODES)
            right = load_oof(_write_run(root / "b", CODES, folds, altered, deep), CODES)

            with self.assertRaisesRegex(ValueError, "labels"):
                align_oof(left, right)

    def test_align_accepts_matching_frames(self) -> None:
        folds, truth, tree, deep = _synthetic()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = load_oof(_write_run(root / "a", CODES, folds, truth, tree), CODES)
            right = load_oof(_write_run(root / "b", CODES, folds, truth, deep), CODES)

            aligned_left, aligned_right = align_oof(left, right)

        np.testing.assert_array_equal(aligned_left.folds, aligned_right.folds)


if __name__ == "__main__":
    unittest.main()
