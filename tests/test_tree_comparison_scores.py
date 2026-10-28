from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "scripts"))
sys.path.insert(0, str(CORE_ROOT / "src"))

import summarize_tree_comparison as summary  # noqa: E402


class TreeComparisonScoreTests(unittest.TestCase):
    def test_f1_and_recall_use_probabilities_and_fixed_threshold(self) -> None:
        metrics = summary._threshold_metrics(
            np.asarray([[0, 0], [1, 0], [1, 0]]),
            np.asarray([[0.4, 0.9], [0.8, 0.8], [0.3, 0.2]]),
            ("flash", "tunnel"),
        )

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["per_code_threshold_metrics"]["flash"]["recall"], 0.5)
        self.assertEqual(metrics["per_code_threshold_metrics"]["flash"]["f1"], 2 / 3)
        self.assertIsNone(metrics["per_code_threshold_metrics"]["tunnel"]["f1"])
        self.assertEqual(metrics["macro_recall"], 0.5)

    def test_reads_ranking_scores_in_sample_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            rows = [
                {"sample_id": "b", "score_flash": "2", "score_blur": "-1"},
                {"sample_id": "a", "score_flash": "-2", "score_blur": "1"},
            ]
            with (run / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            values = summary._load_ranking_scores(run, ("flash", "blur"))

        np.testing.assert_array_equal(values, [[-2.0, 1.0], [2.0, -1.0]])

    def test_legacy_oof_has_no_ranking_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            with (run / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "prob_flash"))
                writer.writeheader()
                writer.writerow({"sample_id": "a", "prob_flash": "0.5"})

            values = summary._load_ranking_scores(run, ("flash",))

        self.assertIsNone(values)

    def test_ap_uses_scores_while_brier_uses_probabilities(self) -> None:
        probabilities = np.array([[0.4], [0.9]])
        ranking_scores = np.array([[-3.0], [4.0]])
        frame = SimpleNamespace(
            truth=np.array([[0], [1]]),
            probabilities=probabilities,
            folds=np.array([0, 1]),
            run_directory=Path("run"),
            image_groups=("g1", "g2"),
        )
        seen = {}

        def capture_ap(_truth, scores):
            seen["ap_scores"] = scores
            return 0.75

        def capture_brier(_truth, scores):
            seen["brier_probabilities"] = scores
            return 0.2, [0.2]

        with patch.object(summary, "macro_average_precision", side_effect=capture_ap):
            with patch.object(
                summary, "fold_mean_macro_average_precision", return_value=0.75
            ):
                with patch.object(summary, "_per_code", return_value={"flash": 0.75}):
                    with patch(
                        "mmdii.fusion.late_fusion.brier_scores",
                        side_effect=capture_brier,
                    ):
                        report = summary._run_metrics(
                            frame,
                            ("flash",),
                            ranking_scores=ranking_scores,
                            reference=None,
                            n_boot=10,
                            bootstrap_seed=7,
                        )

        self.assertIs(report["pooled_macro_pr_auc"], 0.75)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["macro_recall"], 1.0)
        self.assertEqual(report["threshold"], 0.5)
        np.testing.assert_array_equal(seen["ap_scores"], ranking_scores)
        np.testing.assert_array_equal(seen["brier_probabilities"], probabilities)


if __name__ == "__main__":
    unittest.main()
