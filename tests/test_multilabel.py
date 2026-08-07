from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.evaluation.multilabel import (
    compute_positive_class_weights,
    evaluate_multilabel,
)


class MultilabelEvaluationTests(unittest.TestCase):
    def test_positive_weights_use_only_training_targets(self) -> None:
        targets = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )

        weights = compute_positive_class_weights(targets)

        np.testing.assert_allclose(weights, [0.5, 2.0])

    def test_rejects_training_target_without_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "one class"):
            compute_positive_class_weights(np.zeros((3, 1), dtype=np.float64))

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None,
        "scikit-learn train extra is not installed",
    )
    def test_reports_per_code_metrics_and_undefined_positive_code(self) -> None:
        truth = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        probabilities = np.array(
            [[0.1, 0.2], [0.9, 0.4], [0.8, 0.3]],
            dtype=np.float64,
        )

        report = evaluate_multilabel(
            truth,
            probabilities,
            target_codes=("flash", "pore"),
            threshold=0.5,
        )

        self.assertGreater(report["per_code"]["flash"]["pr_auc"], 0.9)
        self.assertEqual(report["per_code"]["flash"]["recall"], 1.0)
        self.assertIsNone(report["per_code"]["pore"]["pr_auc"])
        self.assertEqual(report["per_code"]["pore"]["valid_positive_count"], 0)
        self.assertEqual(report["valid_code_count"], 1)


if __name__ == "__main__":
    unittest.main()
