from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.training.tree_tuning import tune_and_predict_forest_ovr  # noqa: E402


class TreeTuningTests(unittest.TestCase):
    def _data(self):
        fold_ids = np.repeat((1, 2, 3, 4), 4)
        features = np.column_stack((fold_ids, np.arange(len(fold_ids))))
        targets = np.asarray(
            [
                [position % 2, (position // 2) % 2, position // 3]
                for _fold in (1, 2, 3, 4)
                for position in range(4)
            ],
            dtype=np.float64,
        )
        return features, targets, fold_ids

    def test_tuning_uses_only_inner_training_folds_and_records_candidates(self) -> None:
        features, targets, fold_ids = self._data()
        calls = []

        def fake_fit(train_x, train_y, test_x, **kwargs):
            calls.append((train_x.copy(), test_x.copy(), kwargs))
            return np.full((len(test_x), 3), 0.5)

        with patch("mmdii.training.tree_tuning.fit_predict_forest_ovr", side_effect=fake_fit):
            with patch(
                "mmdii.training.tree_tuning.evaluate_multilabel",
                return_value={"macro_pr_auc": 0.5},
            ):
                probabilities, diagnostics = tune_and_predict_forest_ovr(
                    features,
                    targets,
                    np.zeros((2, 2)),
                    train_fold_ids=fold_ids,
                    target_codes=("flash", "blur", "tunnel"),
                    random_state=17,
                )

        self.assertEqual(probabilities.shape, (2, 3))
        self.assertEqual(diagnostics["candidate_count"], 12)
        self.assertEqual(len(diagnostics["candidates"]), 12)
        self.assertEqual(diagnostics["inner_folds"], 4)
        self.assertEqual(len(calls), 12 * 4 + 1)
        for train_x, valid_x, _kwargs in calls[:-1]:
            self.assertFalse(set(train_x[:, 0]) & set(valid_x[:, 0]))
        self.assertEqual(diagnostics["selected_params"]["min_samples_leaf"], 4)

    def test_missing_inner_validation_class_fails_before_fitting(self) -> None:
        features, targets, fold_ids = self._data()
        targets[fold_ids == 1, 2] = 0
        targets[fold_ids == 2, 2] = 0
        with patch("mmdii.training.tree_tuning.fit_predict_forest_ovr") as fit:
            with self.assertRaisesRegex(ValueError, "lacks a positive or negative class"):
                tune_and_predict_forest_ovr(
                    features,
                    targets,
                    np.zeros((2, 2)),
                    train_fold_ids=fold_ids,
                    target_codes=("flash", "blur", "tunnel"),
                    random_state=7,
                )
        fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
