from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "scripts"))
sys.path.insert(0, str(CORE_ROOT / "src"))

from calibrate_oof_thresholds import calibrate  # noqa: E402


class CalibrateOofThresholdsTests(unittest.TestCase):
    def test_ap_uses_score_columns_and_thresholds_use_probability_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "oof_predictions.csv"
            rows = [
                {"sample_id": "a", "fold": "0", "target_codes_json": "[]", "prob_flash": "0.9", "score_flash": "-2"},
                {"sample_id": "b", "fold": "0", "target_codes_json": '["flash"]', "prob_flash": "0.1", "score_flash": "2"},
                {"sample_id": "c", "fold": "1", "target_codes_json": "[]", "prob_flash": "0.8", "score_flash": "-1"},
                {"sample_id": "d", "fold": "1", "target_codes_json": '["flash"]', "prob_flash": "0.2", "score_flash": "1"},
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            report = calibrate(path)

        self.assertEqual(report["pooled_per_code"]["flash"]["pr_auc"], 1.0)
        self.assertEqual(report["folds"][0]["thresholds"]["flash"], 0.2)


if __name__ == "__main__":
    unittest.main()
