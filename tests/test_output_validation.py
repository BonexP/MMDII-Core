from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_tracked_outputs.py"
SPEC = importlib.util.spec_from_file_location("validate_tracked_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OutputValidationTests(unittest.TestCase):
    def test_accepts_complete_five_fold_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            (run / ".complete").touch()
            (run / "training_summary.json").write_text(
                json.dumps({"sample_count": 1}), encoding="utf-8"
            )
            (run / "run_config.json").write_text("{}", encoding="utf-8")
            (run / "fold_metrics.json").write_text(
                json.dumps([{}] * 5), encoding="utf-8"
            )
            with (run / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "prob_flash"))
                writer.writeheader()
                writer.writerow({"sample_id": "sample-1", "prob_flash": "0.5"})

            result = MODULE.validate_run(run)

            self.assertTrue(result["complete"])
            self.assertEqual(result["sample_count"], 1)

    def test_rejects_missing_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = MODULE.validate_run(Path(temporary_directory))

            self.assertFalse(result["complete"])
            self.assertIn("missing .complete marker", result["errors"])

    def test_allows_partial_fold_reports_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            (run / ".complete").touch()
            (run / "training_summary.json").write_text(
                json.dumps({"sample_count": 5}), encoding="utf-8"
            )
            (run / "run_config.json").write_text("{}", encoding="utf-8")
            (run / "fold_metrics.json").write_text(json.dumps([{}]), encoding="utf-8")
            with (run / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("sample_id", "weld_id", "fold", "prob_flash")
                )
                writer.writeheader()
                for fold in range(5):
                    writer.writerow(
                        {
                            "sample_id": f"sample-{fold}",
                            "weld_id": f"weld-{fold}",
                            "fold": fold,
                            "prob_flash": "0.5",
                        }
                    )

            self.assertFalse(MODULE.validate_run(run)["complete"])
            self.assertTrue(MODULE.validate_run(run, allow_partial=True)["complete"])

    def test_enforces_expected_weld_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            (run / ".complete").touch()
            (run / "training_summary.json").write_text(
                json.dumps({"sample_count": 1}), encoding="utf-8"
            )
            (run / "run_config.json").write_text(
                json.dumps({"fold_scheme": "weld_independent", "seed": 7}),
                encoding="utf-8",
            )
            (run / "fold_metrics.json").write_text(json.dumps([{}]), encoding="utf-8")
            with (run / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("sample_id", "weld_id", "fold", "prob_flash")
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample-1",
                        "weld_id": "weld-1",
                        "fold": 0,
                        "prob_flash": "0.5",
                    }
                )

            result = MODULE.validate_run(
                run,
                allow_partial=True,
                expected_weld_count=101,
                expected_fold_count=1,
                expected_fold_scheme="weld_independent",
            )

            self.assertFalse(result["complete"])
            self.assertIn("expected 101 unique welds, found 1", result["errors"])


if __name__ == "__main__":
    unittest.main()
