from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.reporting import (  # noqa: E402
    ProgressReporter,
    configure_logging,
    format_duration,
    format_eta,
    get_logger,
    log_results_table,
    resolve_level,
)
from mmdii.training.cross_validation import (  # noqa: E402
    ExperimentConfig,
    run_cross_validation,
)
from mmdii.training import cross_validation  # noqa: E402


def _pristine_mmdii_logger() -> logging.Logger:
    """Restore the ``mmdii`` logger to its silent, unconfigured default."""

    logger = logging.getLogger("mmdii")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
    logger.propagate = False
    return logger


def _is_managed(handler: logging.Handler) -> bool:
    return bool(getattr(handler, "_mmdii_managed", False))


class ReportingTests(unittest.TestCase):
    def tearDown(self) -> None:
        _pristine_mmdii_logger()

    def test_library_is_silent_by_default(self) -> None:
        logger = _pristine_mmdii_logger()
        self.assertFalse(logger.propagate)
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in logger.handlers))

        output = io.StringIO()
        with redirect_stderr(output):
            get_logger("mmdii.training").info("boom")
            get_logger("mmdii").warning("also silent")
        self.assertEqual(output.getvalue(), "")

    def test_configure_logging_emits_expected_format(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            get_logger("mmdii.training").info("hello")

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertRegex(
            lines[0],
            r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] INFO  training: hello$",
        )
        self.assertNotIn("\r", output.getvalue())

    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging("info")
        configure_logging("debug")
        logger = logging.getLogger("mmdii")
        self.assertEqual(sum(1 for handler in logger.handlers if _is_managed(handler)), 1)

    def test_configure_logging_respects_env_var(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"MMDII_LOG_LEVEL": "warning"}):
            self.assertEqual(resolve_level(), logging.WARNING)
            with redirect_stderr(output):
                configure_logging()
                get_logger("mmdii.training").info("suppressed")
                get_logger("mmdii.training").warning("kept")

        text = output.getvalue()
        self.assertNotIn("suppressed", text)
        self.assertIn("WARNING", text)
        self.assertIn("kept", text)

    def test_resolve_level_falls_back_to_info(self) -> None:
        with patch.dict(os.environ, {"MMDII_LOG_LEVEL": "not-a-level"}):
            self.assertEqual(resolve_level(), logging.INFO)
        self.assertEqual(resolve_level(15), 15)

    def test_format_duration_and_eta(self) -> None:
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(151.4), "00:02:31")
        self.assertEqual(format_duration(3723), "01:02:03")
        self.assertEqual(format_eta(0, 10, 1.0), "--:--:--")
        self.assertEqual(format_eta(1, 1, 2.0), "00:00:00")

    def test_progress_reporter_throttles_by_steps(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            reporter = ProgressReporter(
                get_logger("mmdii.training"),
                total=10,
                label="fold 0 train",
                every_steps=5,
                every_seconds=1e9,
            )
            for step in range(1, 11):
                reporter.update(step)
            reporter.close()

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("fold 0 train 5/10 (50%)", lines[0])
        self.assertIn("fold 0 train 10/10 (100%)", lines[1])

    def test_progress_reporter_emits_final_line(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            reporter = ProgressReporter(
                get_logger("mmdii.training"),
                total=4,
                label="tf transform train",
                every_steps=100,
                every_seconds=1e9,
            )
            for step in range(1, 5):
                reporter.update(step)
            reporter.close()

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("tf transform train 4/4 (100%)", lines[0])

    def test_progress_reporter_is_silent_when_unconfigured(self) -> None:
        _pristine_mmdii_logger()
        output = io.StringIO()
        with redirect_stderr(output):
            reporter = ProgressReporter(
                get_logger("mmdii.training"), total=3, label="quiet"
            )
            for step in range(1, 4):
                reporter.update(step)
            reporter.close()
        self.assertEqual(output.getvalue(), "")

    def test_import_emits_nothing(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(CORE_ROOT / "src")}
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mmdii.reporting, mmdii.training.cross_validation, mmdii.training.readiness",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_log_results_table_renders_none_values(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            configure_logging("info")
            log_results_table(
                get_logger("mmdii.training"),
                rows=[
                    {"fold": 0, "macro_f1": 0.5, "epochs_ran": 3},
                    {"fold": 1, "macro_f1": None, "epochs_ran": 4},
                ],
                columns=("fold", "macro_f1", "epochs_ran"),
            )
        text = output.getvalue()
        self.assertIn("results", text)
        self.assertIn("n/a", text)
        self.assertIn("0.5000", text)

    def test_run_cross_validation_summary_unchanged_with_logging(self) -> None:
        target_codes = ("flash", "blur", "tunnel")
        records = tuple(
            SimpleNamespace(
                sample_id=f"sample-{position}",
                weld_id=f"weld-{position}",
                image_group=f"group-{position}",
                fold=position % 5,
                target=(float(position % 2), float((position + 1) % 2), float(position % 2)),
                defect_codes=(),
            )
            for position in range(10)
        )
        index = SimpleNamespace(records=records, target_codes=target_codes)
        base = replace(ExperimentConfig.for_test(), mode="statistical", aggregator="mean")

        def fake_fold(_index, _train, valid, config, **_kwargs):
            return np.zeros((len(valid), len(config.target_codes)))

        def fake_metrics(_truth, _probabilities, *, target_codes):
            return {
                "per_code": {},
                "valid_code_count": 0,
                "macro_pr_auc": 0.6,
                "macro_recall": 0.4,
                "macro_f1": 0.5,
                "threshold": 0.5,
            }

        with tempfile.TemporaryDirectory() as quiet_dir, tempfile.TemporaryDirectory() as loud_dir:
            with patch.object(cross_validation, "_run_statistical_fold", side_effect=fake_fold):
                with patch.object(cross_validation, "evaluate_multilabel", side_effect=fake_metrics):
                    quiet = run_cross_validation(
                        index, replace(base, output_directory=Path(quiet_dir))
                    )
                    with redirect_stderr(io.StringIO()):
                        configure_logging("info")
                        loud = run_cross_validation(
                            index, replace(base, output_directory=Path(loud_dir))
                        )

        self.assertEqual(quiet, loud)


if __name__ == "__main__":
    unittest.main()
