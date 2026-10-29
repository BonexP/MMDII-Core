from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from unittest.mock import patch


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
sys.path.insert(0, str(CORE_ROOT / "scripts"))

import train_baseline


class TrainBaselineCliTests(unittest.TestCase):
    def tearDown(self) -> None:
        # train_baseline.main() calls configure_logging(); restore the silent
        # default so library logging never leaks into other tests.
        import logging

        logger = logging.getLogger("mmdii")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.NOTSET)
        logger.propagate = False

    def test_cli_loads_config_and_prints_summary(self) -> None:
        summary = {"mode": "window_mil", "sample_count": 101}
        output = io.StringIO()

        with patch.object(train_baseline, "run_cross_validation", return_value=summary):
            with redirect_stdout(output):
                exit_code = train_baseline.main(
                    ["--config", str(CORE_ROOT / "configs" / "moderntcn_mil_v0_1.toml")]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), summary)

    def test_cli_requires_a_configuration_path(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train_baseline.main([])

    def test_cli_routes_new_tabular_modes_and_tree_contract_overrides(self) -> None:
        captured = {}
        output = io.StringIO()
        release = CORE_ROOT / "training-release-override"

        def capture(_index, config):
            captured["config"] = config
            return {"mode": config.mode}

        with patch.object(train_baseline, "run_cross_validation", side_effect=capture):
            with patch.object(train_baseline.DatasetIndex, "from_release", return_value=object()):
                with redirect_stdout(output):
                    exit_code = train_baseline.main(
                        [
                            "--config",
                            str(CORE_ROOT / "configs" / "time_frequency_v0_3_1_tree.toml"),
                            "--release-dir",
                            str(release),
                            "--mode",
                            "rbf_svm",
                            "--tree-bootstrap",
                            "default",
                            "--tree-feature-set",
                            "stats15_no_ptp",
                            "--tree-interval-features",
                            "--tree-process-features",
                            "--tree-duration-feature",
                        ]
                    )

        config = captured["config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(config.mode, "rbf_svm")
        self.assertEqual(config.release_directory, release.resolve())
        self.assertIsNone(config.tree.bootstrap)
        self.assertEqual(config.tree.feature_set, "stats15_no_ptp")
        self.assertTrue(config.tree.interval_features)
        self.assertTrue(config.tree.process_features)
        self.assertTrue(config.tree.duration_feature)


if __name__ == "__main__":
    unittest.main()
