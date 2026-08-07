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


if __name__ == "__main__":
    unittest.main()
