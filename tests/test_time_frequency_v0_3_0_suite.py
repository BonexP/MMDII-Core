from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "time_frequency_v0_3_0_matrix.toml"
SCRIPT_PATH = ROOT / "scripts" / "run_time_frequency_v0_3_0_suite.sh"


class TimeFrequencyV030SuiteTests(unittest.TestCase):
    def test_matrix_counts_match_the_declared_axes(self) -> None:
        with MATRIX_PATH.open("rb") as handle:
            matrix = tomllib.load(handle)["matrix"]

        deep = (
            len(matrix["feature_fusions"])
            * len(matrix["encoders"])
            * len(matrix["aggregators"])
        )
        per_seed = deep + len(matrix["controls"])
        self.assertEqual(matrix["version"], "v0.3.0")
        self.assertEqual(deep, matrix["counts"]["deep_per_seed"])
        self.assertEqual(len(matrix["controls"]), matrix["counts"]["control_per_seed"])
        self.assertEqual(per_seed, matrix["counts"]["per_seed"])
        self.assertEqual(per_seed * len(matrix["seeds"]), matrix["counts"]["total_seed_runs"])
        self.assertEqual(per_seed, 21)
        self.assertEqual(matrix["counts"]["total_seed_runs"], 63)

    def test_script_uses_the_declared_v030_axes_and_standard_validator(self) -> None:
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        with MATRIX_PATH.open("rb") as handle:
            matrix = tomllib.load(handle)["matrix"]

        for value in matrix["feature_fusions"] + matrix["encoders"] + matrix["aggregators"]:
            self.assertIn(value, script)
        self.assertIn('run_control "seed-${seed}-b0-statistical" statistical', script)
        self.assertIn('run_control "seed-${seed}-random-forest" random_forest random_forest', script)
        self.assertIn('run_control "seed-${seed}-extra-trees" random_forest extra_trees', script)
        self.assertIn("--expected-weld-count 101", script)
        self.assertIn("--expected-fold-count 5", script)
        self.assertIn('--expected-configurations-per-seed "$per_seed"', script)
        self.assertRegex(script, re.compile(r"per_seed=.*#BRANCHES.*#ENCODERS.*#AGGREGATORS"))


if __name__ == "__main__":
    unittest.main()
