from __future__ import annotations

from pathlib import Path
import re
import shutil
import shlex
import subprocess
import tomllib
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CORE_ROOT / "configs" / "tree_research_v0_3_2.toml"
SCRIPT = CORE_ROOT / "scripts" / "run_tree_research_v0_3_2_suite.sh"


def _bash() -> str | None:
    for candidate in (
        shutil.which("bash"),
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
    ):
        if not candidate or not Path(candidate).exists() or "WindowsApps" in candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return candidate
    return None


class TreeResearchV032SuiteTests(unittest.TestCase):
    def test_matrix_is_exactly_24_runs_and_keeps_the_shared_protocol(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        experiment = config["experiment"]
        matrix = config["matrix"]

        self.assertEqual(matrix["version"], "v0.3.2")
        self.assertEqual(matrix["seeds"], [7, 17, 27])
        self.assertEqual(experiment["fold_scheme"], "weld_independent")
        self.assertEqual(experiment["fold_count"], 5)
        self.assertEqual(matrix["release_name"], "weld-independent-v0-2-1-r2")
        self.assertEqual(matrix["expected_weld_count"], 101)
        self.assertEqual(matrix["counts"]["phase_a_total"], 12)
        self.assertEqual(matrix["counts"]["phase_b_total"], 6)
        self.assertEqual(matrix["counts"]["phase_c_total"], 6)
        self.assertEqual(matrix["counts"]["total"], 24)
        self.assertEqual(
            sum(matrix["counts"][key] for key in ("phase_a_total", "phase_b_total", "phase_c_total")),
            matrix["counts"]["total"],
        )
        script = SCRIPT.read_text(encoding="utf-8")
        for variable, field in (
            ("LEARNERS", "phase_a_learners"),
            ("BOOTSTRAP_VALUES", "bootstrap_values"),
            ("FEATURE_SETS", "phase_c_feature_sets"),
        ):
            match = re.search(rf"^{variable}=\(([^)]*)\)$", script, re.MULTILINE)
            self.assertIsNotNone(match, f"runner is missing {variable}")
            values = shlex.split(match.group(1))
            if field == "bootstrap_values":
                values = [value.lower() == "true" for value in values]
            self.assertEqual(values, matrix[field])
        seed_match = re.search(
            r'^SEEDS="\$\{SEEDS:-([^}]*)\}"$', script, re.MULTILINE
        )
        self.assertIsNotNone(seed_match, "runner seed defaults differ from the config")
        self.assertEqual(
            seed_match.group(1).split(), [str(seed) for seed in matrix["seeds"]]
        )
        self.assertEqual(matrix["phase_b_models"], ["gradient_boosting", "rbf_svm"])

    def test_runner_declares_matrix_axes_and_validates_before_marking_complete(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('SEEDS="${SEEDS:-7 17 27}"', text)
        self.assertIn("BOOTSTRAP_VALUES=(true false)", text)
        self.assertIn("LEARNERS=(random_forest extra_trees)", text)
        self.assertIn("gradient_boosting", text)
        self.assertIn("rbf_svm", text)
        self.assertIn("stats15_no_ptp", text)
        self.assertIn("stats12_no_ptp_rms", text)
        self.assertIn('--expected-run-count "$PLANNED"', text)
        self.assertIn('--expected-configurations-per-seed 8', text)

        marker = text.index('touch "$destination/.complete"')
        post_training_validation = text.rfind('validate_run "$destination"', 0, marker)
        self.assertGreater(post_training_validation, text.index('"$PYTHON" scripts/train_baseline.py'))
        self.assertLess(post_training_validation, marker)
        self.assertIn("true", text[post_training_validation:marker])
        self.assertRegex(
            text,
            re.compile(r'if validate_run "\$destination" "\$seed" "\$mode" "\$learner"'),
        )
        self.assertLess(text.index("if validate_run"), text.index("skip validated run"))

    @unittest.skipUnless(_bash(), "bash is not available")
    def test_runner_syntax_and_usage(self) -> None:
        bash = _bash()
        assert bash is not None
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        result = subprocess.run(
            [bash, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
