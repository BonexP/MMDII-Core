from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tomllib
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CORE_ROOT / "configs" / "tree_interval_v0_3_3.toml"
SCRIPT = CORE_ROOT / "scripts" / "run_tree_interval_v0_3_3_suite.sh"


def _bash() -> str | None:
    for candidate in (
        shutil.which("bash"),
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
    ):
        if candidate and Path(candidate).exists() and "WindowsApps" not in candidate:
            try:
                result = subprocess.run(
                    [candidate, "--version"], capture_output=True, text=True,
                    check=False, timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                return candidate
    return None


class TreeIntervalV033SuiteTests(unittest.TestCase):
    def test_factorial_matrix_and_shared_protocol(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        matrix = config["matrix"]
        self.assertEqual(matrix["version"], "v0.3.3")
        self.assertEqual(matrix["seeds"], [7, 17, 27])
        self.assertEqual(matrix["configurations_per_seed"], 8)
        self.assertEqual(matrix["total_runs"], 24)
        self.assertEqual(config["experiment"]["fold_count"], 5)
        self.assertEqual(config["experiment"]["fold_scheme"], "weld_independent")
        self.assertEqual(matrix["release_name"], "weld-independent-v0-2-1-r2")
        self.assertEqual(matrix["expected_weld_count"], 101)

    def test_runner_has_all_axes_and_validates_before_marker(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('SEEDS="${SEEDS:-7 17 27}"', text)
        for axis in ("interval", "process", "duration"):
            self.assertIn(f"for {axis} in false true", text)
        self.assertIn('--expected-run-count "$PLANNED"', text)
        marker = text.index('touch "$destination/.complete"')
        validation = text.rfind(
            'validate_run "$destination" "$seed" "$interval" "$process" "$duration" true',
            0,
            marker,
        )
        self.assertGreater(validation, text.index('scripts/train_baseline.py "${args[@]}"'))
        self.assertLess(validation, marker)
        self.assertIn("tree.get(key) != value", text)
        self.assertIn("if validate_run \"$destination\"", text)

    @unittest.skipUnless(_bash(), "bash is not available")
    def test_runner_shell_syntax_and_usage(self) -> None:
        bash = _bash()
        assert bash is not None
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [bash, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
