from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
PURE_CONFIG = ROOT / "configs" / "time_frequency_v0_3_1_pure_deep.toml"
TREE_CONFIG = ROOT / "configs" / "time_frequency_v0_3_1_tree.toml"
SCRIPTS = (
    ROOT / "scripts" / "run_time_frequency_v0_3_1_pure_deep_suite.sh",
    ROOT / "scripts" / "run_tree_v0_3_1_controls.sh",
    ROOT / "scripts" / "run_tree_on_deep_v0_3_1_suite.sh",
)


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


class V031SuiteConfigTests(unittest.TestCase):
    def test_pure_deep_matrix_is_five_by_three(self) -> None:
        matrix = tomllib.loads(PURE_CONFIG.read_text(encoding="utf-8"))["matrix"]
        self.assertEqual(matrix["version"], "v0.3.1")
        self.assertEqual(matrix["purpose"], "pure_deep")
        self.assertEqual(matrix["encoders"], [
            "resnet2d_small",
            "resnet2d_wide",
            "resnet2d_separable",
            "convnext2d_lite",
            "convnext2d_deep",
        ])
        self.assertEqual(matrix["seeds"], [7, 17, 27])
        self.assertEqual(matrix["counts"]["deep_per_seed"], 5)
        self.assertEqual(matrix["counts"]["total_seed_runs"], 15)

    def test_tree_matrix_declares_controls_and_48_run_deep_arm(self) -> None:
        matrix = tomllib.loads(TREE_CONFIG.read_text(encoding="utf-8"))["matrix"]
        self.assertEqual(matrix["control_models"], [
            "statistical",
            "random_forest_legacy",
            "random_forest_tuned",
            "extra_trees_legacy",
            "extra_trees_tuned",
        ])
        self.assertEqual(matrix["feature_sources"], [
            "embedding", "embedding_pca8", "both", "both_pca8"
        ])
        self.assertEqual(matrix["counts"]["tree_controls_total"], 15)
        self.assertEqual(matrix["counts"]["tree_on_deep_pilot"], 16)
        self.assertEqual(matrix["counts"]["tree_on_deep_total"], 48)

    def test_scripts_have_fixed_axes_and_strict_marker_order(self) -> None:
        pure, controls, deep = (path.read_text(encoding="utf-8") for path in SCRIPTS)
        self.assertIn("ENCODERS=(resnet2d_small resnet2d_wide resnet2d_separable convnext2d_lite convnext2d_deep)", pure)
        self.assertIn("touch \"$destination/.complete\"", pure)
        self.assertLess(pure.index("validate_run \"$destination\""), pure.index("touch \"$destination/.complete\""))
        self.assertIn("--tree-tune", controls)
        self.assertIn("run_one \"seed-${seed}-random-forest-tuned\"", controls)
        self.assertIn("FEATURE_SOURCES=\"${FEATURE_SOURCES:-embedding embedding_pca8 both both_pca8}\"", deep)
        self.assertIn("--second-stage-projection \"$projection\"", deep)
        self.assertIn("--second-stage-components 8", deep)
        self.assertIn("ENCODERS=(resnet2d_small convnext2d_lite)", deep)

    def test_scripts_are_valid_bash(self) -> None:
        bash = _bash()
        if bash is None:
            self.skipTest("bash is not available")
        for script in SCRIPTS:
            result = subprocess.run(
                [bash, "-n", str(script)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_scripts_reject_missing_arguments(self) -> None:
        bash = _bash()
        if bash is None:
            self.skipTest("bash is not available")
        for script in SCRIPTS:
            result = subprocess.run(
                [bash, str(script)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, str(script))
            self.assertRegex(result.stderr, re.compile("Usage:"))


if __name__ == "__main__":
    unittest.main()
