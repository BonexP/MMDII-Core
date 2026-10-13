from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CORE_ROOT / "scripts" / "run_tree_on_deep_v0_3_suite.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
def _find_bash() -> str | None:
    candidates = (
        shutil.which("bash"),
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
    )
    for candidate in candidates:
        if not candidate or "WindowsApps" in candidate or not Path(candidate).exists():
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


BASH = _find_bash()


def _default(name: str) -> str:
    match = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', TEXT, re.MULTILINE)
    assert match is not None, f"{name} default not found"
    return match.group(1)


class TreeOnDeepV03SuiteTests(unittest.TestCase):
    def test_default_matrix_is_48_runs(self) -> None:
        seeds = _default("SEEDS").split()
        learners = _default("TREE_LEARNERS").split()
        sources = _default("FEATURE_SOURCES").split()
        candidates = re.findall(r'"(raw_plus_(?:stft|cwt|dwt)\|[^"]+)"', TEXT)

        self.assertEqual(seeds, ["7", "17", "27"])
        self.assertEqual(learners, ["random_forest", "extra_trees"])
        self.assertEqual(sources, ["embedding", "both"])
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(seeds) * len(learners) * len(sources) * len(candidates), 48)
        self.assertIn("per_seed=$((${#candidates[@]} * ${#learner_array[@]} * ${#source_array[@]}))", TEXT)

    def test_deep_tree_uses_second_stage_cross_fitting(self) -> None:
        self.assertIn("--mode second_stage", TEXT)
        self.assertIn('--second-stage-source "$source"', TEXT)
        self.assertIn("--second-stage-inner-folds", TEXT)
        self.assertIn("--second-stage-encoder-epochs", TEXT)
        self.assertIn("--expected-weld-count 101", TEXT)
        self.assertIn("--expected-fold-count 5", TEXT)

    @unittest.skipUnless(BASH, "bash is not available")
    def test_script_parses_and_rejects_missing_arguments(self) -> None:
        result = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    @unittest.skipUnless(BASH, "bash is not available")
    def test_script_syntax_is_valid(self) -> None:
        result = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
