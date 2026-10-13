from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CORE_ROOT / "scripts" / "run_tree_fusion_suite.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
BASH = shutil.which("bash")


def _default(name: str) -> str:
    match = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', TEXT, re.MULTILINE)
    assert match is not None, f"{name} default not found"
    return match.group(1)


class TreeFusionSuiteTests(unittest.TestCase):
    def test_planned_runs_are_accumulated_not_hard_coded(self) -> None:
        self.assertIn("PLANNED=0", TEXT)
        self.assertIn("PLANNED=$((PLANNED + 1))", TEXT)
        self.assertIn('--expected-run-count "$PLANNED"', TEXT)
        # The canonical suite's frozen count must not be copied into an expectation.
        self.assertFalse("$((103" in TEXT)
        self.assertFalse("expected-configurations-per-seed 103" in TEXT)

    def test_canonical_suite_is_left_untouched(self) -> None:
        canonical = (CORE_ROOT / "scripts" / "run_time_frequency_suite.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--expected-run-count "$((103 * ${#seed_array[@]}))"', canonical)

    def test_offline_fusion_is_not_invoked_by_the_training_suite(self) -> None:
        # Named in the header comment as an out-of-scope step, never executed.
        command_lines = [
            line for line in TEXT.splitlines() if not line.strip().startswith("#")
        ]
        self.assertFalse(any("fuse_oof" in line for line in command_lines))

    def test_declared_default_plan_matches_the_loops(self) -> None:
        seeds = _default("SEEDS").split()
        controls = _default("CONTROL_LEARNERS").split()
        arms = _default("ARMS").split()
        arm_learners = _default("ARM_LEARNERS").split()

        per_seed = len(controls) + len(arms) * len(arm_learners)

        self.assertEqual(len(seeds), 3)
        self.assertEqual(per_seed, 5)
        self.assertEqual(len(seeds) * per_seed, 15)
        # The script must derive the per-seed count from the same three knobs.
        self.assertIn(
            "per_seed=$(( ${#control_array[@]} + ${#arm_array[@]} * ${#arm_learner_array[@]} ))",
            TEXT,
        )

    def test_second_stage_arms_are_flagged_correctly(self) -> None:
        self.assertIn('--second-stage-source stats', TEXT)
        self.assertIn('--second-stage-source "$arm"', TEXT)
        self.assertIn("--second-stage-inner-folds", TEXT)
        self.assertIn("--second-stage-encoder-epochs", TEXT)
        # The stats arm must not pay for an encoder.
        stats_block = TEXT.split('if [[ "$arm" == "stats" ]]', 1)[1].split("else", 1)[0]
        self.assertNotIn("--second-stage-inner-folds", stats_block)
        self.assertIn("--second-stage-source stats", stats_block)

    @unittest.skipUnless(BASH, "bash is not available")
    def test_script_parses_and_rejects_missing_arguments(self) -> None:
        result = subprocess.run(
            [BASH, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    @unittest.skipUnless(BASH, "bash is not available")
    def test_script_syntax_is_valid(self) -> None:
        result = subprocess.run(
            [BASH, "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
