from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TimeFrequencyMatrixTests(unittest.TestCase):
    def test_matrix_has_expected_configuration_count(self) -> None:
        with (ROOT / "configs" / "time_frequency_matrix.toml").open("rb") as handle:
            matrix = tomllib.load(handle)["matrix"]
        single = len(matrix["representations"]) * len(matrix["encoders"]) * len(matrix["aggregators"])
        dual = len(matrix["dual_branches"]) * len(matrix["encoders"]) * len(matrix["aggregators"])
        self.assertEqual(single + dual + 7, 103)
        self.assertEqual((single + dual) * len(matrix["seeds"]), 288)
        self.assertEqual((single + dual + 7) * len(matrix["seeds"]), 309)


if __name__ == "__main__":
    unittest.main()
