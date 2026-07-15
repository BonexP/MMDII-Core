"""Tests for read-only MATLAB header inspection."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from scipy.io import savemat


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data import inspect_mat_directory, inspect_mat_file


class MatHeaderTests(unittest.TestCase):
    def test_inspect_mat_file_reports_sorted_variable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "sample.mat"
            savemat(path, {"z": [[1.0]], "a": [[1.0, 2.0]]})

            result = inspect_mat_file(path, root)

        self.assertEqual(result["path"], "sample.mat")
        self.assertEqual(
            result["variables"],
            [
                {"name": "a", "matlab_class": "double", "shape": [1, 2]},
                {"name": "z", "matlab_class": "double", "shape": [1, 1]},
            ],
        )

    def test_directory_keeps_special_headers_and_corrupt_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            savemat(root / "normal.mat", {"af": [[1.0]]})
            savemat(root / "special.mat", {"Artdata": [[1.0, 2.0]]})
            (root / "broken.mat").write_bytes(b"not a mat file")

            report = inspect_mat_directory(root)

        self.assertEqual(report["inspected_file_count"], 3)
        self.assertEqual(report["successful_file_count"], 2)
        self.assertEqual(report["failed_file_count"], 1)
        self.assertEqual(
            report["variable_frequency"],
            [
                {"name": "Artdata", "file_count": 1},
                {"name": "af", "file_count": 1},
            ],
        )
        self.assertEqual(report["files"][0]["path"], "broken.mat")
        self.assertEqual(report["files"][1]["variables"][0]["name"], "af")
        self.assertEqual(report["files"][2]["variables"][0]["name"], "Artdata")

    def test_directory_reports_file_and_field_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            savemat(root / "alpha.mat", {"shared": [[1.0]], "only_alpha": [[1.0]]})
            savemat(root / "bravo.mat", {"shared": [[1.0]], "only_bravo": [[1.0]]})
            savemat(
                root / "complete.mat",
                {
                    "shared": [[1.0]],
                    "only_alpha": [[1.0]],
                    "only_bravo": [[1.0]],
                },
            )
            savemat(root / "empty.mat", {})
            (root / "broken.mat").write_bytes(b"not a mat file")

            coverage = inspect_mat_directory(root)["coverage"]

        fields = {field["name"]: field for field in coverage["fields"]}
        files = {item["path"]: item for item in coverage["files"]}
        self.assertEqual(coverage["successful_file_count"], 4)
        self.assertEqual(coverage["total_distinct_variables"], 3)
        self.assertEqual(coverage["universal_variables"], [])
        self.assertEqual(coverage["fully_covered_files"], ["complete.mat"])
        self.assertEqual(
            fields["only_bravo"]["files"], ["bravo.mat", "complete.mat"]
        )
        self.assertEqual(fields["shared"]["coverage_percent"], 75.0)
        self.assertEqual(files["alpha.mat"]["missing_variables"], ["only_bravo"])
        self.assertEqual(
            files["empty.mat"]["missing_variables"],
            ["only_alpha", "only_bravo", "shared"],
        )

    def test_directory_reports_the_file_for_a_unique_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            savemat(root / "alpha.mat", {"shared": [[1.0]], "only_alpha": [[1.0]]})
            savemat(root / "bravo.mat", {"shared": [[1.0]]})

            coverage = inspect_mat_directory(root)["coverage"]

        self.assertEqual(
            coverage["unique_variables"],
            [{"name": "only_alpha", "file": "alpha.mat"}],
        )
        self.assertEqual(coverage["universal_variables"], ["shared"])


if __name__ == "__main__":
    unittest.main()
