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


if __name__ == "__main__":
    unittest.main()
