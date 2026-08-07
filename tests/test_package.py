"""Minimal checks for the standalone MMDII-Core package."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii import __version__
from mmdii.data import DataSourceConfig, iter_mat_files


class PackageTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertEqual(__version__, "0.2.0")

    def test_missing_source_directory_is_reported(self) -> None:
        source = DataSourceConfig(
            root=CORE_ROOT / "missing-mat-directory",
            format="mat",
            read_only=True,
        )

        with self.assertRaises(FileNotFoundError):
            iter_mat_files(source)


if __name__ == "__main__":
    unittest.main()
