"""Tests for MATLAB experiment filename records."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.mat_records import (
    MatFilenameError,
    inventory_mat_files,
    parse_mat_filename,
    sample_id_for,
)


class MatRecordTests(unittest.TestCase):
    def test_parses_base_filename_metadata(self) -> None:
        root = Path("C:/source")
        record = parse_mat_filename(
            root / "2018-7-9-1.9-1.92-10000-200-84~.mat", root
        )

        self.assertEqual(record.relative_path, "2018-7-9-1.9-1.92-10000-200-84~.mat")
        self.assertEqual(record.date, "2018-7-9")
        self.assertEqual(record.depth_start, 1.9)
        self.assertEqual(record.depth_end, 1.92)
        self.assertEqual(record.rpm, 10000)
        self.assertEqual(record.welding_speed, 200)
        self.assertEqual(record.run_id, 84)
        self.assertEqual(record.duplicate_suffix, "")
        self.assertFalse(record.is_variant)

    def test_parses_variant_suffix(self) -> None:
        root = Path("C:/source")
        record = parse_mat_filename(
            root / "2018-7-9-1.9-1.92-10000-200-84~1.mat", root
        )

        self.assertEqual(record.duplicate_suffix, "1")
        self.assertTrue(record.is_variant)

    def test_rejects_nonconforming_filename(self) -> None:
        with self.assertRaisesRegex(MatFilenameError, "does not match"):
            parse_mat_filename(Path("bad.mat"), Path("."))

    def test_sample_ids_are_deterministic_and_path_specific(self) -> None:
        root = Path("C:/source")
        first = parse_mat_filename(
            root / "2018-7-9-1.9-1.92-10000-200-84~.mat", root
        )
        second = parse_mat_filename(
            root / "2018-7-9-1.9-1.95-10000-200-84~.mat", root
        )

        self.assertEqual(sample_id_for(first), sample_id_for(first))
        self.assertRegex(sample_id_for(first), r"^mat-084-[0-9a-f]{8}$")
        self.assertNotEqual(sample_id_for(first), sample_id_for(second))

    def test_inventory_keeps_valid_records_and_reports_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "2018-7-9-1.9-1.92-10000-200-84~.mat").touch()
            (root / "invalid.mat").touch()

            records, errors = inventory_mat_files(root)

        self.assertEqual([record.run_id for record in records], [84])
        self.assertEqual(
            errors,
            [
                {
                    "path": "invalid.mat",
                    "issue_code": "invalid_filename",
                    "message": "MAT filename does not match the experiment pattern: invalid.mat",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
