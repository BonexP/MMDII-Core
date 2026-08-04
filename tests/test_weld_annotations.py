"""Tests for immutable weld annotation releases and MAT mappings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.mat_records import parse_mat_filename
from mmdii.data.weld_annotations import (
    AnnotationReleaseError,
    build_weld_mappings,
    load_annotation_release,
)


ANNOTATION_HEADERS = [
    "project_id",
    "project_name",
    "image_id",
    "image_relative_path",
    "image_order",
    "weld_index",
    "weld_id",
    "annotation_status",
    "is_normal",
    "defect_codes_json",
    "notes",
    "created_at",
    "updated_at",
]

DEFECT_HEADERS = [
    "project_id",
    "project_name",
    "image_id",
    "image_relative_path",
    "image_order",
    "weld_index",
    "weld_id",
    "defect_code",
    "created_at",
]


def annotation_row(weld_id: str, defects: list[str]) -> dict[str, str]:
    return {
        "project_id": "project-1",
        "project_name": "MMDII",
        "image_id": f"image-{weld_id}",
        "image_relative_path": f"image-{weld_id}.jpg",
        "image_order": weld_id,
        "weld_index": "1",
        "weld_id": weld_id,
        "annotation_status": "complete",
        "is_normal": "true" if not defects else "false",
        "defect_codes_json": json.dumps(defects),
        "notes": "",
        "created_at": "2026-08-04T00:00:00Z",
        "updated_at": "2026-08-04T00:00:00Z",
    }


def write_release(
    root: Path,
    rows: list[dict[str, str]],
    *,
    mode: str = "dataset_ready",
    annotation_count: int | None = None,
) -> Path:
    root.mkdir()
    defect_rows = []
    for row in rows:
        for code in json.loads(row["defect_codes_json"]):
            defect_rows.append(
                {
                    key: row[key]
                    for key in DEFECT_HEADERS
                    if key != "defect_code"
                }
                | {"defect_code": code}
            )

    with (root / "weld_annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "weld_defects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFECT_HEADERS)
        writer.writeheader()
        writer.writerows(defect_rows)
    (root / "export_manifest.json").write_text(
        json.dumps(
            {
                "release_id": "release-1",
                "project_id": "project-1",
                "project_name": "MMDII",
                "mode": mode,
                "generated_at": "2026-08-04T00:00:00Z",
                "annotation_contract_version": "1.1.0",
                "annotation_count": len(rows) if annotation_count is None else annotation_count,
                "confirmed_defect_count": len(defect_rows),
                "files": [
                    "weld_annotations.csv",
                    "weld_defects.csv",
                    "export_manifest.json",
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class WeldAnnotationTests(unittest.TestCase):
    def test_loads_dataset_ready_release_and_normalizes_defects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = write_release(
                Path(temporary_directory) / "release",
                [annotation_row("7", ["flash", "blur"]), annotation_row("8", [])],
            )

            release = load_annotation_release(release_dir)

        self.assertEqual(release.release_id, "release-1")
        self.assertEqual(release.annotations["7"].defect_codes, ("flash", "blur"))
        self.assertFalse(release.annotations["7"].is_normal)
        self.assertTrue(release.annotations["8"].is_normal)

    def test_rejects_non_ready_or_count_mismatched_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit = write_release(root / "audit", [annotation_row("7", [])], mode="audit")
            mismatch = write_release(
                root / "mismatch", [annotation_row("7", [])], annotation_count=2
            )

            with self.assertRaisesRegex(AnnotationReleaseError, "dataset_ready"):
                load_annotation_release(audit)
            with self.assertRaisesRegex(AnnotationReleaseError, "annotation_count"):
                load_annotation_release(mismatch)

    def test_rejects_duplicate_weld_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = write_release(
                Path(temporary_directory) / "release",
                [annotation_row("7", []), annotation_row("7", [])],
            )

            with self.assertRaisesRegex(AnnotationReleaseError, "Duplicate weld_id"):
                load_annotation_release(release_dir)

    def test_builds_only_unique_owner_confirmed_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            release = load_annotation_release(
                write_release(root / "release", [annotation_row("7", ["flash"]), annotation_row("192083", [])])
            )
            source = root / "source"
            source.mkdir()
            names = [
                "2018-7-9-1.0-1.1-10000-200-7~.mat",
                "2018-7-9-1.9-1.92-10000-200-84~.mat",
                "2018-7-9-1.9-1.95-10000-200-84~.mat",
                "2018-7-9-1.0-1.1-10000-200-9~.mat",
                "2018-7-9-1.0-1.1-10000-200-7~1.mat",
            ]
            records = [parse_mat_filename(source / name, source) for name in names]

            result = build_weld_mappings(
                records,
                release,
                excluded_run_ids={84},
                mapping_source="owner-confirmed run_id equals weld_id",
            )

        self.assertEqual([mapping.weld_id for mapping in result.accepted], ["7"])
        self.assertEqual(result.accepted[0].mapping_source, "owner-confirmed run_id equals weld_id")
        self.assertEqual(result.issues_by_path[names[1]], ("ambiguous_run_id",))
        self.assertEqual(result.issues_by_path[names[2]], ("ambiguous_run_id",))
        self.assertEqual(result.issues_by_path[names[3]], ("annotation_not_found",))
        self.assertEqual(result.issues_by_path[names[4]], ("variant_not_sample",))


if __name__ == "__main__":
    unittest.main()
