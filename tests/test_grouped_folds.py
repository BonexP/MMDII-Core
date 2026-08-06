from __future__ import annotations

from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.grouped_folds import (
    LabelledSample,
    assign_grouped_folds,
    build_split_report,
)


class GroupedFoldTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_keeps_image_groups_together(self) -> None:
        samples = (
            LabelledSample("s1", "1", "image-a", True, ()),
            LabelledSample("s2", "2", "image-a", False, ("flash",)),
            LabelledSample("s3", "3", "image-b", False, ("blur",)),
            LabelledSample("s4", "4", "image-c", False, ("tunnel",)),
            LabelledSample("s5", "5", "image-d", False, ("flash",)),
            LabelledSample("s6", "6", "image-e", True, ()),
            LabelledSample("s7", "7", "image-f", False, ("pore",)),
            LabelledSample("s8", "8", "image-f", False, ("blur", "flash")),
        )

        assignments = assign_grouped_folds(samples, fold_count=5)

        self.assertEqual(assignments, assign_grouped_folds(samples, fold_count=5))
        self.assertEqual({row.fold for row in assignments}, {0, 1, 2, 3, 4})
        folds_by_group: dict[str, set[int]] = {}
        for row in assignments:
            folds_by_group.setdefault(row.image_group, set()).add(row.fold)
        self.assertTrue(all(len(folds) == 1 for folds in folds_by_group.values()))

        report = build_split_report(samples, assignments, fold_count=5)
        warnings = {
            item["label"]: item["positive_group_count"]
            for item in report["warnings"]
        }
        self.assertEqual(warnings["pore"], 1)
        self.assertEqual(warnings["normal"], 2)

    def test_rejects_fewer_groups_than_folds(self) -> None:
        samples = (
            LabelledSample("s1", "1", "image-a", True, ()),
            LabelledSample("s2", "2", "image-b", False, ("flash",)),
        )

        with self.assertRaisesRegex(ValueError, "fewer image groups"):
            assign_grouped_folds(samples, fold_count=5)


if __name__ == "__main__":
    unittest.main()
