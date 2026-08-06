"""Synthetic integration tests for staged dataset construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from mmdii.data.dataset_builder import (
    EXCLUDED_HEADERS,
    INVENTORY_HEADERS,
    build_dataset_stage,
)
from mmdii.data.dataset_config import load_dataset_config


ANNOTATION_HEADERS = [
    "project_id", "project_name", "image_id", "image_relative_path",
    "image_order", "weld_index", "weld_id", "annotation_status",
    "is_normal", "defect_codes_json", "notes", "created_at", "updated_at",
]
DEFECT_HEADERS = [
    "project_id", "project_name", "image_id", "image_relative_path",
    "image_order", "weld_index", "weld_id", "defect_code", "created_at",
]


def write_annotation_release(path: Path) -> None:
    path.mkdir()
    rows = []
    defect_rows = []
    for weld_id, codes in (
        ("7", ["flash"]),
        ("8", []),
        ("9", ["blur"]),
        ("10", ["tunnel"]),
        ("11", ["blur", "flash"]),
        ("84", ["blur"]),
    ):
        row = {
            "project_id": "p", "project_name": "MMDII", "image_id": weld_id,
            "image_relative_path": f"image-{weld_id}.jpg", "image_order": weld_id,
            "weld_index": "1", "weld_id": weld_id, "annotation_status": "complete",
            "is_normal": "true" if not codes else "false",
            "defect_codes_json": json.dumps(codes), "notes": "", "created_at": "now",
            "updated_at": "now",
        }
        rows.append(row)
        for code in codes:
            defect_rows.append({key: row[key] for key in DEFECT_HEADERS if key != "defect_code"} | {"defect_code": code})
    with (path / "weld_annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_HEADERS)
        writer.writeheader(); writer.writerows(rows)
    with (path / "weld_defects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFECT_HEADERS)
        writer.writeheader(); writer.writerows(defect_rows)
    (path / "export_manifest.json").write_text(json.dumps({
        "release_id": "annotation-release", "mode": "dataset_ready",
        "annotation_contract_version": "1.1.0", "annotation_count": len(rows),
        "confirmed_defect_count": len(defect_rows),
    }), encoding="utf-8")


def signal_payload(length: int = 4, *, bad_af_length: int | None = None) -> dict[str, object]:
    fs = 6000.0
    return {
        "af": np.arange(bad_af_length or length, dtype=np.float64),
        "sf": np.arange(length, dtype=np.float64),
        "axialf": np.arange(length, dtype=np.float64),
        "time": np.arange(length, dtype=np.float64) / fs,
        "Fs": fs,
    }


def write_config(
    root: Path,
    *,
    read_only: bool = True,
    mapping_source: str = "owner confirmed",
    contract_version: str = "0.1.0",
    stride_seconds: float = 1.0,
) -> Path:
    (root / "raw_source.toml").write_text(
        f'[source]\nroot = "source"\nformat = "mat"\nread_only = {str(read_only).lower()}\n',
        encoding="utf-8",
    )
    config = root / "dataset.toml"
    config.write_text(
        "\n".join([
            "[dataset]",
            f'contract_version = "{contract_version}"',
            'raw_source_config = "raw_source.toml"',
            'annotation_release = "annotation-release"',
            'expected_annotation_release_id = "annotation-release"',
            'destination = "interim"',
            f'mapping_source = "{mapping_source}"',
            "excluded_run_ids = [84]",
            "",
            "[signals]",
            'fields = ["af", "sf", "axialf", "time"]',
            "time_step_rtol = 0.000001",
            "time_step_atol = 0.000000000001",
            "rpm_fs_rtol = 0.000001",
            "rpm_fs_atol = 0.000000001",
            *(
                [
                    "",
                    "[preprocessing]",
                    'target_fs = "auto"',
                    "window_seconds = 2.0",
                    f"stride_seconds = {stride_seconds}",
                    'normalization = "train_fold_zscore"',
                    "spectral_energy_fraction = 0.995",
                    "spectral_record_percentile = 0.95",
                    "nyquist_margin = 1.25",
                    "",
                    "[splits]",
                    "fold_count = 5",
                    'group_field = "image_relative_path"',
                ]
                if contract_version == "0.2.0"
                else []
            ),
        ]) + "\n",
        encoding="utf-8",
    )
    return config


class DatasetBuilderTests(unittest.TestCase):
    def test_config_resolves_paths_and_rejects_unsafe_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "source").mkdir()
            (root / "annotation-release").mkdir()
            config = load_dataset_config(write_config(root))

            self.assertEqual(config.source.root, (root / "source").resolve())
            self.assertEqual(config.annotation_release, (root / "annotation-release").resolve())
            self.assertEqual(config.destination, (root / "interim").resolve())
            self.assertEqual(config.excluded_run_ids, frozenset({84}))

            with self.assertRaisesRegex(ValueError, "read_only"):
                load_dataset_config(write_config(root, read_only=False))
            with self.assertRaisesRegex(ValueError, "mapping_source"):
                load_dataset_config(write_config(root, mapping_source=""))

    def test_config_loads_dataset_v0_2_training_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "source").mkdir()
            (root / "annotation-release").mkdir()

            config = load_dataset_config(
                write_config(root, contract_version="0.2.0")
            )

            self.assertEqual(config.contract_version, "0.2.0")
            self.assertEqual(config.preprocessing.target_fs, "auto")
            self.assertEqual(config.preprocessing.window_seconds, 2.0)
            self.assertEqual(config.preprocessing.stride_seconds, 1.0)
            self.assertEqual(
                config.preprocessing.normalization, "train_fold_zscore"
            )
            self.assertEqual(config.splits.fold_count, 5)
            self.assertEqual(config.splits.group_field, "image_relative_path")

            with self.assertRaisesRegex(ValueError, "stride_seconds"):
                load_dataset_config(
                    write_config(
                        root,
                        contract_version="0.2.0",
                        stride_seconds=3.0,
                    )
                )

    def test_builds_deterministic_stage_and_excludes_bad_base_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            write_annotation_release(root / "annotation-release")
            names = {
                "accepted": "2018-7-9-1.0-1.1-10000-200-7~.mat",
                "bad_quality": "2018-7-9-1.0-1.1-10000-200-8~.mat",
                "ambiguous_a": "2018-7-9-1.9-1.92-10000-200-84~.mat",
                "ambiguous_b": "2018-7-9-1.9-1.95-10000-200-84~.mat",
                "variant": "2018-7-9-1.0-1.1-10000-200-7~1.mat",
            }
            savemat(source / names["accepted"], signal_payload())
            savemat(source / names["bad_quality"], signal_payload(bad_af_length=3))
            savemat(source / names["ambiguous_a"], signal_payload())
            savemat(source / names["ambiguous_b"], signal_payload())
            savemat(source / names["variant"], signal_payload())
            config = load_dataset_config(write_config(root))
            stage = root / "stage"

            result = build_dataset_stage(config, stage)

            with (stage / "mat_inventory.csv").open(newline="", encoding="utf-8") as handle:
                inventory_reader = csv.DictReader(handle)
                inventory = list(inventory_reader)
                self.assertEqual(tuple(inventory_reader.fieldnames or ()), INVENTORY_HEADERS)
            with (stage / "excluded_samples.csv").open(newline="", encoding="utf-8") as handle:
                excluded_reader = csv.DictReader(handle)
                excluded = list(excluded_reader)
                self.assertEqual(tuple(excluded_reader.fieldnames or ()), EXCLUDED_HEADERS)
            with (stage / "samples.csv").open(newline="", encoding="utf-8") as handle:
                samples = list(csv.DictReader(handle))
            with (stage / "weld_sample_map.csv").open(newline="", encoding="utf-8") as handle:
                mappings = list(csv.DictReader(handle))

            self.assertEqual(len(inventory), 5)
            self.assertEqual(len(excluded), 3)
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(mappings), 1)
            self.assertEqual(samples[0]["run_id"], "7")
            self.assertEqual(samples[0]["label"], "1")
            self.assertEqual(mappings[0]["weld_id"], "7")
            self.assertEqual(result.discovered_count, 5)
            self.assertEqual(result.base_count, 4)
            self.assertEqual(result.variant_count, 1)
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.excluded_count, 3)
            npz_files = list((stage / "signals").glob("*.npz"))
            self.assertEqual(len(npz_files), 1)
            with np.load(npz_files[0], allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"af", "sf", "axialf", "time"})
            quality = json.loads((stage / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["issue_counts"]["ambiguous_run_id"], 2)
            self.assertEqual(quality["issue_counts"]["length_mismatch"], 1)

    def test_dataset_v0_2_writes_self_contained_sample_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            write_annotation_release(root / "annotation-release")
            for weld_id in range(7, 12):
                savemat(
                    source / f"2018-7-9-1.0-1.1-10000-200-{weld_id}~.mat",
                    signal_payload(),
                )
            config = load_dataset_config(
                write_config(root, contract_version="0.2.0")
            )
            stage = root / "stage"

            result = build_dataset_stage(config, stage)

            with (stage / "sample_labels.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                labels = list(reader)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    (
                        "sample_id",
                        "weld_id",
                        "is_normal",
                        "defect_codes_json",
                        "image_group",
                    ),
                )
            self.assertEqual(result.accepted_count, 5)
            self.assertEqual(len(labels), 5)
            by_weld = {row["weld_id"]: row for row in labels}
            self.assertEqual(json.loads(by_weld["7"]["defect_codes_json"]), ["flash"])
            self.assertEqual(by_weld["8"]["is_normal"], "true")
            self.assertEqual(by_weld["11"]["image_group"], "image-11.jpg")


if __name__ == "__main__":
    unittest.main()
