"""Build deterministic dataset artifacts in an unpublished staging directory."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset_config import DatasetPreparationConfig
from .mat_records import MatRecord, inventory_mat_files, sample_id_for
from .signal_quality import SignalAudit, audit_primary_signals
from .weld_annotations import build_weld_mappings, load_annotation_release


INVENTORY_HEADERS = (
    "mat_path",
    "parse_status",
    "is_variant",
    "base_mat_path",
    "sample_id",
    "run_id",
    "depth_start",
    "depth_end",
    "rpm",
    "welding_speed",
    "duplicate_suffix",
    "mapping_status",
    "quality_status",
    "issue_codes_json",
    "sample_count",
    "fs",
    "duration_seconds",
)

SAMPLE_HEADERS = (
    "sample_id",
    "mat_path",
    "depth_start",
    "depth_end",
    "rpm",
    "welding_speed",
    "run_id",
    "duplicate_suffix",
    "label",
    "vision_path",
    "split",
    "notes",
)

WELD_MAP_HEADERS = ("weld_id", "sample_id", "mapping_source", "notes")

EXCLUDED_HEADERS = (
    "mat_path",
    "sample_id",
    "run_id",
    "weld_id",
    "issue_codes_json",
    "notes",
)


@dataclass(frozen=True)
class DatasetBuildResult:
    stage_directory: Path
    annotation_release_id: str
    discovered_count: int
    base_count: int
    variant_count: int
    mapped_count: int
    accepted_count: int
    excluded_count: int
    issue_counts: dict[str, int]


def _write_csv(
    path: Path, headers: tuple[str, ...], rows: Iterable[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_codes(issues: tuple[str, ...] | list[str]) -> str:
    return json.dumps(list(issues), ensure_ascii=True, separators=(",", ":"))


def _base_path(record: MatRecord) -> str:
    if not record.is_variant:
        return record.relative_path
    return record.relative_path.removesuffix("~1.mat") + "~.mat"


def _number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return format(value, ".17g")


def _inventory_row(
    record: MatRecord,
    *,
    mapping_status: str,
    quality_status: str,
    issues: tuple[str, ...],
    audit: SignalAudit | None,
) -> dict[str, object]:
    return {
        "mat_path": record.relative_path,
        "parse_status": "parsed",
        "is_variant": str(record.is_variant).lower(),
        "base_mat_path": _base_path(record),
        "sample_id": "" if record.is_variant else sample_id_for(record),
        "run_id": record.run_id,
        "depth_start": _number(record.depth_start),
        "depth_end": _number(record.depth_end),
        "rpm": record.rpm,
        "welding_speed": record.welding_speed,
        "duplicate_suffix": record.duplicate_suffix,
        "mapping_status": mapping_status,
        "quality_status": quality_status,
        "issue_codes_json": _json_codes(issues),
        "sample_count": "" if audit is None else _number(audit.sample_count),
        "fs": "" if audit is None else _number(audit.fs),
        "duration_seconds": "" if audit is None else _number(audit.duration_seconds),
    }


def build_dataset_stage(
    config: DatasetPreparationConfig, stage_directory: str | Path
) -> DatasetBuildResult:
    """Build all non-manifest artifacts without publishing a release pointer."""

    stage = Path(stage_directory).resolve()
    if stage.exists():
        raise FileExistsError(f"Dataset stage already exists: {stage}")
    stage.mkdir(parents=True)
    signals_directory = stage / "signals"
    signals_directory.mkdir()

    release = load_annotation_release(config.annotation_release)
    if release.release_id != config.expected_annotation_release_id:
        raise ValueError(
            "Annotation release ID does not match expected_annotation_release_id."
        )
    records, parse_errors = inventory_mat_files(config.source.root)
    mapping_result = build_weld_mappings(
        records,
        release,
        excluded_run_ids=config.excluded_run_ids,
        mapping_source=config.mapping_source,
    )
    mapping_by_path = {
        mapping.record.relative_path: mapping for mapping in mapping_result.accepted
    }

    inventory_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    weld_map_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    issue_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    defect_counter: Counter[str] = Counter()

    for error in parse_errors:
        issue_counter[error["issue_code"]] += 1
        inventory_rows.append(
            {
                "mat_path": error["path"],
                "parse_status": "invalid",
                "is_variant": "",
                "base_mat_path": "",
                "sample_id": "",
                "run_id": "",
                "depth_start": "",
                "depth_end": "",
                "rpm": "",
                "welding_speed": "",
                "duplicate_suffix": "",
                "mapping_status": "unmapped",
                "quality_status": "not_audited",
                "issue_codes_json": _json_codes((error["issue_code"],)),
                "sample_count": "",
                "fs": "",
                "duration_seconds": "",
            }
        )

    accepted_count = 0
    for record in records:
        mapping_issues = mapping_result.issues_by_path.get(record.relative_path, ())
        if mapping_issues:
            issue_counter.update(mapping_issues)
            inventory_rows.append(
                _inventory_row(
                    record,
                    mapping_status="variant" if record.is_variant else "unmapped",
                    quality_status="not_applicable" if record.is_variant else "not_audited",
                    issues=mapping_issues,
                    audit=None,
                )
            )
            if not record.is_variant:
                excluded_rows.append(
                    {
                        "mat_path": record.relative_path,
                        "sample_id": sample_id_for(record),
                        "run_id": record.run_id,
                        "weld_id": "",
                        "issue_codes_json": _json_codes(mapping_issues),
                        "notes": "",
                    }
                )
            continue

        mapping = mapping_by_path[record.relative_path]
        audit = audit_primary_signals(
            record.path, rpm=record.rpm, config=config.signal_quality
        )
        if audit.issues:
            issue_counter.update(audit.issues)
            inventory_rows.append(
                _inventory_row(
                    record,
                    mapping_status="mapped",
                    quality_status="failed",
                    issues=audit.issues,
                    audit=audit,
                )
            )
            excluded_rows.append(
                {
                    "mat_path": record.relative_path,
                    "sample_id": mapping.sample_id,
                    "run_id": record.run_id,
                    "weld_id": mapping.weld_id,
                    "issue_codes_json": _json_codes(audit.issues),
                    "notes": "",
                }
            )
            continue

        assert audit.arrays is not None
        np.savez_compressed(
            signals_directory / f"{mapping.sample_id}.npz",
            time=audit.arrays[config.signal_quality.time_field],
            af=audit.arrays[config.signal_quality.force_fields[0]],
            sf=audit.arrays[config.signal_quality.force_fields[1]],
            axialf=audit.arrays[config.signal_quality.force_fields[2]],
        )
        label = "0" if mapping.annotation.is_normal else "1"
        label_counter["normal" if label == "0" else "fault"] += 1
        defect_counter.update(mapping.annotation.defect_codes)
        sample_rows.append(
            {
                "sample_id": mapping.sample_id,
                "mat_path": record.relative_path,
                "depth_start": _number(record.depth_start),
                "depth_end": _number(record.depth_end),
                "rpm": record.rpm,
                "welding_speed": record.welding_speed,
                "run_id": record.run_id,
                "duplicate_suffix": record.duplicate_suffix,
                "label": label,
                "vision_path": mapping.annotation.image_relative_path,
                "split": "",
                "notes": mapping.annotation.notes,
            }
        )
        weld_map_rows.append(
            {
                "weld_id": mapping.weld_id,
                "sample_id": mapping.sample_id,
                "mapping_source": mapping.mapping_source,
                "notes": "",
            }
        )
        inventory_rows.append(
            _inventory_row(
                record,
                mapping_status="mapped",
                quality_status="passed",
                issues=(),
                audit=audit,
            )
        )
        accepted_count += 1

    inventory_rows.sort(key=lambda row: str(row["mat_path"]).casefold())
    sample_rows.sort(key=lambda row: str(row["mat_path"]).casefold())
    weld_map_rows.sort(key=lambda row: str(row["sample_id"]))
    excluded_rows.sort(key=lambda row: str(row["mat_path"]).casefold())

    _write_csv(stage / "mat_inventory.csv", INVENTORY_HEADERS, inventory_rows)
    _write_csv(stage / "samples.csv", SAMPLE_HEADERS, sample_rows)
    _write_csv(stage / "weld_sample_map.csv", WELD_MAP_HEADERS, weld_map_rows)
    _write_csv(stage / "excluded_samples.csv", EXCLUDED_HEADERS, excluded_rows)

    base_count = sum(not record.is_variant for record in records)
    variant_count = sum(record.is_variant for record in records)
    report = {
        "accepted_count": accepted_count,
        "annotation_release_id": release.release_id,
        "base_count": base_count,
        "defect_counts": dict(sorted(defect_counter.items())),
        "discovered_count": len(records) + len(parse_errors),
        "excluded_count": len(excluded_rows),
        "issue_counts": dict(sorted(issue_counter.items())),
        "label_counts": dict(sorted(label_counter.items())),
        "mapped_count": len(mapping_result.accepted),
        "variant_count": variant_count,
    }
    (stage / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return DatasetBuildResult(
        stage_directory=stage,
        annotation_release_id=release.release_id,
        discovered_count=report["discovered_count"],
        base_count=base_count,
        variant_count=variant_count,
        mapped_count=len(mapping_result.accepted),
        accepted_count=accepted_count,
        excluded_count=len(excluded_rows),
        issue_counts=dict(sorted(issue_counter.items())),
    )
