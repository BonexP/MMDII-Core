"""Validate immutable weld releases and materialize explicit MAT mappings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .mat_records import MatRecord, sample_id_for


ANNOTATION_HEADERS = (
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
)

DEFECT_HEADERS = (
    "project_id",
    "project_name",
    "image_id",
    "image_relative_path",
    "image_order",
    "weld_index",
    "weld_id",
    "defect_code",
    "created_at",
)


class AnnotationReleaseError(ValueError):
    """Raised when an annotation release violates its immutable contract."""


@dataclass(frozen=True)
class WeldAnnotation:
    weld_id: str
    image_relative_path: str
    annotation_status: str
    is_normal: bool
    defect_codes: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class AnnotationRelease:
    path: Path
    release_id: str
    contract_version: str
    annotations: dict[str, WeldAnnotation]
    confirmed_defect_count: int


@dataclass(frozen=True)
class WeldMapping:
    weld_id: str
    sample_id: str
    mapping_source: str
    record: MatRecord
    annotation: WeldAnnotation


@dataclass(frozen=True)
class MappingResult:
    accepted: tuple[WeldMapping, ...]
    issues_by_path: dict[str, tuple[str, ...]]


def _read_csv(path: Path, expected_headers: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_headers:
                raise AnnotationReleaseError(
                    f"Unexpected headers in {path.name}: {reader.fieldnames}"
                )
            return list(reader)
    except OSError as error:
        raise AnnotationReleaseError(f"Could not read {path.name}: {error}") from error


def load_annotation_release(path: str | Path) -> AnnotationRelease:
    """Load and cross-check a dataset-ready weld annotation release."""

    release_path = Path(path).resolve()
    manifest_path = release_path / "export_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationReleaseError(f"Could not read export_manifest.json: {error}") from error

    if manifest.get("mode") != "dataset_ready":
        raise AnnotationReleaseError("Annotation release mode must be dataset_ready.")
    if manifest.get("annotation_contract_version") != "1.1.0":
        raise AnnotationReleaseError("Annotation contract version must be 1.1.0.")

    annotation_rows = _read_csv(
        release_path / "weld_annotations.csv", ANNOTATION_HEADERS
    )
    defect_rows = _read_csv(release_path / "weld_defects.csv", DEFECT_HEADERS)
    if manifest.get("annotation_count") != len(annotation_rows):
        raise AnnotationReleaseError("Manifest annotation_count does not match the CSV.")
    if manifest.get("confirmed_defect_count") != len(defect_rows):
        raise AnnotationReleaseError(
            "Manifest confirmed_defect_count does not match the CSV."
        )

    annotations: dict[str, WeldAnnotation] = {}
    expected_relations: set[tuple[str, str]] = set()
    for row in annotation_rows:
        weld_id = row["weld_id"].strip()
        if not weld_id:
            raise AnnotationReleaseError("weld_id must not be empty.")
        if weld_id in annotations:
            raise AnnotationReleaseError(f"Duplicate weld_id in annotation release: {weld_id}")
        if row["annotation_status"] != "complete":
            raise AnnotationReleaseError(
                f"Dataset-ready weld {weld_id} is not complete."
            )
        try:
            parsed_codes = json.loads(row["defect_codes_json"])
        except json.JSONDecodeError as error:
            raise AnnotationReleaseError(
                f"Invalid defect_codes_json for weld {weld_id}."
            ) from error
        if (
            not isinstance(parsed_codes, list)
            or any(not isinstance(code, str) or not code for code in parsed_codes)
            or len(set(parsed_codes)) != len(parsed_codes)
        ):
            raise AnnotationReleaseError(
                f"defect_codes_json for weld {weld_id} must be unique strings."
            )
        if row["is_normal"] not in {"true", "false"}:
            raise AnnotationReleaseError(f"Invalid is_normal for weld {weld_id}.")
        is_normal = row["is_normal"] == "true"
        if is_normal != (len(parsed_codes) == 0):
            raise AnnotationReleaseError(
                f"Normal/defect semantics are inconsistent for weld {weld_id}."
            )
        defect_codes = tuple(parsed_codes)
        annotations[weld_id] = WeldAnnotation(
            weld_id=weld_id,
            image_relative_path=row["image_relative_path"],
            annotation_status=row["annotation_status"],
            is_normal=is_normal,
            defect_codes=defect_codes,
            notes=row["notes"],
        )
        expected_relations.update((weld_id, code) for code in defect_codes)

    actual_relations = [(row["weld_id"], row["defect_code"]) for row in defect_rows]
    if len(set(actual_relations)) != len(actual_relations):
        raise AnnotationReleaseError("Duplicate normalized weld-defect relation.")
    if set(actual_relations) != expected_relations:
        raise AnnotationReleaseError(
            "weld_defects.csv does not match defect_codes_json relationships."
        )

    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise AnnotationReleaseError("Manifest release_id must be a non-empty string.")
    return AnnotationRelease(
        path=release_path,
        release_id=release_id,
        contract_version="1.1.0",
        annotations=annotations,
        confirmed_defect_count=len(actual_relations),
    )


def build_weld_mappings(
    records: Iterable[MatRecord],
    release: AnnotationRelease,
    *,
    excluded_run_ids: set[int] | frozenset[int],
    mapping_source: str,
) -> MappingResult:
    """Materialize unique owner-confirmed run-id mappings for base records."""

    if not mapping_source.strip():
        raise ValueError("mapping_source must not be empty.")

    ordered_records = sorted(records, key=lambda record: record.relative_path.casefold())
    base_by_run: dict[int, list[MatRecord]] = {}
    issues: dict[str, tuple[str, ...]] = {}
    for record in ordered_records:
        if record.is_variant:
            issues[record.relative_path] = ("variant_not_sample",)
        else:
            base_by_run.setdefault(record.run_id, []).append(record)

    accepted: list[WeldMapping] = []
    for run_id in sorted(base_by_run):
        candidates = base_by_run[run_id]
        if len(candidates) != 1 or run_id in excluded_run_ids:
            for record in candidates:
                issues[record.relative_path] = ("ambiguous_run_id",)
            continue

        record = candidates[0]
        weld_id = str(run_id)
        annotation = release.annotations.get(weld_id)
        if annotation is None:
            issues[record.relative_path] = ("annotation_not_found",)
            continue
        accepted.append(
            WeldMapping(
                weld_id=weld_id,
                sample_id=sample_id_for(record),
                mapping_source=mapping_source,
                record=record,
                annotation=annotation,
            )
        )

    weld_ids = [mapping.weld_id for mapping in accepted]
    sample_ids = [mapping.sample_id for mapping in accepted]
    if len(weld_ids) != len(set(weld_ids)) or len(sample_ids) != len(set(sample_ids)):
        raise AnnotationReleaseError("mapping_conflict in accepted weld mappings.")

    return MappingResult(
        accepted=tuple(accepted),
        issues_by_path=dict(sorted(issues.items())),
    )
