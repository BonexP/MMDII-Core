"""Validate and atomically publish immutable dataset releases."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from uuid import uuid4

import numpy as np

from .dataset_builder import DatasetBuildResult, build_dataset_stage
from .dataset_config import DatasetPreparationConfig


DATASET_MANIFEST = "dataset_manifest.json"
REQUIRED_ARTIFACTS = {
    "excluded_samples.csv",
    "mat_inventory.csv",
    "quality_report.json",
    "samples.csv",
    "weld_sample_map.csv",
}

V0_2_REQUIRED_ARTIFACTS = {
    "folds.csv",
    "preprocessing.json",
    "sample_labels.csv",
    "signal_spectrum.csv",
    "split_report.json",
}


class DatasetReleaseError(ValueError):
    """Raised when a staged or published dataset release is invalid."""


@dataclass(frozen=True)
class PublishedDataset:
    release_id: str
    release_directory: Path
    manifest_path: Path
    current_pointer: Path
    result: DatasetBuildResult


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_checksums(stage_directory: str | Path) -> list[dict[str, object]]:
    """Describe every staged artifact except the self-referential manifest."""

    stage = Path(stage_directory).resolve()
    artifacts = []
    for path in sorted(
        (item for item in stage.rglob("*") if item.is_file() and item.name != DATASET_MANIFEST),
        key=lambda item: item.relative_to(stage).as_posix(),
    ):
        artifacts.append(
            {
                "path": path.relative_to(stage).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


def _git_commit(config: DatasetPreparationConfig) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.config_path.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _write_manifest(
    stage: Path,
    *,
    release_id: str,
    config: DatasetPreparationConfig,
    result: DatasetBuildResult,
) -> dict[str, object]:
    artifacts = build_artifact_checksums(stage)
    configuration: dict[str, object] = {
        "excluded_run_ids": sorted(config.excluded_run_ids),
        "fields": list(config.signal_quality.fields),
        "mapping_source": config.mapping_source,
        "rpm_fs_atol": config.signal_quality.rpm_fs_atol,
        "rpm_fs_rtol": config.signal_quality.rpm_fs_rtol,
        "time_step_atol": config.signal_quality.time_step_atol,
        "time_step_rtol": config.signal_quality.time_step_rtol,
    }
    if config.contract_version == "0.2.0":
        assert config.preprocessing is not None and config.splits is not None
        configuration["preprocessing"] = {
            "normalization": config.preprocessing.normalization,
            "nyquist_margin": config.preprocessing.nyquist_margin,
            "spectral_energy_fraction": config.preprocessing.spectral_energy_fraction,
            "spectral_record_percentile": config.preprocessing.spectral_record_percentile,
            "stride_seconds": config.preprocessing.stride_seconds,
            "target_fs": config.preprocessing.target_fs,
            "window_seconds": config.preprocessing.window_seconds,
        }
        configuration["splits"] = {
            "fold_count": config.splits.fold_count,
            "group_field": config.splits.group_field,
        }
    manifest = {
        "annotation_contract_version": "1.1.0",
        "annotation_release_id": result.annotation_release_id,
        "artifacts": artifacts,
        "configuration": configuration,
        "counts": {
            "accepted": result.accepted_count,
            "base": result.base_count,
            "discovered": result.discovered_count,
            "excluded": result.excluded_count,
            "mapped": result.mapped_count,
            "variant": result.variant_count,
        },
        "dataset_contract_version": config.contract_version,
        "files": [artifact["path"] for artifact in artifacts],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(config),
        "release_id": release_id,
        "source_config_path": config.source_config_path.as_posix(),
        "source_root": config.source.root.as_posix(),
    }
    (stage / DATASET_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _safe_artifact_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise DatasetReleaseError("Artifact path must be a non-empty string.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DatasetReleaseError(f"Unsafe artifact path: {value}")
    return path


def _read_csv(
    path: Path, expected_headers: tuple[str, ...]
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_headers:
                raise DatasetReleaseError(f"Unexpected CSV schema: {path.name}")
            return list(reader)
    except OSError as error:
        raise DatasetReleaseError(f"Could not read {path.name}: {error}") from error


def _validate_v0_2_release(
    release: Path, manifest: dict[str, object], accepted: int
) -> None:
    samples = _read_csv(
        release / "samples.csv",
        (
            "sample_id", "mat_path", "depth_start", "depth_end", "rpm",
            "welding_speed", "run_id", "duplicate_suffix", "label",
            "vision_path", "split", "notes",
        ),
    )
    labels = _read_csv(
        release / "sample_labels.csv",
        ("sample_id", "weld_id", "is_normal", "defect_codes_json", "image_group"),
    )
    folds = _read_csv(
        release / "folds.csv", ("sample_id", "weld_id", "image_group", "fold")
    )
    spectrum = _read_csv(
        release / "signal_spectrum.csv",
        (
            "sample_id", "channel", "original_fs_hz", "duration_seconds",
            "energy_fraction", "energy_cutoff_hz", "nyquist_hz",
        ),
    )

    sample_ids = [row["sample_id"] for row in samples]
    expected_ids = set(sample_ids)
    if len(sample_ids) != accepted or len(expected_ids) != accepted or "" in expected_ids:
        raise DatasetReleaseError("samples.csv does not match accepted samples.")
    for name, rows in (("sample_labels.csv", labels), ("folds.csv", folds)):
        ids = [row["sample_id"] for row in rows]
        if len(ids) != accepted or len(set(ids)) != accepted or set(ids) != expected_ids:
            raise DatasetReleaseError(f"{name} sample IDs do not match samples.csv.")

    labels_by_id = {row["sample_id"]: row for row in labels}
    for row in labels:
        try:
            defect_codes = json.loads(row["defect_codes_json"])
        except json.JSONDecodeError as error:
            raise DatasetReleaseError("Invalid defect_codes_json.") from error
        if (
            row["is_normal"] not in {"true", "false"}
            or not isinstance(defect_codes, list)
            or any(not isinstance(code, str) or not code for code in defect_codes)
            or defect_codes != sorted(set(defect_codes))
            or (row["is_normal"] == "true") != (len(defect_codes) == 0)
            or not row["weld_id"]
            or not row["image_group"]
        ):
            raise DatasetReleaseError("Invalid sample label semantics.")

    configuration = manifest.get("configuration")
    splits = configuration.get("splits") if isinstance(configuration, dict) else None
    fold_count = splits.get("fold_count") if isinstance(splits, dict) else None
    if not isinstance(fold_count, int) or fold_count < 2:
        raise DatasetReleaseError("Manifest fold configuration is invalid.")
    group_folds: dict[str, set[int]] = {}
    for row in folds:
        label = labels_by_id[row["sample_id"]]
        if row["weld_id"] != label["weld_id"] or row["image_group"] != label["image_group"]:
            raise DatasetReleaseError("Fold metadata does not match sample labels.")
        try:
            fold = int(row["fold"])
        except ValueError as error:
            raise DatasetReleaseError("Invalid fold value.") from error
        if str(fold) != row["fold"] or fold < 0 or fold >= fold_count:
            raise DatasetReleaseError("Fold value is outside the configured range.")
        group_folds.setdefault(row["image_group"], set()).add(fold)
    if any(len(values) != 1 for values in group_folds.values()):
        raise DatasetReleaseError("An image group crosses folds.")
    if {int(row["fold"]) for row in folds} != set(range(fold_count)):
        raise DatasetReleaseError("Not every configured fold is populated.")

    spectrum_keys = [(row["sample_id"], row["channel"]) for row in spectrum]
    expected_spectrum_keys = {
        (sample_id, channel)
        for sample_id in expected_ids
        for channel in ("af", "sf", "axialf")
    }
    if len(spectrum_keys) != 3 * accepted or set(spectrum_keys) != expected_spectrum_keys:
        raise DatasetReleaseError("Spectrum rows do not match accepted force signals.")
    try:
        preprocessing = json.loads(
            (release / "preprocessing.json").read_text(encoding="utf-8")
        )
        split_report = json.loads(
            (release / "split_report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetReleaseError(f"Invalid v0.2 JSON artifact: {error}") from error
    if (
        not isinstance(preprocessing, dict)
        or not isinstance(preprocessing.get("recommended_target_fs_hz"), (int, float))
        or preprocessing["recommended_target_fs_hz"] <= 0
        or not isinstance(split_report, dict)
        or split_report.get("fold_count") != fold_count
    ):
        raise DatasetReleaseError("Invalid v0.2 preprocessing or split report.")


def validate_dataset_release(path: str | Path) -> dict[str, object]:
    """Independently validate a complete staged or published dataset release."""

    release = Path(path).resolve()
    manifest_path = release / DATASET_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetReleaseError(f"Could not read dataset manifest: {error}") from error

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise DatasetReleaseError("Manifest artifacts must be a list.")
    manifest_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise DatasetReleaseError("Each manifest artifact must be an object.")
        relative = _safe_artifact_path(item.get("path"))
        relative_text = relative.as_posix()
        if relative_text in manifest_paths:
            raise DatasetReleaseError(f"Duplicate artifact path: {relative_text}")
        manifest_paths.add(relative_text)
        artifact_path = release.joinpath(*relative.parts)
        if not artifact_path.is_file():
            raise DatasetReleaseError(f"Missing artifact: {relative_text}")
        if item.get("size_bytes") != artifact_path.stat().st_size:
            raise DatasetReleaseError(f"Artifact size mismatch: {relative_text}")
        if item.get("sha256") != sha256_file(artifact_path):
            raise DatasetReleaseError(f"Artifact checksum mismatch: {relative_text}")

    actual_paths = {
        item.relative_to(release).as_posix()
        for item in release.rglob("*")
        if item.is_file() and item.name != DATASET_MANIFEST
    }
    if actual_paths != manifest_paths:
        raise DatasetReleaseError("Manifest artifact list does not match release files.")
    required_artifacts = set(REQUIRED_ARTIFACTS)
    if manifest.get("dataset_contract_version") == "0.2.0":
        required_artifacts.update(V0_2_REQUIRED_ARTIFACTS)
    if not required_artifacts.issubset(manifest_paths):
        missing = sorted(required_artifacts - manifest_paths)
        raise DatasetReleaseError(f"Required artifacts are missing: {missing}")

    npz_paths = sorted(path for path in manifest_paths if path.startswith("signals/") and path.endswith(".npz"))
    accepted = manifest.get("counts", {}).get("accepted") if isinstance(manifest.get("counts"), dict) else None
    if accepted != len(npz_paths):
        raise DatasetReleaseError("Accepted count does not match NPZ artifact count.")
    for relative_text in npz_paths:
        with np.load(release / relative_text, allow_pickle=False) as payload:
            if set(payload.files) != {"time", "af", "sf", "axialf"}:
                raise DatasetReleaseError(f"Unexpected NPZ arrays: {relative_text}")
            arrays = [payload[name] for name in ("time", "af", "sf", "axialf")]
            if (
                any(array.dtype != np.dtype("float64") or array.ndim != 1 or array.size == 0 for array in arrays)
                or len({array.size for array in arrays}) != 1
            ):
                raise DatasetReleaseError(f"Invalid NPZ signal schema: {relative_text}")
    if manifest.get("dataset_contract_version") == "0.2.0":
        assert isinstance(accepted, int)
        _validate_v0_2_release(release, manifest, accepted)
    return manifest


def prepare_dataset(
    config: DatasetPreparationConfig, *, release_id: str | None = None
) -> PublishedDataset:
    """Build, validate and publish one immutable dataset release."""

    identifier = release_id or uuid4().hex
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in identifier):
        raise ValueError("release_id contains unsupported characters.")

    destination = config.destination
    releases = destination / "releases"
    release_directory = releases / identifier
    stage = destination / f".stage-{identifier}"
    current_pointer = destination / "current-dataset.json"
    if release_directory.exists():
        raise FileExistsError(f"Dataset release already exists: {release_directory}")
    if stage.exists():
        raise FileExistsError(f"Dataset stage already exists: {stage}")

    destination.mkdir(parents=True, exist_ok=True)
    releases.mkdir(exist_ok=True)
    result: DatasetBuildResult | None = None
    renamed = False
    try:
        result = build_dataset_stage(config, stage)
        _write_manifest(
            stage,
            release_id=identifier,
            config=config,
            result=result,
        )
        validate_dataset_release(stage)
        stage.replace(release_directory)
        renamed = True

        pointer_payload = {
            "release_id": identifier,
            "release_directory": f"releases/{identifier}",
            "manifest": f"releases/{identifier}/{DATASET_MANIFEST}",
        }
        pointer_stage = destination / f".current-dataset-{identifier}.tmp"
        pointer_stage.write_text(
            json.dumps(pointer_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pointer_stage.replace(current_pointer)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if renamed and release_directory.exists():
            shutil.rmtree(release_directory)
        raise

    assert result is not None
    return PublishedDataset(
        release_id=identifier,
        release_directory=release_directory,
        manifest_path=release_directory / DATASET_MANIFEST,
        current_pointer=current_pointer,
        result=result,
    )
