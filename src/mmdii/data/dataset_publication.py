"""Validate and atomically publish immutable dataset releases."""

from __future__ import annotations

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
    manifest = {
        "annotation_contract_version": "1.1.0",
        "annotation_release_id": result.annotation_release_id,
        "artifacts": artifacts,
        "configuration": {
            "excluded_run_ids": sorted(config.excluded_run_ids),
            "fields": list(config.signal_quality.fields),
            "mapping_source": config.mapping_source,
            "rpm_fs_atol": config.signal_quality.rpm_fs_atol,
            "rpm_fs_rtol": config.signal_quality.rpm_fs_rtol,
            "time_step_atol": config.signal_quality.time_step_atol,
            "time_step_rtol": config.signal_quality.time_step_rtol,
        },
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
    if not REQUIRED_ARTIFACTS.issubset(manifest_paths):
        missing = sorted(REQUIRED_ARTIFACTS - manifest_paths)
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
