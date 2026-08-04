"""Load project-supplied configuration for dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .config import DataSourceConfig, load_data_source
from .signal_quality import SignalQualityConfig


@dataclass(frozen=True)
class DatasetPreparationConfig:
    config_path: Path
    contract_version: str
    source_config_path: Path
    source: DataSourceConfig
    annotation_release: Path
    expected_annotation_release_id: str
    destination: Path
    mapping_source: str
    excluded_run_ids: frozenset[int]
    signal_quality: SignalQualityConfig


def _resolve(parent: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string.")
    return (parent / value).resolve()


def load_dataset_config(path: str | Path) -> DatasetPreparationConfig:
    """Load and validate a dataset preparation TOML file."""

    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        dataset = payload["dataset"]
        signals = payload["signals"]
    except KeyError as error:
        raise ValueError(f"Missing configuration section: {error.args[0]}") from error

    parent = config_path.parent
    source_config_path = _resolve(
        parent, dataset.get("raw_source_config"), "raw_source_config"
    )
    source = load_data_source(source_config_path)
    if not source.read_only:
        raise ValueError("Raw source configuration must set read_only = true.")
    if source.format.casefold() != "mat":
        raise ValueError("Raw source format must be mat.")

    contract_version = dataset.get("contract_version")
    if contract_version != "0.1.0":
        raise ValueError("Dataset contract_version must be 0.1.0.")
    mapping_source = dataset.get("mapping_source")
    if not isinstance(mapping_source, str) or not mapping_source.strip():
        raise ValueError("mapping_source must be a non-empty string.")
    expected_release_id = dataset.get("expected_annotation_release_id")
    if not isinstance(expected_release_id, str) or not expected_release_id.strip():
        raise ValueError("expected_annotation_release_id must be a non-empty string.")
    excluded_values = dataset.get("excluded_run_ids", [])
    if (
        not isinstance(excluded_values, list)
        or any(not isinstance(value, int) or value < 0 for value in excluded_values)
    ):
        raise ValueError("excluded_run_ids must be a list of non-negative integers.")

    fields = signals.get("fields")
    if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
        raise ValueError("signals.fields must be a list of four strings.")
    try:
        signal_quality = SignalQualityConfig(
            fields=tuple(fields),  # type: ignore[arg-type]
            time_step_rtol=float(signals["time_step_rtol"]),
            time_step_atol=float(signals["time_step_atol"]),
            rpm_fs_rtol=float(signals["rpm_fs_rtol"]),
            rpm_fs_atol=float(signals["rpm_fs_atol"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid signal quality configuration: {error}") from error

    return DatasetPreparationConfig(
        config_path=config_path,
        contract_version=contract_version,
        source_config_path=source_config_path,
        source=source,
        annotation_release=_resolve(
            parent, dataset.get("annotation_release"), "annotation_release"
        ),
        expected_annotation_release_id=expected_release_id,
        destination=_resolve(parent, dataset.get("destination"), "destination"),
        mapping_source=mapping_source.strip(),
        excluded_run_ids=frozenset(excluded_values),
        signal_quality=signal_quality,
    )
