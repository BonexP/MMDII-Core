"""Load project-supplied configuration for dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .config import DataSourceConfig, load_data_source
from .signal_quality import SignalQualityConfig


@dataclass(frozen=True)
class PreprocessingConfig:
    target_fs: float | str
    window_seconds: float
    stride_seconds: float
    normalization: str
    spectral_energy_fraction: float
    spectral_record_percentile: float
    nyquist_margin: float


@dataclass(frozen=True)
class SplitConfig:
    fold_count: int
    group_field: str


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
    preprocessing: PreprocessingConfig | None
    splits: SplitConfig | None


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
    if contract_version not in {"0.1.0", "0.2.0"}:
        raise ValueError("Dataset contract_version must be 0.1.0 or 0.2.0.")
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

    preprocessing = None
    splits = None
    if contract_version == "0.2.0":
        try:
            preprocessing_payload = payload["preprocessing"]
            split_payload = payload["splits"]
            target_fs_value = preprocessing_payload["target_fs"]
            if target_fs_value != "auto":
                target_fs_value = float(target_fs_value)
                if target_fs_value <= 0:
                    raise ValueError("target_fs must be 'auto' or positive.")
            window_seconds = float(preprocessing_payload["window_seconds"])
            stride_seconds = float(preprocessing_payload["stride_seconds"])
            if window_seconds <= 0:
                raise ValueError("window_seconds must be positive.")
            if stride_seconds <= 0 or stride_seconds > window_seconds:
                raise ValueError(
                    "stride_seconds must be positive and no greater than window_seconds."
                )
            normalization = preprocessing_payload["normalization"]
            if normalization != "train_fold_zscore":
                raise ValueError("normalization must be train_fold_zscore.")
            energy_fraction = float(
                preprocessing_payload["spectral_energy_fraction"]
            )
            record_percentile = float(
                preprocessing_payload["spectral_record_percentile"]
            )
            nyquist_margin = float(preprocessing_payload["nyquist_margin"])
            if not 0 < energy_fraction <= 1:
                raise ValueError("spectral_energy_fraction must be in (0, 1].")
            if not 0 < record_percentile <= 1:
                raise ValueError("spectral_record_percentile must be in (0, 1].")
            if nyquist_margin < 1:
                raise ValueError("nyquist_margin must be at least 1.")
            fold_count = split_payload["fold_count"]
            group_field = split_payload["group_field"]
            if fold_count != 5:
                raise ValueError("fold_count must be 5.")
            if group_field != "image_relative_path":
                raise ValueError("group_field must be image_relative_path.")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid v0.2 training configuration: {error}") from error
        preprocessing = PreprocessingConfig(
            target_fs=target_fs_value,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            normalization=normalization,
            spectral_energy_fraction=energy_fraction,
            spectral_record_percentile=record_percentile,
            nyquist_margin=nyquist_margin,
        )
        splits = SplitConfig(fold_count=fold_count, group_field=group_field)

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
        preprocessing=preprocessing,
        splits=splits,
    )
