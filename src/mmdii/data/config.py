"""Read a raw dataset location without depending on ML libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataSourceConfig:
    """Resolved read-only MATLAB source configuration."""

    root: Path
    format: str
    read_only: bool


def load_data_source(config_path: str | Path) -> DataSourceConfig:
    """Load and resolve the source directory declared in a TOML config."""

    path = Path(config_path).resolve()
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)

    source = payload["source"]
    return DataSourceConfig(
        root=(path.parent / source["root"]).resolve(),
        format=source["format"],
        read_only=source["read_only"],
    )


def iter_mat_files(source: DataSourceConfig) -> list[Path]:
    """Return stable source-file ordering and fail clearly for a bad path."""

    if not source.root.is_dir():
        raise FileNotFoundError(f"MAT source directory does not exist: {source.root}")
    if source.format.lower() != "mat":
        raise ValueError(f"Unsupported source format: {source.format}")
    return sorted(source.root.glob("*.mat"))
