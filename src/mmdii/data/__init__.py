"""Dataset source and manifest helpers."""

from .config import DataSourceConfig, iter_mat_files, load_data_source
from .mat_headers import build_header_coverage, inspect_mat_directory, inspect_mat_file
from .mat_records import (
    MatFilenameError,
    MatRecord,
    inventory_mat_files,
    parse_mat_filename,
    sample_id_for,
)

__all__ = [
    "DataSourceConfig",
    "MatFilenameError",
    "MatRecord",
    "build_header_coverage",
    "inspect_mat_directory",
    "inspect_mat_file",
    "inventory_mat_files",
    "iter_mat_files",
    "load_data_source",
    "parse_mat_filename",
    "sample_id_for",
]
