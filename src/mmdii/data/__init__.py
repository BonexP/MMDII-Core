"""Dataset source and manifest helpers."""

from .config import DataSourceConfig, iter_mat_files, load_data_source
from .mat_headers import build_header_coverage, inspect_mat_directory, inspect_mat_file

__all__ = [
    "DataSourceConfig",
    "build_header_coverage",
    "inspect_mat_directory",
    "inspect_mat_file",
    "iter_mat_files",
    "load_data_source",
]
