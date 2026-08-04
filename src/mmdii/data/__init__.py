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
from .weld_annotations import (
    AnnotationRelease,
    AnnotationReleaseError,
    MappingResult,
    WeldAnnotation,
    WeldMapping,
    build_weld_mappings,
    load_annotation_release,
)
from .signal_quality import SignalAudit, SignalQualityConfig, audit_primary_signals
from .dataset_config import DatasetPreparationConfig, load_dataset_config
from .dataset_builder import DatasetBuildResult, build_dataset_stage

__all__ = [
    "AnnotationRelease",
    "AnnotationReleaseError",
    "DataSourceConfig",
    "DatasetBuildResult",
    "DatasetPreparationConfig",
    "MatFilenameError",
    "MatRecord",
    "MappingResult",
    "SignalAudit",
    "SignalQualityConfig",
    "WeldAnnotation",
    "WeldMapping",
    "build_header_coverage",
    "build_dataset_stage",
    "build_weld_mappings",
    "audit_primary_signals",
    "inspect_mat_directory",
    "inspect_mat_file",
    "inventory_mat_files",
    "iter_mat_files",
    "load_data_source",
    "load_annotation_release",
    "load_dataset_config",
    "parse_mat_filename",
    "sample_id_for",
]
