"""Read MATLAB variable metadata without loading array contents."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from scipy.io import whosmat


FORMAT_VERSION = 1


def build_header_coverage(file_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize field-to-file and file-to-field coverage for successful records."""

    successful_files = sorted(
        (record for record in file_records if "variables" in record),
        key=lambda record: record["path"],
    )
    variable_names = sorted(
        {
            variable["name"]
            for record in successful_files
            for variable in record["variables"]
        }
    )
    successful_file_count = len(successful_files)
    fields: list[dict[str, Any]] = []

    for name in variable_names:
        paths = [
            record["path"]
            for record in successful_files
            if name in {variable["name"] for variable in record["variables"]}
        ]
        fields.append(
            {
                "name": name,
                "file_count": len(paths),
                "coverage_percent": round(
                    100 * len(paths) / successful_file_count, 1
                ),
                "files": paths,
            }
        )

    total_distinct_variables = len(variable_names)
    file_coverage: list[dict[str, Any]] = []
    for record in successful_files:
        present_variables = {
            variable["name"] for variable in record["variables"]
        }
        missing_variables = [
            name for name in variable_names if name not in present_variables
        ]
        file_coverage.append(
            {
                "path": record["path"],
                "present_variable_count": len(present_variables),
                "total_distinct_variables": total_distinct_variables,
                "coverage_percent": round(
                    100 * len(present_variables) / total_distinct_variables, 1
                )
                if total_distinct_variables
                else 100.0,
                "missing_variables": missing_variables,
            }
        )

    return {
        "total_distinct_variables": total_distinct_variables,
        "successful_file_count": successful_file_count,
        "universal_variables": [
            field["name"]
            for field in fields
            if field["file_count"] == successful_file_count
        ],
        "unique_variables": [
            {"name": field["name"], "file": field["files"][0]}
            for field in fields
            if field["file_count"] == 1
        ],
        "fully_covered_files": [
            item["path"]
            for item in file_coverage
            if item["present_variable_count"] == total_distinct_variables
        ],
        "fields": fields,
        "files": file_coverage,
    }


def inspect_mat_file(path: str | Path, source_root: str | Path) -> dict[str, Any]:
    """Return JSON-safe variable metadata for one MAT file.

    The source root is used only to give the record a stable relative path.
    Any MAT parsing exception is intentionally propagated so directory scans can
    record it against the individual file and continue with the remaining files.
    """

    root = Path(source_root).resolve()
    mat_path = Path(path).resolve()
    relative_path = mat_path.relative_to(root).as_posix()
    variables = [
        {
            "name": name,
            "matlab_class": matlab_class,
            "shape": list(shape),
        }
        for name, shape, matlab_class in whosmat(mat_path)
    ]
    variables.sort(
        key=lambda variable: (
            variable["name"],
            variable["matlab_class"],
            variable["shape"],
        )
    )
    return {"path": relative_path, "variables": variables}


def inspect_mat_directory(source_root: str | Path) -> dict[str, Any]:
    """Inspect every MAT file under a source root without enforcing a schema."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MAT source directory does not exist: {root}")

    files: list[dict[str, Any]] = []
    for mat_path in sorted(root.glob("*.mat"), key=lambda path: path.name.casefold()):
        try:
            files.append(inspect_mat_file(mat_path, root))
        except Exception as error:
            files.append(
                {
                    "path": mat_path.relative_to(root).as_posix(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    files.sort(key=lambda record: record["path"])
    successful_files = [record for record in files if "variables" in record]
    variable_frequency = Counter(
        variable["name"]
        for record in successful_files
        for variable in record["variables"]
    )
    schema_frequency: Counter[tuple[tuple[str, str], ...]] = Counter(
        tuple(
            (variable["name"], variable["matlab_class"])
            for variable in record["variables"]
        )
        for record in successful_files
    )

    return {
        "format_version": FORMAT_VERSION,
        "source_root": root.as_posix(),
        "inspected_file_count": len(files),
        "successful_file_count": len(successful_files),
        "failed_file_count": len(files) - len(successful_files),
        "variable_frequency": [
            {"name": name, "file_count": count}
            for name, count in sorted(variable_frequency.items())
        ],
        "schema_groups": [
            {
                "variables": [
                    {"name": name, "matlab_class": matlab_class}
                    for name, matlab_class in schema
                ],
                "file_count": count,
            }
            for schema, count in sorted(schema_frequency.items())
        ],
        "coverage": build_header_coverage(files),
        "files": files,
    }
