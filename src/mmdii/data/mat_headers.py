"""Read MATLAB variable metadata without loading array contents."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from scipy.io import whosmat


FORMAT_VERSION = 1


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
        "files": files,
    }
