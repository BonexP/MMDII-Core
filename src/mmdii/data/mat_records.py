"""Parse immutable MATLAB experiment records without loading signal values."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re


_MAT_FILENAME = re.compile(
    r"^(?P<date>\d{4}-\d{1,2}-\d{1,2})-"
    r"(?P<depth_start>\d+(?:\.\d+)?)-"
    r"(?P<depth_end>\d+(?:\.\d+)?)-"
    r"(?P<rpm>\d+)-(?P<welding_speed>\d+)-(?P<run_id>\d+)"
    r"~(?P<duplicate_suffix>1)?\.mat$"
)


class MatFilenameError(ValueError):
    """Raised when a source filename does not match the experiment contract."""


@dataclass(frozen=True)
class MatRecord:
    """Filename-derived metadata for one source MAT file."""

    path: Path
    relative_path: str
    date: str
    depth_start: float
    depth_end: float
    rpm: int
    welding_speed: int
    run_id: int
    duplicate_suffix: str
    is_variant: bool


def parse_mat_filename(path: str | Path, source_root: str | Path) -> MatRecord:
    """Parse one configured source path into immutable experiment metadata."""

    root = Path(source_root).resolve()
    mat_path = Path(path).resolve()
    try:
        relative_path = mat_path.relative_to(root).as_posix()
    except ValueError as error:
        raise MatFilenameError(f"MAT file is outside the source root: {mat_path}") from error

    match = _MAT_FILENAME.fullmatch(mat_path.name)
    if match is None:
        raise MatFilenameError(
            f"MAT filename does not match the experiment pattern: {relative_path}"
        )

    suffix = match.group("duplicate_suffix") or ""
    return MatRecord(
        path=mat_path,
        relative_path=relative_path,
        date=match.group("date"),
        depth_start=float(match.group("depth_start")),
        depth_end=float(match.group("depth_end")),
        rpm=int(match.group("rpm")),
        welding_speed=int(match.group("welding_speed")),
        run_id=int(match.group("run_id")),
        duplicate_suffix=suffix,
        is_variant=bool(suffix),
    )


def sample_id_for(record: MatRecord) -> str:
    """Return the deterministic sample identifier for a base MAT record."""

    digest = sha256(record.relative_path.encode("utf-8")).hexdigest()[:8]
    return f"mat-{record.run_id:03d}-{digest}"


def inventory_mat_files(
    source_root: str | Path,
) -> tuple[list[MatRecord], list[dict[str, str]]]:
    """Parse every top-level MAT file and retain deterministic filename errors."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MAT source directory does not exist: {root}")

    records: list[MatRecord] = []
    errors: list[dict[str, str]] = []
    for path in sorted(root.glob("*.mat"), key=lambda item: item.name.casefold()):
        try:
            records.append(parse_mat_filename(path, root))
        except MatFilenameError as error:
            errors.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "issue_code": "invalid_filename",
                    "message": str(error),
                }
            )

    records.sort(key=lambda record: record.relative_path.casefold())
    errors.sort(key=lambda record: record["path"].casefold())
    return records, errors
