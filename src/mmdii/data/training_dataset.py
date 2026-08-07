"""Load Dataset v0.2 and create leakage-safe weld observations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import resample_poly

from .dataset_publication import DatasetReleaseError, validate_dataset_release


FORCE_CHANNELS = ("af", "sf", "axialf")


@dataclass(frozen=True)
class WeldRecord:
    sample_id: str
    weld_id: str
    image_group: str
    fold: int
    target: tuple[float, ...]
    defect_codes: tuple[str, ...]
    is_normal: bool
    signal_path: Path
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class WindowedSignal:
    windows: np.ndarray
    window_mask: np.ndarray
    sample_mask: np.ndarray
    starts_seconds: tuple[float, ...]


@dataclass(frozen=True)
class WindowSpec:
    target_fs: float
    window_seconds: float
    stride_seconds: float

    def __post_init__(self) -> None:
        values = (self.target_fs, self.window_seconds, self.stride_seconds)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Window parameters must be positive and finite.")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride_seconds must not exceed window_seconds.")


@dataclass(frozen=True)
class FullSignalSpec:
    target_fs: float
    output_samples: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fs) or self.target_fs <= 0:
            raise ValueError("target_fs must be positive and finite.")
        if self.output_samples < 2:
            raise ValueError("output_samples must be at least 2.")

    def transform(self, signal: np.ndarray, time: np.ndarray) -> WindowedSignal:
        values, timestamps = _validated_signal(signal, time)
        source_fs = _source_fs(timestamps)
        resampled = resample_signal(values, source_fs, self.target_fs)
        source_positions = np.linspace(0.0, 1.0, resampled.shape[1])
        target_positions = np.linspace(0.0, 1.0, self.output_samples)
        transformed = np.vstack(
            [np.interp(target_positions, source_positions, channel) for channel in resampled]
        )
        return WindowedSignal(
            windows=transformed[np.newaxis, :, :],
            window_mask=np.ones(1, dtype=bool),
            sample_mask=np.ones((1, self.output_samples), dtype=bool),
            starts_seconds=(0.0,),
        )


class DatasetIndex:
    def __init__(
        self,
        release_directory: Path,
        target_codes: tuple[str, ...],
        records: tuple[WeldRecord, ...],
    ) -> None:
        self.release_directory = release_directory
        self.target_codes = target_codes
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_release(
        cls,
        release_directory: str | Path,
        target_codes: Iterable[str],
    ) -> "DatasetIndex":
        release = Path(release_directory).resolve()
        manifest = validate_dataset_release(release)
        if manifest.get("dataset_contract_version") != "0.2.0":
            raise DatasetReleaseError("Training requires Dataset v0.2.0.")
        targets = tuple(target_codes)
        if not targets or any(not code for code in targets) or len(targets) != len(set(targets)):
            raise ValueError("target_codes must be non-empty and unique.")

        samples = _read_csv(release / "samples.csv")
        labels = _read_csv(release / "sample_labels.csv")
        folds = _read_csv(release / "folds.csv")
        sample_rows = _unique_by_sample(samples, "samples.csv")
        label_rows = _unique_by_sample(labels, "sample_labels.csv")
        fold_rows = _unique_by_sample(folds, "folds.csv")
        if set(sample_rows) != set(label_rows) or set(sample_rows) != set(fold_rows):
            raise DatasetReleaseError("Training CSV sample IDs do not match.")

        records = []
        group_folds: dict[str, set[int]] = {}
        for sample_id in sorted(sample_rows):
            sample = sample_rows[sample_id]
            label = label_rows[sample_id]
            fold_row = fold_rows[sample_id]
            if (
                label["weld_id"] != fold_row["weld_id"]
                or label["image_group"] != fold_row["image_group"]
            ):
                raise DatasetReleaseError("Fold metadata does not match labels.")
            try:
                defect_codes_value = json.loads(label["defect_codes_json"])
                fold = int(fold_row["fold"])
            except (json.JSONDecodeError, ValueError) as error:
                raise DatasetReleaseError("Invalid training label or fold value.") from error
            if not isinstance(defect_codes_value, list) or any(
                not isinstance(code, str) or not code for code in defect_codes_value
            ):
                raise DatasetReleaseError("Invalid defect code list.")
            defect_codes = tuple(defect_codes_value)
            group_folds.setdefault(label["image_group"], set()).add(fold)
            records.append(
                WeldRecord(
                    sample_id=sample_id,
                    weld_id=label["weld_id"],
                    image_group=label["image_group"],
                    fold=fold,
                    target=target_vector(defect_codes, targets),
                    defect_codes=defect_codes,
                    is_normal=label["is_normal"] == "true",
                    signal_path=release / "signals" / f"{sample_id}.npz",
                    metadata=tuple(sorted(sample.items())),
                )
            )
        if any(len(fold_values) != 1 for fold_values in group_folds.values()):
            raise DatasetReleaseError("An image group crosses folds.")
        return cls(release, targets, tuple(records))

    def load_signal(self, record: WeldRecord) -> tuple[np.ndarray, np.ndarray]:
        with np.load(record.signal_path, allow_pickle=False) as payload:
            time = np.asarray(payload["time"], dtype=np.float64)
            signal = np.vstack(
                [np.asarray(payload[channel], dtype=np.float64) for channel in FORCE_CHANNELS]
            )
        return _validated_signal(signal, time)


@dataclass(frozen=True)
class FoldNormalizer:
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def from_arrays(cls, arrays: Iterable[np.ndarray]) -> "FoldNormalizer":
        values = tuple(np.asarray(array, dtype=np.float64) for array in arrays)
        if not values:
            raise ValueError("At least one training signal is required.")
        channel_count = values[0].shape[0]
        if any(array.ndim != 2 or array.shape[0] != channel_count for array in values):
            raise ValueError("Training signals must have a common channel count.")
        joined = np.concatenate(values, axis=1)
        means = joined.mean(axis=1)
        stds = joined.std(axis=1)
        stds = np.where(stds == 0.0, 1.0, stds)
        return cls(means=means, stds=stds)

    @classmethod
    def fit(
        cls, index: DatasetIndex, records: Iterable[WeldRecord]
    ) -> "FoldNormalizer":
        return cls.from_arrays(index.load_signal(record)[0] for record in records)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.means.size:
            raise ValueError("Signal channel count does not match normalizer.")
        return (values - self.means[:, np.newaxis]) / self.stds[:, np.newaxis]


class WeldWindowDataset:
    """Materialize one normalized bag of windows per weld on demand."""

    def __init__(
        self,
        index: DatasetIndex,
        folds: Iterable[int],
        spec: WindowSpec | FullSignalSpec,
        normalizer: FoldNormalizer,
    ) -> None:
        requested_folds = frozenset(folds)
        if not requested_folds:
            raise ValueError("At least one fold is required.")
        self.index = index
        self.records = tuple(
            record for record in index.records if record.fold in requested_folds
        )
        self.spec = spec
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> dict[str, object]:
        record = self.records[position]
        signal, time = self.index.load_signal(record)
        normalized = self.normalizer.transform(signal)
        if isinstance(self.spec, FullSignalSpec):
            windowed = self.spec.transform(normalized, time)
        else:
            windowed = window_signal(normalized, time, self.spec)
        return {
            "windows": windowed.windows,
            "window_mask": windowed.window_mask,
            "sample_mask": windowed.sample_mask,
            "starts_seconds": windowed.starts_seconds,
            "targets": np.asarray(record.target, dtype=np.float32),
            "sample_id": record.sample_id,
            "weld_id": record.weld_id,
            "image_group": record.image_group,
            "fold": record.fold,
        }


def collate_weld_batch(items: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = tuple(items)
    if not rows:
        raise ValueError("Cannot collate an empty weld batch.")
    windows = [np.asarray(row["windows"], dtype=np.float32) for row in rows]
    channels = windows[0].shape[1]
    samples = windows[0].shape[2]
    if any(
        values.ndim != 3 or values.shape[1:] != (channels, samples)
        for values in windows
    ):
        raise ValueError("All weld bags must have common channel and sample dimensions.")
    max_windows = max(values.shape[0] for values in windows)
    batch_windows = np.zeros(
        (len(rows), max_windows, channels, samples), dtype=np.float32
    )
    batch_window_mask = np.zeros((len(rows), max_windows), dtype=bool)
    batch_sample_mask = np.zeros((len(rows), max_windows, samples), dtype=bool)
    starts = np.full((len(rows), max_windows), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        count = windows[row_index].shape[0]
        batch_windows[row_index, :count] = windows[row_index]
        batch_window_mask[row_index, :count] = np.asarray(
            row["window_mask"], dtype=bool
        )
        batch_sample_mask[row_index, :count] = np.asarray(
            row["sample_mask"], dtype=bool
        )
        starts[row_index, :count] = np.asarray(row["starts_seconds"], dtype=np.float64)
    return {
        "windows": batch_windows,
        "window_mask": batch_window_mask,
        "sample_mask": batch_sample_mask,
        "starts_seconds": starts,
        "targets": np.stack([np.asarray(row["targets"], dtype=np.float32) for row in rows]),
        "sample_ids": tuple(str(row["sample_id"]) for row in rows),
        "weld_ids": tuple(str(row["weld_id"]) for row in rows),
        "image_groups": tuple(str(row["image_group"]) for row in rows),
        "folds": tuple(int(row["fold"]) for row in rows),
    }


def target_vector(
    defect_codes: Iterable[str], target_codes: Iterable[str]
) -> tuple[float, ...]:
    defects = set(defect_codes)
    return tuple(1.0 if code in defects else 0.0 for code in target_codes)


def resample_signal(
    signal: np.ndarray, source_fs: float, target_fs: float
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Signal must be a finite [channels, samples] array.")
    if any(not math.isfinite(value) or value <= 0 for value in (source_fs, target_fs)):
        raise ValueError("Sampling rates must be positive and finite.")
    if math.isclose(source_fs, target_fs, rel_tol=1e-12, abs_tol=1e-12):
        return values.copy()
    ratio = Fraction(target_fs / source_fs).limit_denominator(10000)
    return np.asarray(
        resample_poly(values, ratio.numerator, ratio.denominator, axis=1),
        dtype=np.float64,
    )


def window_signal(
    signal: np.ndarray, time: np.ndarray, spec: WindowSpec
) -> WindowedSignal:
    values, timestamps = _validated_signal(signal, time)
    resampled = resample_signal(values, _source_fs(timestamps), spec.target_fs)
    window_samples = int(round(spec.window_seconds * spec.target_fs))
    stride_samples = int(round(spec.stride_seconds * spec.target_fs))
    if window_samples < 1 or stride_samples < 1:
        raise ValueError("Window and stride must contain at least one sample.")

    if resampled.shape[1] < window_samples:
        windows = np.zeros((1, resampled.shape[0], window_samples), dtype=np.float64)
        windows[0, :, : resampled.shape[1]] = resampled
        sample_mask = np.zeros((1, window_samples), dtype=bool)
        sample_mask[0, : resampled.shape[1]] = True
        starts = (0.0,)
    else:
        start_samples = tuple(
            range(0, resampled.shape[1] - window_samples + 1, stride_samples)
        )
        windows = np.stack(
            [resampled[:, start : start + window_samples] for start in start_samples]
        )
        sample_mask = np.ones((len(start_samples), window_samples), dtype=bool)
        starts = tuple(start / spec.target_fs for start in start_samples)
    return WindowedSignal(
        windows=windows,
        window_mask=np.ones(windows.shape[0], dtype=bool),
        sample_mask=sample_mask,
        starts_seconds=starts,
    )


def _validated_signal(
    signal: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64)
    timestamps = np.asarray(time, dtype=np.float64)
    if (
        values.ndim != 2
        or timestamps.ndim != 1
        or values.shape[1] != timestamps.size
        or timestamps.size < 2
        or not np.isfinite(values).all()
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Signal and time arrays are invalid.")
    return values, timestamps


def _source_fs(time: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(time)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise DatasetReleaseError(f"Could not read {path.name}: {error}") from error


def _unique_by_sample(
    rows: Iterable[dict[str, str]], name: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in result:
            raise DatasetReleaseError(f"{name} contains invalid or duplicate sample IDs.")
        result[sample_id] = row
    return result
