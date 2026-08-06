"""Assign deterministic multi-label folds without splitting image groups."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LabelledSample:
    sample_id: str
    weld_id: str
    image_group: str
    is_normal: bool
    defect_codes: tuple[str, ...]


@dataclass(frozen=True)
class FoldAssignment:
    sample_id: str
    weld_id: str
    image_group: str
    fold: int


def _validated_samples(samples: Iterable[LabelledSample]) -> tuple[LabelledSample, ...]:
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    if not ordered:
        raise ValueError("At least one labelled sample is required.")
    sample_ids = [sample.sample_id for sample in ordered]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be non-empty and unique.")
    for sample in ordered:
        if not sample.weld_id or not sample.image_group:
            raise ValueError("weld_id and image_group must not be empty.")
        if len(sample.defect_codes) != len(set(sample.defect_codes)):
            raise ValueError("defect_codes must be unique.")
        if sample.is_normal != (len(sample.defect_codes) == 0):
            raise ValueError("Normal/defect semantics are inconsistent.")
    return ordered


def _sample_labels(sample: LabelledSample) -> tuple[str, ...]:
    return ("normal",) if sample.is_normal else sample.defect_codes


def assign_grouped_folds(
    samples: Iterable[LabelledSample], fold_count: int
) -> tuple[FoldAssignment, ...]:
    """Greedily balance labels and sample counts while keeping images intact."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2.")
    ordered = _validated_samples(samples)
    groups: dict[str, list[LabelledSample]] = {}
    for sample in ordered:
        groups.setdefault(sample.image_group, []).append(sample)
    if len(groups) < fold_count:
        raise ValueError("Cannot create folds from fewer image groups than folds.")

    group_labels = {
        name: Counter(
            label for sample in members for label in _sample_labels(sample)
        )
        for name, members in groups.items()
    }
    labels = sorted({label for counts in group_labels.values() for label in counts})
    positive_group_counts = {
        label: sum(label in counts for counts in group_labels.values())
        for label in labels
    }
    ordered_groups = sorted(
        groups,
        key=lambda name: (
            -sum(
                group_labels[name][label] / positive_group_counts[label]
                for label in group_labels[name]
            ),
            -len(groups[name]),
            name,
        ),
    )

    total_samples = len(ordered)
    total_labels = Counter(label for sample in ordered for label in _sample_labels(sample))
    fold_samples = [0] * fold_count
    fold_labels = [Counter() for _ in range(fold_count)]
    group_fold: dict[str, int] = {}

    def assignment_score(candidate: int, name: str) -> float:
        sample_target = total_samples / fold_count
        score = 0.0
        for fold in range(fold_count):
            sample_count = fold_samples[fold]
            if fold == candidate:
                sample_count += len(groups[name])
            score += ((sample_count - sample_target) / max(sample_target, 1.0)) ** 2
            for label in labels:
                target = total_labels[label] / fold_count
                count = fold_labels[fold][label]
                if fold == candidate:
                    count += group_labels[name][label]
                score += ((count - target) / max(target, 1.0)) ** 2
        return score

    for index, name in enumerate(ordered_groups):
        if index < fold_count:
            fold = index
        else:
            fold = min(
                range(fold_count),
                key=lambda candidate: (assignment_score(candidate, name), candidate),
            )
        group_fold[name] = fold
        fold_samples[fold] += len(groups[name])
        fold_labels[fold].update(group_labels[name])

    return tuple(
        FoldAssignment(
            sample_id=sample.sample_id,
            weld_id=sample.weld_id,
            image_group=sample.image_group,
            fold=group_fold[sample.image_group],
        )
        for sample in ordered
    )


def build_split_report(
    samples: Iterable[LabelledSample],
    assignments: Iterable[FoldAssignment],
    fold_count: int,
) -> dict[str, object]:
    """Summarize fold balance and labels that cannot cover every fold."""

    ordered = _validated_samples(samples)
    assignment_rows = tuple(assignments)
    by_sample = {row.sample_id: row for row in assignment_rows}
    if set(by_sample) != {sample.sample_id for sample in ordered}:
        raise ValueError("Fold assignments must match labelled sample IDs.")
    if len(by_sample) != len(assignment_rows):
        raise ValueError("Fold assignments contain duplicate sample IDs.")
    if any(row.fold < 0 or row.fold >= fold_count for row in assignment_rows):
        raise ValueError("Fold assignment is outside the configured range.")

    group_folds: dict[str, set[int]] = {}
    for row in assignment_rows:
        group_folds.setdefault(row.image_group, set()).add(row.fold)
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("An image group was assigned to multiple folds.")

    labels = sorted({label for sample in ordered for label in _sample_labels(sample)})
    folds = []
    for fold in range(fold_count):
        fold_samples = [sample for sample in ordered if by_sample[sample.sample_id].fold == fold]
        counts = Counter(label for sample in fold_samples for label in _sample_labels(sample))
        folds.append(
            {
                "fold": fold,
                "sample_count": len(fold_samples),
                "group_count": len({sample.image_group for sample in fold_samples}),
                "normal_count": counts["normal"],
                "defect_counts": {
                    label: counts[label] for label in labels if label != "normal"
                },
            }
        )

    positive_group_counts = {
        label: len(
            {
                sample.image_group
                for sample in ordered
                if label in _sample_labels(sample)
            }
        )
        for label in labels
    }
    warnings = [
        {
            "code": "insufficient_positive_groups",
            "label": label,
            "positive_group_count": positive_group_counts[label],
            "fold_count": fold_count,
        }
        for label in labels
        if positive_group_counts[label] < fold_count
    ]
    return {
        "fold_count": fold_count,
        "folds": folds,
        "label_positive_group_counts": positive_group_counts,
        "warnings": warnings,
    }
