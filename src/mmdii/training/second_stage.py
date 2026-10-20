"""Leak-safe second-stage trees over weld-level features.

Three feature sources are supported so that the deep representation can be
ablated against the hand-crafted statistics:

``stats``
    18 whole-signal statistics, no encoder involved.  This arm is the honest
    reference because it is the same model as the tabular control run.
``embedding``
    the frozen weld-level deep representation.
``both``
    the statistics concatenated with the representation.

The training features for the deep arms must be out-of-sample, otherwise the
tree is fitted on representations that already saw their own labels and the
result is optimistic.  ``harvest_out_of_sample_representations`` therefore
trains one throwaway encoder per inner partition of the training folds and
embeds only the held-out inner partition.  No inner model ever sees the outer
held-out fold, and the tree never sees an in-sample representation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from mmdii.evaluation.multilabel import evaluate_multilabel
from mmdii.models.statistical import extract_statistical_features


FEATURE_SOURCES = ("stats", "embedding", "both")
RunDeep = Callable[..., tuple[np.ndarray, dict[str, object]]]
FitTree = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def inner_partitions(fold_ids: Iterable[int], count: int) -> tuple[tuple[int, ...], ...]:
    """Split training fold identifiers into ``count`` groups used as inner holdouts.

    Each returned group is the inner validation set for one throwaway encoder;
    everything else in the training folds is that encoder's training set.  The
    groups are interleaved so they stay as evenly sized as possible.
    """

    folds = tuple(sorted({int(value) for value in fold_ids}))
    if count < 2:
        raise ValueError("inner_folds must be at least 2.")
    if count > len(folds):
        raise ValueError("inner_folds cannot exceed the number of training folds.")
    groups: list[list[int]] = [[] for _ in range(count)]
    for position, fold in enumerate(folds):
        groups[position % count].append(fold)
    if any(not group for group in groups):
        raise ValueError("Every inner partition must contain at least one fold.")
    return tuple(tuple(group) for group in groups)


def statistical_feature_matrix(index: Any, records: Sequence[Any]) -> np.ndarray:
    """Whole-signal statistics in record order, matching the tabular control run."""

    return np.stack(
        [extract_statistical_features(index.load_signal(record)[0]) for record in records]
    )


def harvest_out_of_sample_representations(
    train_records: Sequence[Any],
    *,
    fold: int,
    partitions: Sequence[Sequence[int]],
    run_deep: RunDeep,
) -> np.ndarray:
    """Collect one out-of-sample weld representation per training weld.

    ``run_deep(train_subset, valid_subset, harvest_embeddings=True)`` must train
    a fresh encoder on ``train_subset`` and return the pooled representations of
    ``valid_subset`` under the key ``weld_representations``.
    """

    train_fold_ids = {int(record.fold) for record in train_records}
    collected: dict[str, np.ndarray] = {}
    for group in partitions:
        group_ids = tuple(int(value) for value in group)
        if fold in group_ids:
            raise ValueError(
                f"Inner partition {group_ids} contains the outer held-out fold {fold}."
            )
        if not set(group_ids) <= train_fold_ids:
            raise ValueError(
                f"Inner partition {group_ids} is not a subset of the training folds "
                f"{sorted(train_fold_ids)}."
            )
        inner_train = tuple(r for r in train_records if r.fold not in group_ids)
        inner_valid = tuple(r for r in train_records if r.fold in group_ids)
        if not inner_train or not inner_valid:
            raise ValueError(f"Inner partition {group_ids} leaves an empty side.")
        _, info = run_deep(inner_train, inner_valid, harvest_embeddings=True)
        representations = np.asarray(info["weld_representations"], dtype=np.float64)
        sample_ids = tuple(str(value) for value in info["sample_ids"])
        if representations.shape[0] != len(inner_valid) or len(sample_ids) != len(inner_valid):
            raise ValueError("Harvested representations do not match the inner holdout size.")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Harvested representations contain duplicate sample ids.")
        expected = {record.sample_id for record in inner_valid}
        if set(sample_ids) != expected:
            raise ValueError("Harvested representations do not match the inner holdout welds.")
        for sample_id, representation in zip(sample_ids, representations, strict=True):
            collected[sample_id] = representation

    missing = [record.sample_id for record in train_records if record.sample_id not in collected]
    if missing:
        raise ValueError(
            f"No out-of-sample representation for {len(missing)} training welds "
            f"(for example {missing[:3]})."
        )
    return np.stack([collected[record.sample_id] for record in train_records])


def assemble_features(
    statistics: np.ndarray | None,
    representations: np.ndarray | None,
    feature_source: str,
    *,
    representation_shared: bool,
) -> np.ndarray:
    """Build the tree feature matrix for one arm."""

    if feature_source not in FEATURE_SOURCES:
        raise ValueError(f"feature_source must be one of {', '.join(FEATURE_SOURCES)}.")
    blocks: list[np.ndarray] = []
    if feature_source in {"stats", "both"}:
        if statistics is None:
            raise ValueError("Statistics are required for this feature source.")
        blocks.append(np.asarray(statistics, dtype=np.float64))
    if feature_source in {"embedding", "both"}:
        if representations is None:
            raise ValueError("Representations are required for this feature source.")
        values = np.asarray(representations, dtype=np.float64)
        if values.ndim == 3:
            # [n, targets, dim]; mean aggregation repeats one vector per target.
            values = values[:, 0, :] if representation_shared else values.reshape(values.shape[0], -1)
        if values.ndim != 2:
            raise ValueError("Representations must be two- or three-dimensional.")
        blocks.append(values)
    if not blocks:
        raise ValueError("No feature block was assembled.")
    return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def project_embedding_features(
    train_representations: np.ndarray,
    valid_representations: np.ndarray,
    *,
    projection: str,
    components: int,
    representation_shared: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit optional PCA on outer-training OOS embeddings and transform both sides."""

    train_values = np.asarray(train_representations, dtype=np.float64)
    valid_values = np.asarray(valid_representations, dtype=np.float64)
    if train_values.ndim == 3:
        train_values = (
            train_values[:, 0, :]
            if representation_shared
            else train_values.reshape(train_values.shape[0], -1)
        )
    if valid_values.ndim == 3:
        valid_values = (
            valid_values[:, 0, :]
            if representation_shared
            else valid_values.reshape(valid_values.shape[0], -1)
        )
    if (
        train_values.ndim != 2
        or valid_values.ndim != 2
        or train_values.shape[1] != valid_values.shape[1]
        or not np.isfinite(train_values).all()
        or not np.isfinite(valid_values).all()
    ):
        raise ValueError("Embedding matrices must be finite, two-dimensional and aligned.")
    if projection == "none":
        return train_values, valid_values, {
            "projection": projection,
            "input_dimension": int(train_values.shape[1]),
            "output_dimension": int(train_values.shape[1]),
        }
    if projection != "pca":
        raise ValueError("embedding_projection must be none or pca.")
    if components < 1 or components > min(train_values.shape):
        raise ValueError(
            "embedding_components must be between 1 and the smaller training matrix dimension."
        )
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("PCA requires the MMDII-Core train extra.") from error
    model = PCA(n_components=components, svd_solver="full")
    train_projected = model.fit_transform(train_values)
    valid_projected = model.transform(valid_values)
    return train_projected, valid_projected, {
        "projection": projection,
        "input_dimension": int(train_values.shape[1]),
        "output_dimension": int(train_projected.shape[1]),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
    }


def run_second_stage_fold(
    *,
    index: Any,
    train_records: Sequence[Any],
    valid_records: Sequence[Any],
    config: Any,
    fold: int,
    run_deep: RunDeep,
    fit_tree: FitTree,
    representation_shared: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the second-stage tree for one outer fold and return its probabilities."""

    source = config.second_stage.feature_source
    train_targets = np.asarray([record.target for record in train_records], dtype=np.float64)
    if source == "stats":
        train_features = statistical_feature_matrix(index, train_records)
        valid_features = statistical_feature_matrix(index, valid_records)
        return fit_tree(train_features, train_targets, valid_features), {
            "second_stage": {
                "feature_source": source,
                "feature_count": int(train_features.shape[1]),
                "inner_folds": 0,
                "embedding_projection": {
                    "projection": "none",
                    "input_dimension": 0,
                    "output_dimension": 0,
                },
                "representation_dim": 0,
                "representation_shared": False,
            }
        }

    partitions = inner_partitions(
        sorted({int(record.fold) for record in train_records}),
        int(config.second_stage.inner_folds),
    )
    train_representations = harvest_out_of_sample_representations(
        train_records, fold=fold, partitions=partitions, run_deep=run_deep
    )
    deep_probabilities, info = run_deep(train_records, valid_records, harvest_embeddings=True)
    valid_representations = np.asarray(info.pop("weld_representations"), dtype=np.float64)
    harvested_order = tuple(str(value) for value in info.pop("sample_ids"))
    expected_order = tuple(record.sample_id for record in valid_records)
    if harvested_order != expected_order:
        raise ValueError(
            "Harvested validation representations are not in validation record order."
        )

    statistics_train = (
        statistical_feature_matrix(index, train_records) if source == "both" else None
    )
    statistics_valid = (
        statistical_feature_matrix(index, valid_records) if source == "both" else None
    )
    train_representations, valid_representations, projection_info = project_embedding_features(
        train_representations,
        valid_representations,
        projection=str(getattr(config.second_stage, "embedding_projection", "none")),
        components=int(getattr(config.second_stage, "embedding_components", 8)),
        representation_shared=representation_shared,
    )
    train_features = assemble_features(
        statistics_train,
        train_representations,
        source,
        representation_shared=representation_shared,
    )
    valid_features = assemble_features(
        statistics_valid,
        valid_representations,
        source,
        representation_shared=representation_shared,
    )
    if train_features.shape[0] != len(train_records):
        raise ValueError("Training feature matrix does not match the training welds.")
    if valid_features.shape[0] != len(valid_records):
        raise ValueError("Validation feature matrix does not match the validation welds.")

    probabilities = fit_tree(train_features, train_targets, valid_features)
    valid_targets = np.asarray([record.target for record in valid_records], dtype=np.float64)
    deep_metrics = evaluate_multilabel(
        valid_targets, deep_probabilities, target_codes=config.target_codes
    )
    info["second_stage"] = {
        "feature_source": source,
        "feature_count": int(train_features.shape[1]),
        "representation_dim": int(train_representations.shape[-1]),
        "representation_shared": bool(representation_shared),
        "embedding_projection": projection_info,
        "inner_folds": int(config.second_stage.inner_folds),
        "inner_partitions": [list(group) for group in partitions],
        "deep_macro_pr_auc": deep_metrics["macro_pr_auc"],
        "deep_per_code_pr_auc": {
            code: report["pr_auc"] for code, report in deep_metrics["per_code"].items()
        },
    }
    return probabilities, info
