"""Deterministic statistical lower bound for weld signals."""

from __future__ import annotations

from typing import Iterable

import numpy as np


STATISTIC_NAMES = (
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
)


LEARNERS = ("random_forest", "extra_trees")


def extract_statistical_features(signal: np.ndarray) -> np.ndarray:
    """Return six fixed summary features for each input channel."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite [channels, samples] array.")
    rows = []
    for channel in values:
        rows.extend(
            (
                float(channel.mean()),
                float(channel.std()),
                float(np.sqrt(np.mean(np.square(channel)))),
                float(channel.min()),
                float(channel.max()),
                float(np.ptp(channel)),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def fit_predict_logistic_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
) -> np.ndarray:
    """Fit one balanced logistic model per target and return probabilities."""

    train_x = np.asarray(train_features, dtype=np.float64)
    train_y = np.asarray(train_targets, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    codes = tuple(target_codes)
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or train_y.ndim != 2
        or train_x.shape[0] != train_y.shape[0]
        or train_x.shape[1] != test_x.shape[1]
        or train_y.shape[1] != len(codes)
        or not codes
    ):
        raise ValueError("Feature, target and target-code shapes are inconsistent.")
    for position, code in enumerate(codes):
        if np.unique(train_y[:, position]).size != 2:
            raise ValueError(f"Training target {code!r} contains only one class.")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Statistical training requires the MMDII-Core train extra."
        ) from error

    probabilities = np.empty((test_x.shape[0], len(codes)), dtype=np.float64)
    for position in range(len(codes)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
        )
        model.fit(train_x, train_y[:, position])
        probabilities[:, position] = model.predict_proba(test_x)[:, 1]
    return probabilities


def fit_predict_forest_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
    learner: str = "random_forest",
    n_estimators: int = 300,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    max_depth: int | None = None,
    criterion: str = "gini",
    class_weight: str | None = "balanced",
) -> np.ndarray:
    """Fit one balanced tree ensemble per target and return probabilities.

    ``learner`` selects the ensemble family. Extra trees sample the split
    threshold at random instead of searching for the best one, which is a
    cheaper, lower-variance ensemble on the small fold-local design matrices
    used here. Both learners share every other setting so that a comparison
    isolates the ensemble family alone.
    """

    train_x = np.asarray(train_features, dtype=np.float64)
    train_y = np.asarray(train_targets, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    codes = tuple(target_codes)
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or train_y.ndim != 2
        or train_x.shape[0] != train_y.shape[0]
        or train_x.shape[1] != test_x.shape[1]
        or train_y.shape[1] != len(codes)
        or not codes
        or learner not in LEARNERS
        or n_estimators < 1
        or min_samples_leaf < 1
        or max_features not in {"sqrt", "all", None}
        or (max_depth is not None and max_depth < 1)
        or criterion not in {"gini", "log_loss"}
        or class_weight not in {"balanced", None}
    ):
        raise ValueError("Feature, target and tree-ensemble settings are invalid.")
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    except ImportError as error:
        raise RuntimeError(
            "Tree training requires the MMDII-Core train extra."
        ) from error
    estimator_class = {
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    }[learner]

    probabilities = np.empty((test_x.shape[0], len(codes)), dtype=np.float64)
    for position, code in enumerate(codes):
        target = train_y[:, position]
        if np.unique(target).size != 2:
            raise ValueError(f"Training target {code!r} contains only one class.")
        model = estimator_class(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=None if max_features == "all" else max_features,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train_x, target)
        probabilities[:, position] = model.predict_proba(test_x)[:, 1]
    return probabilities


def fit_predict_random_forest_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
    n_estimators: int = 300,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    max_depth: int | None = None,
    criterion: str = "gini",
    class_weight: str | None = "balanced",
) -> np.ndarray:
    """Fit one balanced random-forest model per target.

    This is a nonlinear, non-neural reference using exactly the same fold-local
    statistical features as B0. It is intentionally not a window model.
    """

    return fit_predict_forest_ovr(
        train_features,
        train_targets,
        test_features,
        target_codes=target_codes,
        random_state=random_state,
        learner="random_forest",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
    )


def fit_predict_extra_trees_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
    n_estimators: int = 300,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    max_depth: int | None = None,
    criterion: str = "gini",
    class_weight: str | None = "balanced",
) -> np.ndarray:
    """Fit one balanced extremely-randomized-trees model per target.

    Mirrors :func:`fit_predict_random_forest_ovr` exactly except for the
    ensemble family, so any metric difference is attributable to the learner.
    """

    return fit_predict_forest_ovr(
        train_features,
        train_targets,
        test_features,
        target_codes=target_codes,
        random_state=random_state,
        learner="extra_trees",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
    )
