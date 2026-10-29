"""Deterministic statistical lower bound for weld signals."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


STATISTIC_NAMES = (
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
)
STATISTIC_FEATURE_SETS = {
    "stats18": STATISTIC_NAMES,
    "stats15_no_ptp": (
        "mean",
        "std",
        "rms",
        "minimum",
        "maximum",
    ),
    "stats12_no_ptp_rms": (
        "mean",
        "std",
        "minimum",
        "maximum",
    ),
}


LEARNERS = ("random_forest", "extra_trees")
PROCESS_FEATURE_NAMES = (
    "depth_start",
    "plunge_delta",
    "rpm",
    "welding_speed",
)
CHANNEL_NAMES = ("af", "sf", "axialf")


def statistical_feature_names(
    feature_set: str = "stats18",
    *,
    interval_features: bool = False,
    process_features: bool = False,
    duration_feature: bool = False,
) -> tuple[str, ...]:
    """Return the stable column order used by statistical feature extraction."""
    try:
        names = STATISTIC_FEATURE_SETS[feature_set]
    except KeyError as error:
        raise ValueError(
            f"feature_set must be one of {tuple(STATISTIC_FEATURE_SETS)}"
        ) from error
    result = [
        f"{channel}_{name}"
        for channel in CHANNEL_NAMES
        for name in names
    ]
    if interval_features:
        result.extend(
            f"{channel}_interval_{segment}_{name}"
            for channel in CHANNEL_NAMES
            for segment in range(1, 4)
            for name in ("mean", "std")
        )
    if process_features:
        result.extend(PROCESS_FEATURE_NAMES)
    if duration_feature:
        result.append("duration_seconds")
    return tuple(result)


def extract_statistical_features(
    signal: np.ndarray,
    *,
    feature_set: str = "stats18",
    time: np.ndarray | None = None,
    process_parameters: Mapping[str, float] | None = None,
    interval_features: bool = False,
    process_features: bool = False,
    duration_feature: bool = False,
) -> np.ndarray:
    """Return whole-record stats and explicitly requested metadata feature blocks."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite [channels, samples] array.")
    try:
        names = STATISTIC_FEATURE_SETS[feature_set]
    except KeyError as error:
        raise ValueError(
            f"feature_set must be one of {tuple(STATISTIC_FEATURE_SETS)}"
        ) from error
    if interval_features or duration_feature:
        timestamps = np.asarray(time, dtype=np.float64) if time is not None else None
        if (
            timestamps is None
            or timestamps.ndim != 1
            or timestamps.size != values.shape[1]
            or timestamps.size < 2
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError(
                "time must be finite, strictly increasing and match signal samples."
            )
    else:
        timestamps = None
    rows = []
    for channel in values:
        features = {
            "mean": float(channel.mean()),
            "std": float(channel.std()),
            "rms": float(np.sqrt(np.mean(np.square(channel)))),
            "minimum": float(channel.min()),
            "maximum": float(channel.max()),
            "peak_to_peak": float(np.ptp(channel)),
        }
        rows.extend(features[name] for name in names)
    if interval_features:
        assert timestamps is not None
        boundaries = timestamps[0] + np.asarray((1.0 / 3.0, 2.0 / 3.0)) * (
            timestamps[-1] - timestamps[0]
        )
        edges = (0, *np.searchsorted(timestamps, boundaries, side="left"), len(timestamps))
        for channel in values:
            for start, stop in zip(edges[:-1], edges[1:]):
                segment = channel[start:stop]
                if segment.size == 0:
                    raise ValueError("Relative-time interval contains no signal samples.")
                rows.extend((float(segment.mean()), float(segment.std())))
    if process_features:
        if process_parameters is None:
            raise ValueError("process_parameters are required for process features.")
        try:
            process_values = (
                float(process_parameters["depth_start"]),
                float(process_parameters["depth_end"])
                - float(process_parameters["depth_start"]),
                float(process_parameters["rpm"]),
                float(process_parameters["welding_speed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "process_parameters require numeric depth_start, depth_end, rpm and welding_speed."
            ) from error
        if not np.isfinite(process_values).all():
            raise ValueError("process_parameters must be finite.")
        rows.extend(process_values)
    if duration_feature:
        assert timestamps is not None
        rows.append(float(timestamps[-1] - timestamps[0]))
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
    bootstrap: bool | None = None,
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
        or (bootstrap is not None and not isinstance(bootstrap, (bool, np.bool_)))
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
        estimator_kwargs = dict(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=None if max_features == "all" else max_features,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        if bootstrap is not None:
            estimator_kwargs["bootstrap"] = bool(bootstrap)
        model = estimator_class(**estimator_kwargs)
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
    bootstrap: bool | None = None,
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
        bootstrap=bootstrap,
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
    bootstrap: bool | None = None,
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
        bootstrap=bootstrap,
    )


def _validate_tabular_inputs(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    target_codes: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
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
        or not np.isfinite(train_x).all()
        or not np.isfinite(train_y).all()
        or not np.isfinite(test_x).all()
        or not np.isin(train_y, (0.0, 1.0)).all()
    ):
        raise ValueError("Feature, target and target-code shapes are inconsistent.")
    for position, code in enumerate(codes):
        if np.unique(train_y[:, position]).size != 2:
            raise ValueError(f"Training target {code!r} contains only one class.")
    return train_x, train_y, test_x, codes


def _balanced_sample_weights(target: np.ndarray) -> np.ndarray:
    positives = float(np.sum(target == 1.0))
    negatives = float(np.sum(target == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training target contains only one class.")
    weights = np.ones(target.shape[0], dtype=np.float64)
    weights[target == 1.0] = target.shape[0] / (2.0 * positives)
    weights[target == 0.0] = target.shape[0] / (2.0 * negatives)
    return weights


def fit_predict_gradient_boosting_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
    n_estimators: int = 100,
    min_samples_leaf: int = 2,
    learning_rate: float = 0.1,
    max_depth: int = 1,
    loss: str = "log_loss",
    subsample: float = 1.0,
) -> np.ndarray:
    """Fit one class-balanced GradientBoosting model per target."""

    train_x, train_y, test_x, codes = _validate_tabular_inputs(
        train_features, train_targets, test_features, target_codes
    )
    if (
        n_estimators < 1
        or min_samples_leaf < 1
        or learning_rate <= 0
        or max_depth < 1
        or loss != "log_loss"
        or not 0 < subsample <= 1
    ):
        raise ValueError("GradientBoosting settings are invalid.")
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as error:
        raise RuntimeError(
            "GradientBoosting training requires the MMDII-Core train extra."
        ) from error

    probabilities = np.empty((test_x.shape[0], len(codes)), dtype=np.float64)
    for position, code in enumerate(codes):
        target = train_y[:, position]
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            max_depth=max_depth,
            loss=loss,
            subsample=subsample,
            random_state=random_state,
        )
        model.fit(train_x, target, sample_weight=_balanced_sample_weights(target))
        probabilities[:, position] = model.predict_proba(test_x)[:, 1]
    return probabilities


def fit_predict_rbf_svm_scores_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    target_codes: Iterable[str],
    random_state: int,
    c_value: float = 1.0,
) -> np.ndarray:
    """Return fold-local-scaled RBF-SVM decision scores for each target."""

    train_x, train_y, test_x, codes = _validate_tabular_inputs(
        train_features, train_targets, test_features, target_codes
    )
    if c_value <= 0:
        raise ValueError("c_value must be positive.")
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as error:
        raise RuntimeError(
            "RBF-SVM training requires the MMDII-Core train extra."
        ) from error

    scores = np.empty((test_x.shape[0], len(codes)), dtype=np.float64)
    for position in range(len(codes)):
        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=c_value,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=random_state,
            ),
        )
        model.fit(train_x, train_y[:, position])
        scores[:, position] = model.decision_function(test_x)
    return scores


def fit_predict_rbf_svm_ovr(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    train_fold_ids: Iterable[int],
    target_codes: Iterable[str],
    random_state: int,
    c_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaled RBF SVMs and leakage-safe sigmoid calibration.

    The raw decision function is fitted on all outer-training rows. Calibration
    uses the provided inner partitions through ``PredefinedSplit``.
    """

    train_x, train_y, test_x, codes = _validate_tabular_inputs(
        train_features, train_targets, test_features, target_codes
    )
    fold_ids = np.asarray(tuple(int(value) for value in train_fold_ids), dtype=np.int64)
    if fold_ids.shape != (train_x.shape[0],):
        raise ValueError("train_fold_ids must match the training feature rows.")
    if c_value <= 0:
        raise ValueError("c_value must be positive.")
    unique_folds = tuple(sorted(set(fold_ids.tolist())))
    if len(unique_folds) < 2:
        raise ValueError("SVM calibration requires at least two inner folds.")
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import PredefinedSplit
    except ImportError as error:
        raise RuntimeError(
            "RBF-SVM training requires the MMDII-Core train extra."
        ) from error

    test_fold = np.asarray(
        [unique_folds.index(value) for value in fold_ids], dtype=np.int64
    )
    splitter = PredefinedSplit(test_fold=test_fold)
    for position, code in enumerate(codes):
        target = train_y[:, position]
        for fold in range(len(unique_folds)):
            validation = test_fold == fold
            fitting = ~validation
            if (
                np.unique(target[validation]).size != 2
                or np.unique(target[fitting]).size != 2
            ):
                raise ValueError(
                    f"SVM calibration fold for target {code!r} lacks a positive or negative class."
                )
    scores = fit_predict_rbf_svm_scores_ovr(
        train_x,
        train_y,
        test_x,
        target_codes=codes,
        random_state=random_state,
        c_value=c_value,
    )
    probabilities = np.empty_like(scores)
    for position, code in enumerate(codes):
        target = train_y[:, position]
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        def build_model() -> object:
            return make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=random_state,
                ),
            )

        try:
            calibrated = CalibratedClassifierCV(
                estimator=build_model(),
                method="sigmoid",
                cv=splitter,
                ensemble=False,
            )
        except TypeError:  # sklearn < 1.2 compatibility
            calibrated = CalibratedClassifierCV(
                base_estimator=build_model(),
                method="sigmoid",
                cv=splitter,
                ensemble=False,
            )
        calibrated.fit(train_x, target)
        probabilities[:, position] = calibrated.predict_proba(test_x)[:, 1]
    return scores, probabilities
