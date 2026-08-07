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
