"""Post-hoc combination of independently trained weld-level model outputs."""

from __future__ import annotations

from .late_fusion import (
    OofFrames,
    align_oof,
    clustered_bootstrap_ci,
    evaluate_fusion,
    fuse,
    load_oof,
    select_alpha_fold_local,
)

__all__ = [
    "OofFrames",
    "align_oof",
    "clustered_bootstrap_ci",
    "evaluate_fusion",
    "fuse",
    "load_oof",
    "select_alpha_fold_local",
]
