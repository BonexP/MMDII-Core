"""Line-based, ``tail -f`` friendly progress reporting for MMDII-Core runs.

Runtime progress is written to ``stderr`` only; the command-line entry points
keep ``stdout`` machine-readable. The library is silent by default: the
``mmdii`` logger carries a :class:`logging.NullHandler` and does not propagate,
so importing or unit testing the package never prints anything. Only the CLI and
suite entry points call :func:`configure_logging`, which attaches a single
stderr handler.

Every line is newline-terminated (no carriage returns), so piping the output
through ``tee`` and following it with ``tail -f`` shows stable, greppable
progress.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence


LOGGER_NAME = "mmdii"
LEVEL_ENV_VAR = "MMDII_LOG_LEVEL"
DEFAULT_LEVEL = "info"
_HANDLER_MARKER = "_mmdii_managed"


def _install_null_handler() -> None:
    """Make the ``mmdii`` logger silent until ``configure_logging`` is called."""

    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    logger.propagate = False


_install_null_handler()


class _StderrFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        short_name = record.name.removeprefix(f"{LOGGER_NAME}.")
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"[{stamp}] {record.levelname:<5} {short_name}: {record.getMessage()}"


class _StderrHandler(logging.StreamHandler):
    """A stderr handler that rebinds the stream so redirection keeps working."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)
        self.setFormatter(_StderrFormatter())
        setattr(self, _HANDLER_MARKER, True)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def resolve_level(level: str | int | None = None) -> int:
    """Resolve a level from an explicit value, ``MMDII_LOG_LEVEL`` or the default."""

    if isinstance(level, int):
        return level
    name = level if level is not None else os.environ.get(LEVEL_ENV_VAR, DEFAULT_LEVEL)
    if isinstance(name, str):
        resolved = logging.getLevelNamesMapping().get(name.strip().upper())
        if resolved is not None:
            return resolved
    return logging.INFO


def configure_logging(level: str | int | None = None) -> None:
    """Attach a single stderr handler to the ``mmdii`` logger (idempotent)."""

    _install_null_handler()
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()
    resolved = resolve_level(level)
    handler = _StderrHandler()
    handler.setLevel(resolved)
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a package logger; output is suppressed until logging is configured."""

    return logging.getLogger(name)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(step: int, total: int, elapsed: float) -> str:
    if step <= 0 or total <= 0 or elapsed <= 0:
        return "--:--:--"
    rate = elapsed / step
    return format_duration(max(0.0, rate * (total - step)))


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


class ProgressReporter:
    """Throttled, line-based progress updates for a long-running loop."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        total: int,
        label: str,
        every_steps: int = 1,
        every_seconds: float = 15.0,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger
        self._total = max(0, int(total))
        self._label = label
        self._every_steps = max(1, int(every_steps))
        self._every_seconds = float(every_seconds)
        self._level = level
        self._enabled = self._total > 0 and logger.isEnabledFor(level)
        self._start = time.monotonic()
        self._last_time = self._start
        self._last_step = 0

    def update(self, step: int) -> None:
        if not self._enabled:
            return
        step = int(step)
        now = time.monotonic()
        final = step >= self._total
        if (
            not final
            and (step - self._last_step) < self._every_steps
            and (now - self._last_time) < self._every_seconds
        ):
            return
        self._last_step = step
        self._last_time = now
        elapsed = now - self._start
        percent = 100.0 * step / self._total
        self._logger.log(
            self._level,
            "%s %d/%d (%.0f%%) elapsed=%s eta=%s",
            self._label,
            step,
            self._total,
            percent,
            format_duration(elapsed),
            format_eta(step, self._total, elapsed),
        )

    def close(self) -> None:
        if self._enabled and self._last_step < self._total:
            self.update(self._total)


def log_banner(
    logger: logging.Logger,
    *,
    experiment: str,
    mode: str,
    aggregator: str,
    seed: int,
    folds: Iterable[int],
    epochs: int,
    device: str,
    representation: str,
) -> None:
    logger.info(
        "experiment=%s mode=%s aggregator=%s seed=%s folds=%s epochs=%s device=%s representation=%s",
        experiment,
        mode,
        aggregator,
        seed,
        list(folds),
        epochs,
        device,
        representation,
    )


def log_dataset(
    logger: logging.Logger,
    *,
    sample_count: int,
    fold_count: int,
    target_codes: Sequence[str],
    representation: str,
) -> None:
    logger.info(
        "dataset samples=%d fold_count=%d targets=%s representation=%s",
        sample_count,
        fold_count,
        ",".join(str(code) for code in target_codes),
        representation,
    )


def log_epoch(
    logger: logging.Logger,
    *,
    fold: Any,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    learning_rate: float,
    gradient_norm_mean: float,
    gradient_norm_max: float,
    elapsed: float,
    eta: str,
) -> None:
    logger.info(
        "fold %s epoch %d/%d loss=%.4f lr=%.2e grad_mean=%.3f grad_max=%.3f elapsed=%s eta=%s",
        fold,
        epoch,
        total_epochs,
        train_loss,
        learning_rate,
        gradient_norm_mean,
        gradient_norm_max,
        format_duration(elapsed),
        eta,
    )


def log_fold_summary(logger: logging.Logger, *, fold: Any, metrics: Mapping[str, Any]) -> None:
    logger.info(
        "fold %s summary macro_f1=%s macro_recall=%s macro_pr_auc=%s epochs_ran=%s best_epoch=%s",
        fold,
        _format_value(metrics.get("macro_f1")),
        _format_value(metrics.get("macro_recall")),
        _format_value(metrics.get("macro_pr_auc")),
        _format_value(metrics.get("epochs_ran")),
        _format_value(metrics.get("best_epoch")),
    )


def log_results_table(
    logger: logging.Logger,
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    logger.info("results")
    widths: dict[str, int] = {}
    for column in columns:
        widths[column] = max(
            [len(str(column))] + [len(_format_value(row.get(column))) for row in rows]
        )
    logger.info("  %s", "  ".join(str(column).ljust(widths[column]) for column in columns))
    for row in rows:
        logger.info(
            "  %s",
            "  ".join(_format_value(row.get(column)).ljust(widths[column]) for column in columns),
        )


__all__ = [
    "DEFAULT_LEVEL",
    "LEVEL_ENV_VAR",
    "LOGGER_NAME",
    "ProgressReporter",
    "configure_logging",
    "format_duration",
    "format_eta",
    "get_logger",
    "log_banner",
    "log_dataset",
    "log_epoch",
    "log_fold_summary",
    "log_results_table",
    "resolve_level",
]
