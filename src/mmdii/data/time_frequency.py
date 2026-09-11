"""Leakage-safe time/frequency representations for weld windows.

The functions in this module operate on one normalized window at a time.  Any
normalization statistics must be fitted on the training fold only via
``RepresentationNormalizer``; the transforms themselves do not inspect labels
or other windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.signal import stft as scipy_stft


def _signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("signal must be a [channels, samples] array.")
    if not np.isfinite(values).all():
        raise ValueError("signal must contain only finite values.")
    return values


def _mask(sample_mask: np.ndarray | None, samples: int) -> np.ndarray:
    if sample_mask is None:
        return np.ones(samples, dtype=bool)
    values = np.asarray(sample_mask, dtype=bool)
    if values.ndim != 1 or values.size != samples:
        raise ValueError("sample_mask must have shape [samples].")
    if not values.any():
        raise ValueError("sample_mask must contain at least one valid sample.")
    return values


def _resize_rows(values: np.ndarray, target: int) -> np.ndarray:
    """Resize rows without changing the first dimension."""
    if target < 1:
        raise ValueError("target time bins must be positive.")
    if values.shape[-1] == target:
        return values.copy()
    flat = values.reshape(-1, values.shape[-1])
    source = np.linspace(0.0, 1.0, values.shape[-1])
    destination = np.linspace(0.0, 1.0, target)
    resized = np.stack([np.interp(destination, source, row) for row in flat])
    return resized.reshape(*values.shape[:-1], target)


def _resize_mask(mask: np.ndarray, target: int) -> np.ndarray:
    if mask.size == target:
        return mask.copy()
    positions = np.linspace(0.0, 1.0, mask.size)
    target_positions = np.linspace(0.0, 1.0, target)
    return np.interp(target_positions, positions, mask.astype(np.float64)) >= 0.999


def stft_representation(
    signal: np.ndarray,
    *,
    target_fs: float = 5400.0,
    n_fft: int = 256,
    hop_length: int = 64,
    sample_mask: np.ndarray | None = None,
    output_time_bins: int | None = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-power one-sided STFT and its propagated time mask.

    The output has shape ``[channels, n_fft // 2 + 1, time_bins]``.  ``n_fft``
    and ``hop_length`` are intentionally explicit so the two preregistered
    variants (256/64 and 512/128) remain reproducible.
    """
    values = _signal(signal)
    if not math.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be positive and finite.")
    if n_fft < 2 or hop_length < 1 or hop_length > n_fft:
        raise ValueError("n_fft and hop_length are invalid.")
    valid = _mask(sample_mask, values.shape[1])
    if values.shape[1] < n_fft:
        padded = np.zeros((values.shape[0], n_fft), dtype=np.float64)
        padded[:, : values.shape[1]] = values
        values = padded
        valid = np.pad(valid, (0, n_fft - valid.size))
    noverlap = n_fft - hop_length
    spectra = []
    frame_masks = []
    for channel in values:
        _, times, coeff = scipy_stft(
            channel,
            fs=target_fs,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
        )
        spectra.append(np.log1p(np.abs(coeff) ** 2))
        # scipy's frames start at n_fft/2 when boundary=None.
        starts = np.rint(times * target_fs - n_fft / 2).astype(int)
        frame_masks.append(
            np.asarray(
                [valid[max(0, start) : min(valid.size, start + n_fft)].any() for start in starts],
                dtype=bool,
            )
        )
    result = np.stack(spectra, axis=0)
    mask = frame_masks[0]
    if output_time_bins is not None:
        result = _resize_rows(result, output_time_bins)
        mask = _resize_mask(mask, output_time_bins)
    return result.astype(np.float32), mask


def cwt_representation(
    signal: np.ndarray,
    *,
    target_fs: float = 5400.0,
    frequencies: Iterable[float] | None = None,
    output_time_bins: int = 256,
    wavelet: str = "morl",
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Morlet scalogram on a fixed logarithmic frequency grid."""
    values = _signal(signal)
    if not math.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be positive and finite.")
    if output_time_bins < 1:
        raise ValueError("output_time_bins must be positive.")
    valid = _mask(sample_mask, values.shape[1])
    if frequencies is None:
        frequencies = np.geomspace(30.0, target_fs / 2.0, 48)
    freqs = np.asarray(tuple(frequencies), dtype=np.float64)
    if freqs.ndim != 1 or freqs.size < 1 or not np.isfinite(freqs).all():
        raise ValueError("frequencies must be a non-empty finite vector.")
    if np.any(freqs <= 0) or np.any(freqs > target_fs / 2.0):
        raise ValueError("frequencies must lie in (0, Nyquist].")
    try:
        import pywt
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError("CWT requires the PyWavelets package.") from error
    scales = pywt.central_frequency(wavelet) * target_fs / freqs
    rows = []
    masked = values.copy()
    masked[:, ~valid] = 0.0
    for channel in masked:
        coeff, _ = pywt.cwt(channel, scales, wavelet, sampling_period=1.0 / target_fs)
        rows.append(_resize_rows(np.log1p(np.abs(coeff)), output_time_bins))
    return np.stack(rows, axis=0).astype(np.float32), _resize_mask(valid, output_time_bins)


def dwt_representation(
    signal: np.ndarray,
    *,
    level: int = 5,
    wavelet: str = "db4",
    output_time_bins: int = 256,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SWT approximation/detail bands as ``[channels, 6, time]``."""
    values = _signal(signal)
    if level < 1 or output_time_bins < 1:
        raise ValueError("level and output_time_bins must be positive.")
    valid = _mask(sample_mask, values.shape[1])
    try:
        import pywt
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError("DWT requires the PyWavelets package.") from error
    multiple = 2**level
    padded_samples = int(math.ceil(values.shape[1] / multiple) * multiple)
    padded = np.zeros((values.shape[0], padded_samples), dtype=np.float64)
    padded[:, : values.shape[1]] = values
    padded[:, ~np.pad(valid, (0, padded_samples - valid.size))] = 0.0
    outputs = []
    for channel in padded:
        coeffs = pywt.swt(channel, wavelet, level=level, trim_approx=False)
        bands = [coeffs[0][0], *[detail for _, detail in coeffs]]
        # PyWavelets returns approximation/detail pairs from coarse to fine.
        outputs.append(_resize_rows(np.log1p(np.abs(np.stack(bands))), output_time_bins))
    return np.stack(outputs, axis=0).astype(np.float32), _resize_mask(valid, output_time_bins)


def transform_representation(
    signal: np.ndarray,
    representation: str,
    *,
    sample_mask: np.ndarray | None = None,
    target_fs: float = 5400.0,
    output_time_bins: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch one of the preregistered representation names."""
    if representation == "stft_256":
        return stft_representation(
            signal, target_fs=target_fs, n_fft=256, hop_length=64,
            sample_mask=sample_mask, output_time_bins=output_time_bins,
        )
    if representation == "stft_512":
        return stft_representation(
            signal, target_fs=target_fs, n_fft=512, hop_length=128,
            sample_mask=sample_mask, output_time_bins=output_time_bins,
        )
    if representation == "cwt_morl":
        return cwt_representation(
            signal, target_fs=target_fs, output_time_bins=output_time_bins,
            sample_mask=sample_mask,
        )
    if representation == "dwt_swt_db4":
        return dwt_representation(
            signal, output_time_bins=output_time_bins, sample_mask=sample_mask,
        )
    raise ValueError(f"Unknown representation: {representation}")


@dataclass(frozen=True)
class RepresentationNormalizer:
    """Per-input-channel normalization fitted strictly on training arrays."""

    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> "RepresentationNormalizer":
        values = tuple(np.asarray(array, dtype=np.float64) for array in arrays)
        if not values:
            raise ValueError("At least one training representation is required.")
        if any(array.ndim not in (3, 4) or not np.isfinite(array).all() for array in values):
            raise ValueError("Representations must be finite [C,F,T] or [N,C,F,T] arrays.")
        channels = values[0].shape[-3]
        if any(array.shape[-3] != channels for array in values):
            raise ValueError("Representations must have a common channel count.")
        joined = np.concatenate(
            [array.reshape((-1, channels) + array.shape[-2:]) for array in values], axis=0
        )
        means = joined.mean(axis=(0, 2, 3))
        stds = np.where(joined.std(axis=(0, 2, 3)) == 0.0, 1.0, joined.std(axis=(0, 2, 3)))
        return cls(means=means, stds=stds)

    def transform(self, representation: np.ndarray) -> np.ndarray:
        values = np.asarray(representation, dtype=np.float64)
        if values.ndim not in (3, 4) or values.shape[-3] != self.means.size:
            raise ValueError("Representation channel count does not match normalizer.")
        shape = (1, self.means.size, 1, 1) if values.ndim == 4 else (self.means.size, 1, 1)
        return ((values - self.means.reshape(shape)) / self.stds.reshape(shape)).astype(np.float32)
