"""Configurable training utilities for small, reproducible experiments."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class TrainingStrategyConfig:
    scheduler: str = "none"
    warmup_epochs: int = 0
    gradient_clip_norm: float = 0.0
    amp: bool = False


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    amplitude_scale: float = 0.05
    noise_std: float = 0.01
    time_mask_ratio: float = 0.10
    max_time_masks: int = 1


class RawWindowAugmenter:
    """Apply deterministic, masked perturbations to normalized raw windows."""

    def __init__(self, config: AugmentationConfig, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)

    def __call__(self, windows: Any, sample_mask: Any, *, epoch: int, batch_index: int) -> Any:
        import torch

        if not self.config.enabled:
            return windows
        if windows.ndim == 4 and sample_mask.ndim == 3:
            original_shape = windows.shape
            windows = windows.reshape(-1, *windows.shape[-2:])
            sample_mask = sample_mask.reshape(-1, sample_mask.shape[-1])
        else:
            original_shape = None
        if windows.ndim != 3 or sample_mask.ndim != 2:
            raise ValueError("raw augmentation expects [B,C,T] or [B,W,C,T] windows.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch * 1_000_003 + batch_index * 9_973)
        result = windows.clone()
        valid = sample_mask.to(dtype=torch.bool).unsqueeze(1)
        if self.config.amplitude_scale:
            scale = 1.0 + (
                torch.rand(
                    (windows.shape[0], windows.shape[1], 1),
                    generator=generator,
                    dtype=windows.dtype,
                )
                * 2.0
                - 1.0
            ) * self.config.amplitude_scale
            result = result * scale.to(device=windows.device)
        if self.config.noise_std:
            noise = torch.randn(
                windows.shape, generator=generator, dtype=windows.dtype
            ).to(device=windows.device)
            result = result + noise * self.config.noise_std * valid
        mask_ratio = self.config.time_mask_ratio
        if mask_ratio > 0 and self.config.max_time_masks > 0:
            for row in range(result.shape[0]):
                valid_length = int(sample_mask[row].sum().item())
                if valid_length < 1:
                    continue
                width = max(1, min(valid_length, int(round(valid_length * mask_ratio))))
                for _ in range(self.config.max_time_masks):
                    start_limit = max(0, valid_length - width)
                    start = int(
                        torch.randint(
                            start_limit + 1, (1,), generator=generator
                        ).item()
                    )
                    result[row, :, start : start + width] = 0
        result = torch.where(valid, result, windows)
        return result.reshape(original_shape) if original_shape is not None else result


class TrainingStrategy:
    """Own optimizer stepping, scheduling, AMP and gradient instrumentation."""

    def __init__(
        self,
        optimizer: Any,
        *,
        config: TrainingStrategyConfig,
        epochs: int,
        steps_per_epoch: int,
        device: Any,
    ) -> None:
        import torch

        self.optimizer = optimizer
        self.config = config
        self.epochs = int(epochs)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.device = device
        self.scheduler = self._build_scheduler(torch)
        self.amp_enabled = bool(config.amp and device.type == "cuda")
        if hasattr(torch, "amp"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:  # pragma: no cover - compatibility with older supported torch builds
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self._grad_norms: list[float] = []

    def _build_scheduler(self, torch: Any) -> Any | None:
        name = self.config.scheduler
        if name == "none":
            return None
        if name == "one_cycle":
            pct_start = self.config.warmup_epochs / max(1, self.epochs)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max(group["lr"] for group in self.optimizer.param_groups),
                total_steps=max(1, self.epochs * self.steps_per_epoch),
                pct_start=min(0.9, max(0.05, pct_start or 0.1)),
                anneal_strategy="cos",
            )
        if name == "cosine":
            warmup = min(max(0, self.config.warmup_epochs), max(0, self.epochs - 1))

            def schedule(epoch: int) -> float:
                if warmup and epoch < warmup:
                    return float(epoch + 1) / warmup
                remaining = max(1, self.epochs - warmup)
                progress = min(1.0, max(0.0, (epoch + 1 - warmup) / remaining))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        raise ValueError("scheduler must be none, cosine or one_cycle.")

    def autocast(self) -> Any:
        if not self.amp_enabled:
            return nullcontext()
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def backward_step(self, loss: Any, model: Any) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        if self.config.gradient_clip_norm > 0:
            import torch

            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.gradient_clip_norm
            )
            grad_norm = float(norm.detach().cpu())
        else:
            grad_norm = _gradient_norm(model)
        if self.amp_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.config.scheduler == "one_cycle" and self.scheduler is not None:
            self.scheduler.step()
        self._grad_norms.append(grad_norm)
        return grad_norm

    def epoch_step(self) -> None:
        if self.config.scheduler == "cosine" and self.scheduler is not None:
            self.scheduler.step()

    def epoch_stats(self) -> dict[str, float]:
        values = self._grad_norms
        self._grad_norms = []
        return {
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": float(sum(values) / len(values)) if values else 0.0,
            "gradient_norm_max": float(max(values)) if values else 0.0,
        }


def _gradient_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().cpu())
    return math.sqrt(total)


__all__ = [
    "AugmentationConfig",
    "RawWindowAugmenter",
    "TrainingStrategy",
    "TrainingStrategyConfig",
]
