"""Small ModernTCN-style encoder used by the MMDII baselines."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised by import contract test
    raise RuntimeError(
        "ModernTCN requires the MMDII-Core train extra: pip install .[train]."
    ) from error


class _ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = self.depthwise(values)
        transformed = self.mix(self.norm(transformed))
        return values + transformed


class ModernTCNSmall(nn.Module):
    """A compact pure-convolution encoder with cross-channel mixing."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: int = 32,
        embedding_dim: int = 64,
        kernel_size: int = 31,
        block_count: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1 or hidden_channels < 1 or embedding_dim < 1:
            raise ValueError("Model channel dimensions must be positive.")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3.")
        if block_count < 1:
            raise ValueError("block_count must be positive.")
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(
                _ModernTCNBlock(hidden_channels, kernel_size, dropout)
                for _ in range(block_count)
            )
        )
        self.output_projection = nn.Conv1d(hidden_channels, embedding_dim, 1)
        self.output_norm = nn.GroupNorm(1, embedding_dim)

    def forward(
        self,
        signal: torch.Tensor,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError("signal must have shape [batch, channels, samples].")
        if sample_mask is not None:
            if sample_mask.shape != signal.shape[:1] + signal.shape[2:]:
                raise ValueError("sample_mask must have shape [batch, samples].")
            sample_mask = sample_mask.to(dtype=torch.bool, device=signal.device)
            if not sample_mask.any(dim=1).all():
                raise ValueError("Each signal must contain at least one valid sample.")
            signal = signal * sample_mask.unsqueeze(1)
        values = self.input_projection(signal)
        values = self.blocks(values)
        values = self.output_norm(self.output_projection(values))
        if sample_mask is None:
            return values.mean(dim=2)
        weights = sample_mask.unsqueeze(1).to(dtype=values.dtype)
        return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
