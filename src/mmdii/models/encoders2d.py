"""Small 2-D encoders for time--frequency window representations.

The encoders intentionally have no torchvision or pretrained-weight dependency.  They
consume ``[batch, channels, height, width]`` tensors and return one embedding per
window, which is the same contract used by :class:`mmdii.models.mil.WeldMIL`.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised by import contract tests
    raise RuntimeError(
        "2-D encoders require the MMDII-Core train extra: pip install .[train]."
    ) from error


def _check_image(values: torch.Tensor) -> None:
    if values.ndim != 4:
        raise ValueError("time-frequency input must have shape [batch, channels, height, width].")
    if values.shape[1] < 1 or values.shape[2] < 1 or values.shape[3] < 1:
        raise ValueError("time-frequency input dimensions must be positive.")


class _LayerNorm2d(nn.Module):
    """LayerNorm over channels while retaining NCHW storage."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.permute(0, 2, 3, 1)
        values = self.norm(values)
        return values.permute(0, 3, 1, 2)


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class _ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 2) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(values) + self.skip(values))


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = _LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = self.depthwise(values)
        transformed = self.mlp(self.norm(transformed))
        return values + transformed


class _ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gate(self.pool(values))


class _MultiScaleResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
        dilation_rates: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        groups = max(1, min(8, output_channels))
        while output_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, input_channels)
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        first_dilation, time_dilation, frequency_dilation = dilation_rates
        self.branches = nn.ModuleList([
            nn.Conv2d(
                input_channels, output_channels, 3, stride=stride,
                padding=first_dilation, dilation=first_dilation, bias=False,
            ),
            nn.Conv2d(
                input_channels, output_channels, (3, 7), stride=stride,
                padding=(time_dilation, 3 * time_dilation),
                dilation=(time_dilation, time_dilation), bias=False,
            ),
            nn.Conv2d(
                input_channels, output_channels, (7, 3), stride=stride,
                padding=(3 * frequency_dilation, frequency_dilation),
                dilation=(frequency_dilation, frequency_dilation), bias=False,
            ),
        ])
        self.norm2 = nn.GroupNorm(groups, output_channels)
        self.mix = nn.Conv2d(output_channels, output_channels, 1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = self.activation(self.norm1(values))
        transformed = sum(branch(transformed) for branch in self.branches)
        transformed = self.mix(self.activation(self.norm2(transformed)))
        skip = self.projection(values)
        return transformed + skip


class ResTFMultiScaleEncoder(nn.Module):
    """Multi-scale pre-activation residual encoder for time-frequency images."""

    def __init__(self, *, input_channels: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if input_channels < 1 or embedding_dim < 1:
            raise ValueError("Encoder dimensions must be positive.")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            _MultiScaleResidualBlock(32, 32),
            _MultiScaleResidualBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            _MultiScaleResidualBlock(32, 64, stride=2),
            _MultiScaleResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            _MultiScaleResidualBlock(64, 128, stride=2, dilation_rates=(1, 2, 4)),
            _MultiScaleResidualBlock(128, 128, dilation_rates=(1, 2, 4)),
        )
        self.gates = nn.ModuleList([_ChannelGate(32), _ChannelGate(64), _ChannelGate(128)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(32 + 64 + 128, embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _check_image(values)
        values = self.stem(values)
        stage1 = self.gates[0](self.stage1(values))
        stage2 = self.gates[1](self.stage2(stage1))
        stage3 = self.gates[2](self.stage3(stage2))
        pooled = [self.pool(stage).flatten(1) for stage in (stage1, stage2, stage3)]
        return self.projection(torch.cat(pooled, dim=1))


class _ImageEncoder(nn.Module):
    def __init__(self, trunk: nn.Module, output_channels: int, embedding_dim: int) -> None:
        super().__init__()
        if output_channels < 1 or embedding_dim < 1:
            raise ValueError("Encoder dimensions must be positive.")
        self.trunk = trunk
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(output_channels, embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _check_image(values)
        values = self.pool(self.trunk(values)).flatten(1)
        return self.projection(values)


class CNN2DEncoder(_ImageEncoder):
    """Compact Conv2d/BatchNorm/ReLU encoder."""

    def __init__(self, *, input_channels: int, embedding_dim: int = 64) -> None:
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        super().__init__(
            nn.Sequential(
                _ConvBlock(input_channels, 32),
                _ConvBlock(32, 64),
                _ConvBlock(64, 128),
            ),
            128,
            embedding_dim,
        )


class SeparableCNN2DEncoder(_ImageEncoder):
    """Depthwise-separable Conv2d encoder with a small parameter budget."""

    def __init__(self, *, input_channels: int, embedding_dim: int = 64) -> None:
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        super().__init__(
            nn.Sequential(
                _DepthwiseSeparableBlock(input_channels, 32),
                _DepthwiseSeparableBlock(32, 64),
                _DepthwiseSeparableBlock(64, 128),
            ),
            128,
            embedding_dim,
        )


class ResNet2DSmallEncoder(_ImageEncoder):
    """Small residual CNN; no external weights are required."""

    def __init__(self, *, input_channels: int, embedding_dim: int = 64) -> None:
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        super().__init__(
            nn.Sequential(
                _ResidualBlock(input_channels, 32),
                _ResidualBlock(32, 64),
                _ResidualBlock(64, 128),
            ),
            128,
            embedding_dim,
        )


class ConvNeXt2DLiteEncoder(_ImageEncoder):
    """Lightweight ConvNeXt-style encoder for spectrogram-like images."""

    def __init__(self, *, input_channels: int, embedding_dim: int = 64) -> None:
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        super().__init__(
            nn.Sequential(
                _ConvBlock(input_channels, 32),
                _ConvNeXtBlock(32),
                _ConvBlock(32, 64),
                _ConvNeXtBlock(64),
                _ConvBlock(64, 128),
                _ConvNeXtBlock(128),
            ),
            128,
            embedding_dim,
        )


_ENCODERS = {
    "cnn2d": CNN2DEncoder,
    "separable_cnn2d": SeparableCNN2DEncoder,
    "resnet2d_small": ResNet2DSmallEncoder,
    "convnext2d_lite": ConvNeXt2DLiteEncoder,
    "resnet2d_multiscale": ResTFMultiScaleEncoder,
}


def build_2d_encoder(
    name: str,
    *,
    input_channels: int,
    embedding_dim: int = 64,
) -> nn.Module:
    """Build one of the supported image encoders from its config name."""

    try:
        encoder_type = _ENCODERS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_ENCODERS))
        raise ValueError(f"Unsupported 2-D encoder {name!r}; choose from {choices}.") from error
    return encoder_type(input_channels=input_channels, embedding_dim=embedding_dim)


class RawTimeFrequencyFusion(nn.Module):
    """Fuse raw ModernTCN and time-frequency embeddings before WeldMIL.

    ``raw_encoder`` must accept ``(signal, sample_mask=...)`` and return ``[B, D]``;
    ``time_frequency_encoder`` accepts ``[B, C, H, W]`` and returns ``[B, D]``.
    Both embeddings are projected to ``embedding_dim`` and concatenated through a
    small linear projection, yielding the exact embedding contract expected by
    ``WeldMIL``.
    """

    def __init__(
        self,
        *,
        raw_encoder: nn.Module,
        time_frequency_encoder: nn.Module,
        raw_embedding_dim: int = 64,
        time_frequency_embedding_dim: int = 64,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(raw_embedding_dim, time_frequency_embedding_dim, embedding_dim) < 1:
            raise ValueError("Fusion dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.raw_encoder = raw_encoder
        self.time_frequency_encoder = time_frequency_encoder
        self.fusion = nn.Sequential(
            nn.Linear(raw_embedding_dim + time_frequency_embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        raw_signal: torch.Tensor,
        time_frequency: torch.Tensor,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_embedding = self.raw_encoder(raw_signal, sample_mask=sample_mask)
        tf_embedding = self.time_frequency_encoder(time_frequency)
        if raw_embedding.ndim != 2 or tf_embedding.ndim != 2:
            raise ValueError("Fusion encoders must return [batch, embedding] tensors.")
        if raw_embedding.shape[0] != tf_embedding.shape[0]:
            raise ValueError("Raw and time-frequency batch sizes must match.")
        return self.fusion(torch.cat((raw_embedding, tf_embedding), dim=1))


__all__ = [
    "CNN2DEncoder",
    "SeparableCNN2DEncoder",
    "ResNet2DSmallEncoder",
    "ConvNeXt2DLiteEncoder",
    "ResTFMultiScaleEncoder",
    "RawTimeFrequencyFusion",
    "build_2d_encoder",
]
