from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Encoder2DTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_all_encoders_forward_and_backward(self) -> None:
        import torch

        from mmdii.models.encoders2d import build_2d_encoder

        names = (
            "cnn2d",
            "separable_cnn2d",
            "resnet2d_small",
            "resnet2d_wide",
            "resnet2d_separable",
            "resnet2d_multiscale",
            "convnext2d_lite",
            "convnext2d_deep",
        )
        image = torch.randn(4, 3, 20, 32, requires_grad=True)
        for name in names:
            model = build_2d_encoder(name, input_channels=3, embedding_dim=17)
            embedding = model(image)
            self.assertEqual(tuple(embedding.shape), (4, 17))
            self.assertTrue(torch.isfinite(embedding).all())
            embedding.sum().backward()
            self.assertIsNotNone(image.grad)
            image.grad.zero_()

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_encoder_variants_change_only_the_declared_stage_structure(self) -> None:
        import torch
        from torch import nn

        from mmdii.models.encoders2d import (
            _ConvNeXtBlock,
            _DepthwiseSeparableBlock,
            build_2d_encoder,
        )

        small = build_2d_encoder("resnet2d_small", input_channels=3)
        wide = build_2d_encoder("resnet2d_wide", input_channels=3)
        separable = build_2d_encoder("resnet2d_separable", input_channels=3)
        convnext_lite = build_2d_encoder("convnext2d_lite", input_channels=3)
        convnext_deep = build_2d_encoder("convnext2d_deep", input_channels=3)

        def parameter_count(model: nn.Module) -> int:
            return sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(wide.trunk[0].main[0].out_channels, 48)
        self.assertGreater(parameter_count(wide), parameter_count(small))
        self.assertTrue(
            all(
                isinstance(block.main[0], _DepthwiseSeparableBlock)
                and isinstance(block.main[1], _DepthwiseSeparableBlock)
                for block in separable.trunk
            )
        )
        self.assertLess(parameter_count(separable), parameter_count(small))
        self.assertEqual(
            sum(isinstance(module, _ConvNeXtBlock) for module in convnext_lite.modules()),
            3,
        )
        self.assertEqual(
            sum(isinstance(module, _ConvNeXtBlock) for module in convnext_deep.modules()),
            6,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_unknown_encoder_is_rejected(self) -> None:
        from mmdii.models.encoders2d import build_2d_encoder

        with self.assertRaisesRegex(ValueError, "Unsupported 2-D encoder"):
            build_2d_encoder("does_not_exist", input_channels=3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_raw_time_frequency_fusion_contract(self) -> None:
        import torch

        from mmdii.models.encoders2d import RawTimeFrequencyFusion, build_2d_encoder
        from mmdii.models.modern_tcn import ModernTCNSmall

        model = RawTimeFrequencyFusion(
            raw_encoder=ModernTCNSmall(
                input_channels=3,
                hidden_channels=8,
                embedding_dim=12,
                kernel_size=7,
                dropout=0.0,
            ),
            time_frequency_encoder=build_2d_encoder(
                "cnn2d", input_channels=3, embedding_dim=10
            ),
            raw_embedding_dim=12,
            time_frequency_embedding_dim=10,
            embedding_dim=16,
        )
        raw = torch.randn(4, 3, 64, requires_grad=True)
        image = torch.randn(4, 3, 20, 32, requires_grad=True)
        mask = torch.ones(4, 64, dtype=torch.bool)
        embedding = model(raw, image, sample_mask=mask)
        self.assertEqual(tuple(embedding.shape), (4, 16))
        embedding.square().mean().backward()
        self.assertIsNotNone(raw.grad)
        self.assertIsNotNone(image.grad)


if __name__ == "__main__":
    unittest.main()
