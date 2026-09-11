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

        names = ("cnn2d", "separable_cnn2d", "resnet2d_small", "convnext2d_lite")
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
