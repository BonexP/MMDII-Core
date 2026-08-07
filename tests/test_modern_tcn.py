from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ModernTCNTests(unittest.TestCase):
    @unittest.skipIf(TORCH_AVAILABLE, "train extra is installed")
    def test_missing_train_extra_has_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "train extra"):
            importlib.import_module("mmdii.models.modern_tcn")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_encoder_returns_finite_embeddings_and_gradients(self) -> None:
        import torch

        from mmdii.models.modern_tcn import ModernTCNSmall

        model = ModernTCNSmall(
            input_channels=3,
            hidden_channels=8,
            embedding_dim=12,
            kernel_size=7,
            block_count=2,
            dropout=0.0,
        )
        signal = torch.randn(4, 3, 32, requires_grad=True)
        sample_mask = torch.ones(4, 32, dtype=torch.bool)

        embeddings = model(signal, sample_mask=sample_mask)
        embeddings.sum().backward()

        self.assertEqual(tuple(embeddings.shape), (4, 12))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertIsNotNone(signal.grad)


if __name__ == "__main__":
    unittest.main()
