from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class WeldMILTests(unittest.TestCase):
    @unittest.skipIf(TORCH_AVAILABLE, "train extra is installed")
    def test_missing_train_extra_has_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "train extra"):
            importlib.import_module("mmdii.models.mil")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
    def test_all_aggregators_ignore_padded_windows(self) -> None:
        import torch

        from mmdii.models.mil import WeldMIL

        valid = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], dtype=torch.float32
        )
        padded = torch.cat((valid, torch.full((1, 2, 2), 1000.0)), dim=1)
        valid_mask = torch.tensor([[True, True, True]])
        padded_mask = torch.tensor([[True, True, True, False, False]])

        for mode in ("mean", "max", "topk_mean", "gated_attention"):
            model = WeldMIL(
                embedding_dim=2,
                num_targets=3,
                mode=mode,
                top_k=2,
                attention_dim=4,
            )
            model.eval()
            expected, _ = model(valid, valid_mask)
            actual, attention = model(padded, padded_mask)
            torch.testing.assert_close(actual, expected)
            if attention is not None:
                self.assertEqual(tuple(attention.shape), (1, 3, 5))
                self.assertTrue(torch.equal(attention[:, :, 3:], torch.zeros(1, 3, 2)))


if __name__ == "__main__":
    unittest.main()
