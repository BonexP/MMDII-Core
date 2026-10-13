from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from mmdii.models.mil import WeldMIL


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch train extra is not installed")
class WeldMILPoolTests(unittest.TestCase):
    def _batch(self, batch_size: int = 3, windows: int = 4, dim: int = 6, targets: int = 3):
        torch.manual_seed(0)
        embeddings = torch.randn(batch_size, windows, dim)
        window_mask = torch.ones(batch_size, windows, dtype=torch.bool)
        # Weld 0 has only two valid windows; the rest are padding.
        window_mask[0, 2:] = False
        embeddings[0, 2:] = 0.0
        return embeddings, window_mask, targets

    def test_forward_keeps_its_two_tuple_contract(self) -> None:
        embeddings, window_mask, targets = self._batch()
        head = WeldMIL(embedding_dim=6, num_targets=targets, mode="mean")

        result = head(embeddings, window_mask)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (embeddings.shape[0], targets))

    def test_shared_flag_matches_the_aggregation_mode(self) -> None:
        self.assertTrue(WeldMIL(embedding_dim=6, num_targets=3, mode="mean").representation_shared)
        self.assertFalse(
            WeldMIL(embedding_dim=6, num_targets=3, mode="gated_attention").representation_shared
        )

    def test_pool_returns_one_vector_per_weld_and_target(self) -> None:
        embeddings, window_mask, targets = self._batch()
        for mode in ("mean", "gated_attention"):
            head = WeldMIL(embedding_dim=6, num_targets=targets, mode=mode)
            pooled = head.pool(embeddings, window_mask)
            self.assertEqual(pooled.shape, (embeddings.shape[0], targets, 6))

    def test_mean_pool_matches_the_forward_logits(self) -> None:
        embeddings, window_mask, targets = self._batch()
        head = WeldMIL(embedding_dim=6, num_targets=targets, mode="mean")

        logits, _ = head(embeddings, window_mask)
        pooled = head.pool(embeddings, window_mask)

        torch.testing.assert_close(head.window_classifier(pooled[:, 0, :]), logits)

    def test_gated_pool_matches_the_forward_logits(self) -> None:
        embeddings, window_mask, targets = self._batch()
        head = WeldMIL(embedding_dim=6, num_targets=targets, mode="gated_attention")

        logits, attention = head(embeddings, window_mask)
        pooled = head.pool(embeddings, window_mask)

        self.assertEqual(attention.shape, (embeddings.shape[0], targets, embeddings.shape[1]))
        torch.testing.assert_close(
            head.attention_classifier(pooled).diagonal(dim1=1, dim2=2), logits
        )

    def test_pool_ignores_padded_windows(self) -> None:
        embeddings, window_mask, targets = self._batch()
        head = WeldMIL(embedding_dim=6, num_targets=targets, mode="mean")

        pooled = head.pool(embeddings, window_mask)
        expected = embeddings[0, :2].mean(dim=0)

        torch.testing.assert_close(pooled[0, 0], expected)

    def test_max_and_topk_refuse_to_expose_a_representation(self) -> None:
        embeddings, window_mask, targets = self._batch()
        for mode in ("max", "topk_mean"):
            head = WeldMIL(embedding_dim=6, num_targets=targets, mode=mode)
            with self.assertRaisesRegex(ValueError, "gated_attention"):
                head.pool(embeddings, window_mask)

    def test_pool_rejects_a_weld_without_valid_windows(self) -> None:
        embeddings, window_mask, targets = self._batch()
        window_mask[1, :] = False
        head = WeldMIL(embedding_dim=6, num_targets=targets, mode="mean")

        with self.assertRaisesRegex(ValueError, "at least one valid window"):
            head.pool(embeddings, window_mask)


if __name__ == "__main__":
    unittest.main()
