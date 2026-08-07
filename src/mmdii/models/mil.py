"""Masked weld-level multiple-instance aggregators."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised by import contract test
    raise RuntimeError(
        "WeldMIL requires the MMDII-Core train extra: pip install .[train]."
    ) from error


class WeldMIL(nn.Module):
    """Aggregate window embeddings into one multi-label weld prediction."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_targets: int,
        mode: str = "gated_attention",
        top_k: int = 3,
        attention_dim: int = 64,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_targets < 1:
            raise ValueError("MIL dimensions must be positive.")
        if mode not in {"mean", "max", "topk_mean", "gated_attention"}:
            raise ValueError("Unsupported MIL aggregation mode.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.mode = mode
        self.top_k = top_k
        self.num_targets = num_targets
        self.window_classifier = nn.Linear(embedding_dim, num_targets)
        if mode == "gated_attention":
            if attention_dim < 1:
                raise ValueError("attention_dim must be positive.")
            self.attention_v = nn.Linear(embedding_dim, attention_dim)
            self.attention_u = nn.Linear(embedding_dim, attention_dim)
            self.attention_w = nn.Linear(attention_dim, num_targets, bias=False)
            self.attention_classifier = nn.Linear(embedding_dim, num_targets)

    def forward(
        self,
        embeddings: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if embeddings.ndim != 3 or window_mask.shape != embeddings.shape[:2]:
            raise ValueError("embeddings or window_mask has an invalid shape.")
        mask = window_mask.to(dtype=torch.bool, device=embeddings.device)
        if not mask.any(dim=1).all():
            raise ValueError("Each weld must contain at least one valid window.")
        window_logits = self.window_classifier(embeddings)
        if self.mode == "mean":
            weights = mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            return (
                (window_logits * weights).sum(dim=1)
                / weights.sum(dim=1).clamp_min(1.0),
                None,
            )
        if self.mode == "max":
            masked = window_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return masked.max(dim=1).values, None
        if self.mode == "topk_mean":
            values = []
            for batch_index in range(embeddings.shape[0]):
                valid = window_logits[batch_index][mask[batch_index]]
                count = min(self.top_k, valid.shape[0])
                values.append(valid.topk(count, dim=0).values.mean(dim=0))
            return torch.stack(values), None

        scores = self.attention_w(
            torch.tanh(self.attention_v(embeddings))
            * torch.sigmoid(self.attention_u(embeddings))
        ).transpose(1, 2)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = torch.softmax(scores, dim=2)
        pooled = torch.einsum("bkn,bnd->bkd", attention, embeddings)
        logits = self.attention_classifier(pooled)
        return logits.diagonal(dim1=1, dim2=2), attention
