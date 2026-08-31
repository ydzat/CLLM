"""Conditional FFN: top-k block routing with straight-through gradient."""
from __future__ import annotations

import torch
from torch import nn

from .set_attn import SetAttention


class ConditionalRead(nn.Module):
    """Route a fraction ``alpha`` of blocks to intra-block Set attention + FFN.

    The router ``s_b = W_r z_b`` selects top-k blocks (hard selection, no
    gradient through it). The FFN output is gated by the soft router
    probability ``softmax(s)`` so the router receives gradient through the
    selected blocks (Mixture-of-Depths straight-through). Non-selected blocks
    skip the FFN entirely (the compute saving).
    """

    def __init__(self, d: int, d_ff: int, n_heads: int, alpha: float) -> None:
        super().__init__()
        self.router = nn.Linear(d, 1, bias=False)
        self.set_attn = SetAttention(d, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.alpha = alpha

    def forward(
        self, z: torch.Tensor, h_blocks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h_blocks_out, mask)``; ``mask`` is ``(B, M)`` with ``round(alpha*M)`` ones."""
        b, m = z.shape[0], z.shape[1]
        logits = self.router(z).squeeze(-1)  # (B, M)
        probs = logits.softmax(dim=1)  # (B, M)
        k = max(1, round(self.alpha * m))
        _, topk_idx = probs.topk(k, dim=1)
        mask = torch.zeros(b, m, device=z.device).scatter(1, topk_idx, 1.0)

        h_sel = h_blocks[mask.bool()]  # (S, T², d), S = B*k
        h_sel = h_sel + self.set_attn(h_sel)
        gate = probs[mask.bool()].unsqueeze(-1).unsqueeze(-1)  # (S, 1, 1)
        h_sel = h_sel + gate * self.ffn(h_sel)

        out = h_blocks.clone()
        out[mask.bool()] = h_sel
        return out, mask
