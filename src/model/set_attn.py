"""Content-based (permutation-equivariant) intra-block attention."""
from __future__ import annotations

import torch
from torch import nn


class SetAttention(nn.Module):
    """Self-attention within a block's tokens, with no positional bias.

    Tokens attend to each other by content only. The operation is
    permutation-equivariant (permuting input permutes output); combined with
    sum-pooling it yields a permutation-invariant block state.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(S, T², d) -> (S, T², d)``."""
        out, _ = self.attn(x, x, x)
        return out
