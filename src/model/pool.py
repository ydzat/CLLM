"""Block grouping and sum-pooling (DeepSets)."""
from __future__ import annotations

import torch
from torch import nn


def to_blocks(h: torch.Tensor, t: int) -> torch.Tensor:
    """Reshape ``(B, H, W, d)`` to ``(B, M, T², d)`` in block-linear order.

    Blocks are ordered ``(u, v)`` row-major: linear index ``b = u * C + v``.
    """
    b, hgt, wdt, d = h.shape
    r, c = hgt // t, wdt // t
    return h.view(b, r, t, c, t, d).permute(0, 1, 3, 2, 4, 5).reshape(b, r * c, t * t, d)


def from_blocks(hb: torch.Tensor, hgt: int, wdt: int, t: int) -> torch.Tensor:
    """Inverse of :func:`to_blocks`; ``(B, M, T², d) -> (B, H, W, d)``."""
    b = hb.shape[0]
    r, c = hgt // t, wdt // t
    d = hb.shape[-1]
    return hb.view(b, r, c, t, t, d).permute(0, 1, 3, 2, 4, 5).reshape(b, hgt, wdt, d)


class Pool(nn.Module):
    """Sum-pool a block's tokens, then an MLP.

    Summation is order-independent, so the block state ``z_b`` is invariant to
    any permutation of tokens inside the block.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h_blocks: torch.Tensor) -> torch.Tensor:
        """``(B, M, T², d) -> (B, M, d)``."""
        return self.mlp(h_blocks.sum(dim=2))
