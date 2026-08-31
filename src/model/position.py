"""Sparse block-granular position embedding."""
from __future__ import annotations

import torch
from torch import nn


class SparsePosition(nn.Module):
    """Learnable position ``P[u, v]`` at block granularity only.

    ``R * C`` position vectors exist (one per block), not ``N`` (one per token).
    Every token in a block shares the same position, so tokens within a block
    carry no positional information and are permutation-invariant.
    """

    def __init__(self, r: int, c: int, d: int) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.randn(r, c, d) * 0.02)

    @property
    def num_positions(self) -> int:
        """Number of distinct position vectors = R * C."""
        return self.P.shape[0] * self.P.shape[1]

    def forward(self, t: int) -> torch.Tensor:
        """Tile block positions to a ``(H, W, d)`` grid (``H = r*t``, ``W = c*t``)."""
        return self.P.repeat_interleave(t, dim=0).repeat_interleave(t, dim=1)
