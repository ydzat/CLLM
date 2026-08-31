"""One block-scan layer: scan (pool / block-attn / broadcast) + conditional read."""
from __future__ import annotations

import torch
from torch import nn

from .block_attn import BlockAttention
from .pool import Pool
from .routing import ConditionalRead


class BlockScanLayer(nn.Module):
    """Five steps: pool -> block attention -> broadcast -> route -> conditional read."""

    def __init__(self, d: int, d_ff: int, n_heads: int, alpha: float) -> None:
        super().__init__()
        self.pool = Pool(d)
        self.block_attn = BlockAttention(d, n_heads)
        self.read = ConditionalRead(d, d_ff, n_heads, alpha)

    def forward(self, h_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, M, T², d) -> (B, M, T², d)`` plus the routing mask ``(B, M)``."""
        z = self.pool(h_blocks)  # (B, M, d)
        z = z + self.block_attn(z)
        h_blocks = h_blocks + z.unsqueeze(2)  # broadcast
        h_blocks, mask = self.read(z, h_blocks)
        return h_blocks, mask
