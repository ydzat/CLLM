"""One block-scan layer: scan (pool / block-attn / block-FFN / broadcast / cross-read) + conditional read."""
from __future__ import annotations

import torch
from torch import nn

from .block_attn import BlockAttention
from .pool import Pool
from .routing import ConditionalRead


class BlockScanLayer(nn.Module):
    """Pool -> block attention -> block FFN -> broadcast -> token cross-read -> route -> conditional read.

    The block level mirrors a transformer block: ``block_attn`` is its
    self-attention and ``ffn_z`` is the per-megatoken FFN that a transformer
    block always has after attention (without it the block stream has no
    non-linear processing of its own). ``cross`` lets every token read ALL
    block states directly, instead of receiving only its own block's broadcast
    through the pooled bottleneck.
    """

    def __init__(
        self,
        d: int,
        d_ff: int,
        n_heads: int,
        alpha: float,
        block_ffn: bool = True,
        cross_read: bool = True,
    ) -> None:
        super().__init__()
        self.pool = Pool(d)
        self.norm_z = nn.LayerNorm(d)
        self.block_attn = BlockAttention(d, n_heads)
        self.use_block_ffn = block_ffn
        if block_ffn:
            self.norm_z2 = nn.LayerNorm(d)
            self.ffn_z = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.use_cross_read = cross_read
        if cross_read:
            self.norm_x = nn.LayerNorm(d)
            self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_h = nn.LayerNorm(d)
        self.read = ConditionalRead(d, d_ff, n_heads, alpha)

    def forward(self, h_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, M, T², d) -> (B, M, T², d)`` plus the routing mask ``(B, M)``."""
        b, m, tt, d = h_blocks.shape
        z = self.pool(h_blocks)  # (B, M, d)
        z = z + self.block_attn(self.norm_z(z))
        if self.use_block_ffn:
            z = z + self.ffn_z(self.norm_z2(z))  # block-level FFN on the block stream
        h_blocks = self.norm_h(h_blocks + z.unsqueeze(2))  # broadcast + normalize
        if self.use_cross_read:
            q = h_blocks.reshape(b, m * tt, d)
            read, _ = self.cross(self.norm_x(q), z, z)  # every token reads all M block states
            h_blocks = h_blocks + read.reshape(b, m, tt, d)
        h_blocks, mask = self.read(z, h_blocks)
        return h_blocks, mask
