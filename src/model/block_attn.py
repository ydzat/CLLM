"""Global block-level self-attention."""
from __future__ import annotations

import torch
from torch import nn


class BlockAttention(nn.Module):
    """Dense self-attention over block states ``z``.

    With ``M`` small (a few hundred), this is dense attention; the fixed sparse
    mask from the architecture is an ablation knob, not applied in v1.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, M, d) -> (B, M, d)``."""
        out, _ = self.attn(z, z, z)
        return out
