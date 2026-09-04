"""The block-scan reader model."""
from __future__ import annotations

import torch
from torch import nn

from .block import BlockScanLayer
from .pool import from_blocks, to_blocks
from .position import SparsePosition
from .slots import SlotAttention


class BlockScanReader(nn.Module):
    """Char-level input -> 2D grid -> blocks -> sparse position -> layers -> slots.

    ``seq_len`` must equal ``H * W`` with ``H`` and ``W`` both divisible by
    ``T``; the data pipeline pads/truncates to this fixed length.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["d"]
        t = cfg["block_size"]
        w = cfg["width"]
        seq_len = cfg["seq_len"]
        h = seq_len // w
        assert h % t == 0 and w % t == 0, "grid must divide evenly into blocks"

        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.position = SparsePosition(h // t, w // t, d)
        self.intra = nn.Parameter(torch.randn(t * t, d) * 0.02)  # (T², d) intra-block position
        self.layers = nn.ModuleList(
            BlockScanLayer(d, cfg["d_ff"], cfg["heads"], cfg["alpha"])
            for _ in range(cfg["layers"])
        )
        self.slots = SlotAttention(d, t * t, cfg["vocab_size"])
        self.h, self.w, self.t = h, w, t
        self.num_blocks = (h // t) * (w // t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N) -> (B, N, vocab)`` slot logits."""
        b = x.shape[0]
        h = self.embed(x)  # (B, N, d)
        h = h.view(b, self.h, self.w, -1)
        h = h + self.position(self.t).unsqueeze(0)
        hb = to_blocks(h, self.t)  # (B, M, T², d)
        for layer in self.layers:
            hb = hb + self.intra  # re-inject intra-block position each layer (survives sum-pool)
            hb, _ = layer(hb)
        logits = self.slots(hb)  # (B, M, T², vocab)
        return from_blocks(logits, self.h, self.w, self.t).view(b, -1, logits.shape[-1])
