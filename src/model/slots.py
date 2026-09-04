"""Slot attention output: set-to-sequence reconstruction."""
from __future__ import annotations

import math

import torch
from torch import nn


class SlotAttention(nn.Module):
    """Learned slots each select one character from a block's token set.

    Each of ``T²`` slots is a learned query that attends to the block's tokens
    by content and emits a vocabulary distribution, reconstructing the block's
    characters into an ordered "reasonable combination". The slot assignment is
    decided by content, not by position.
    """

    def __init__(self, d: int, n_slots: int, vocab_size: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, d))  # scale 1.0: match key scale, else attention ~uniform
        self.slot_bias = nn.Parameter(torch.zeros(n_slots, vocab_size))  # per-slot positional prior
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        # Non-linear head: a single linear `wo` would constrain the output to a
        # convex combination of the present tokens' logits (copy-only). The GELU
        # breaks that, letting the model GENERATE a char not present in the block.
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, vocab_size))

    def forward(self, h_blocks: torch.Tensor) -> torch.Tensor:
        """``(B, M, T², d) -> (B, M, T², vocab)``."""
        d = h_blocks.shape[-1]
        k = self.wk(h_blocks)  # (B, M, n_tok, d)
        v = self.wv(h_blocks)
        # Slots (s) attend to tokens (t); s and t are both size T² but distinct axes.
        attn = torch.einsum("sd,bmtd->bmst", self.slots, k) / math.sqrt(d)
        attn = attn.softmax(dim=-1)  # over tokens (t)
        out = torch.einsum("bmst,bmtd->bmsd", attn, v)  # (B, M, s, d)
        # Add the slot's own positional prior so a fully-masked block (uniform
        # content) still yields T² distinct outputs: slot j predicts the char at
        # position j, refined by content when the block is unmasked.
        return self.head(out) + self.slot_bias.unsqueeze(0).unsqueeze(0)
