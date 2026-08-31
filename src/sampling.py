"""Iterative mask-predict sampling (discrete diffusion)."""
from __future__ import annotations

import torch

from .model.model import BlockScanReader


def sample(
    model: BlockScanReader,
    seq_len: int,
    mask_id: int,
    steps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate one full sequence by iterative mask-predict.

    Starts from all-``[MASK]``. Each round predicts all tokens, unmasking the
    most-confident ``ceil(remaining / rounds_left)`` of the still-masked tokens,
    so the grid is guaranteed fully unmasked within ``steps`` rounds.
    """
    model.eval()
    with torch.no_grad():
        x = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
        for step in range(1, steps + 1):
            logits = model(x)  # (1, N, V)
            probs = logits.softmax(dim=-1)
            conf, pred = probs.max(dim=-1)  # (1, N)
            masked = x == mask_id
            n_masked = int(masked.sum())
            if n_masked == 0:
                break
            remaining = steps - step + 1
            n_unmask = max(1, (n_masked + remaining - 1) // remaining)
            idx = masked[0].nonzero(as_tuple=False).view(-1)  # (n_masked,)
            conf_masked = conf[0, idx]
            _, top = conf_masked.topk(min(n_unmask, n_masked))
            x[0, idx[top]] = pred[0, idx[top]]
        return x[0]
