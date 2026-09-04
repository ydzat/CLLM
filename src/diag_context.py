"""Diagnostic: can the model COPY a char from block context (not position)?

Every token in block j is char (j mod V). A masked token must be recovered by
copying from its block's other (unmasked) tokens. If the loss drops below the
unigram floor ln(V), the slot attention can use in-block context; if it stays
at ln(V), context flow is broken.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.packing import block_linear_indices, num_blocks
from src.model.model import BlockScanReader


def main() -> None:
    cfg = yaml.safe_load(open("configs/dev.yaml"))
    mcfg = cfg["model"]
    model = BlockScanReader(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    V = 50
    T = mcfg["block_size"]
    W = mcfg["width"]
    N = mcfg["seq_len"]
    B = 8
    blk = torch.tensor(block_linear_indices(N, W, T))
    num_blocks(N, T)

    def make_batch(B: int) -> torch.Tensor:
        x = torch.zeros(B, N, dtype=torch.long)
        for i in range(N):
            j = blk[i].item()
            x[:, i] = (j % V) + 3  # block j -> char (j mod V)
        return x

    def mask_batch(x: torch.Tensor, ratio: float = 0.15):
        xm = x.clone()
        m = torch.zeros(B, N, dtype=torch.bool)
        rng = random.Random(0)
        for b in range(B):
            k = max(1, round(N * ratio))
            idx = torch.tensor(rng.sample(range(N), k))
            xm[b, idx] = 1
            m[b, idx] = True
        return xm, m

    uni = math.log(V)
    print(f"unigram floor over {V} chars = {uni:.4f} nats")
    for step in range(200):
        x = make_batch(B)
        xm, m = mask_batch(x)
        logits = model(xm)
        loss = F.cross_entropy(logits[m], x[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0:
            print(f"step {step:3d}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
