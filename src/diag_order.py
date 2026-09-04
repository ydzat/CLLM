"""Diagnostic: can the model reconstruct ORDERED chars within a block (context + position)?

block j, slot s -> char (j*16 + s) mod V: 16 DISTINCT chars per block in a known
order. Token-level masking. If loss -> 0, ordered reconstruction works (content
path + intra-block position functional). If loss stays ~ln(V) = unigram floor,
the model cannot align slot s to position s (positional-CE vs permutation mismatch).
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
            r, c = i // W, i % W
            s = (r % T) * T + (c % T)  # slot 0..15
            j = blk[i].item()  # block index
            x[:, i] = (j * 16 + s) % V + 3  # block j, slot s -> char (j*16+s) mod V
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
    x = make_batch(B)
    xm, m = mask_batch(x)
    for step in range(500):
        logits = model(xm)
        loss = F.cross_entropy(logits[m], x[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
