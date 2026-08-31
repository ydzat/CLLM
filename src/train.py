"""Training loop: block-level masked language modeling.

The CPU smoke test uses random data; real corpus data arrives via the HPC data
pipeline (see docs/development.md).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# Allow `python src/train.py` (run from the repo root) to import the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.packing import block_linear_indices, mask_whole_blocks, num_blocks
from src.model.model import BlockScanReader


def mask_whole_blocks_batch(
    x: torch.Tensor,
    width: int,
    block_size: int,
    mask_ratio: float,
    mask_id: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask whole blocks per batch item.

    Returns ``(x_masked, mask)`` where ``mask`` is ``(B, N)`` bool, True at
    masked tokens. A block is either fully masked or fully untouched.
    """
    b, n = x.shape
    blk_idx = torch.tensor(block_linear_indices(n, width, block_size), device=x.device)
    m = num_blocks(n, block_size)
    x_masked = x.clone()
    mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
    for i in range(b):
        blocks = mask_whole_blocks(n, block_size, mask_ratio, rng)
        sel = torch.isin(blk_idx, torch.tensor(sorted(blocks), device=x.device))
        x_masked[i, sel] = mask_id
        mask[i] = sel
    return x_masked, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = BlockScanReader(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    rng = random.Random(0)

    for step in range(tcfg["steps"]):
        # ids >= 3 skip the [PAD]/[MASK]/[UNK] specials
        x = torch.randint(
            3, mcfg["vocab_size"], (dcfg["batch_size"], mcfg["seq_len"]), device=device
        )
        x_masked, mask = mask_whole_blocks_batch(
            x, mcfg["width"], mcfg["block_size"], dcfg["mask_ratio"], mask_id=1, rng=rng
        )
        logits = model(x_masked)  # (B, N, V)
        loss = F.cross_entropy(logits[mask], x[mask])  # masked positions only
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step}: loss {loss.item():.4f}")

    print("smoke train complete")


if __name__ == "__main__":
    main()
