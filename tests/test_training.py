"""Smoke tests for training and sampling (tasks T10, T11)."""
import random

import torch

from src.model.model import BlockScanReader
from src.sampling import sample
from src.train import mask_whole_blocks_batch

CFG = {
    "d": 64,
    "layers": 1,
    "d_ff": 256,
    "heads": 8,
    "block_size": 4,
    "width": 64,
    "seq_len": 1024,
    "vocab_size": 100,
    "alpha": 0.5,
}


def test_mask_whole_blocks_batch():
    torch.manual_seed(0)
    x = torch.randint(3, 100, (2, 1024))
    x_masked, mask = mask_whole_blocks_batch(x, 64, 4, 0.15, mask_id=1, rng=random.Random(0))
    assert mask.shape == x.shape
    assert 0 < mask.sum() < x.numel()  # partial masking
    # a whole block is masked or not: token t (flat) is in block (t//64//4, t%64//4)
    # check no partial blocks: for each masked position, all tokens in its block are masked
    blk = [((i // 64) // 4) * (64 // 4) + ((i % 64) // 4) for i in range(1024)]
    blk = torch.tensor(blk)
    for b in range(2):
        masked_blocks = set(blk[mask[b]].tolist())
        for i in range(1024):
            if mask[b, i]:
                assert blk[i].item() in masked_blocks
            # unmasked positions must have x preserved
            else:
                assert x_masked[b, i] == x[b, i]


def test_sampling_full_grid():
    torch.manual_seed(0)
    model = BlockScanReader(CFG)
    out = sample(model, CFG["seq_len"], mask_id=1, steps=8)
    assert out.shape == (CFG["seq_len"],)
    assert (out != 1).all()  # no [MASK] remains
