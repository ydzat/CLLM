"""Smoke tests for the model (tasks T4-T9)."""
import torch
import torch.nn.functional as F

from src.model.model import BlockScanReader
from src.model.pool import Pool, from_blocks, to_blocks
from src.model.position import SparsePosition
from src.model.routing import ConditionalRead

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


def test_position_num_positions():
    # AC-1: R*C position vectors, not N
    pos = SparsePosition(4, 16, 64)  # R=4, C=16
    assert pos.num_positions == 64
    assert pos.P.shape == (4, 16, 64)


def test_pool_permutation_invariant():
    # AC-3: block state invariant to token permutation
    torch.manual_seed(0)
    pool = Pool(64)
    hb = torch.randn(2, 4, 16, 64)  # (B, M, T², d)
    perm = torch.randperm(16)
    assert torch.allclose(pool(hb), pool(hb[:, :, perm]), atol=1e-5)


def test_router_mask_count():
    # AC-2: exactly round(alpha*M) blocks run the FFN
    torch.manual_seed(0)
    read = ConditionalRead(64, 256, 8, 0.5)
    z = torch.randn(2, 64, 64)
    hb = torch.randn(2, 64, 16, 64)
    _, mask = read(z, hb)
    assert mask.shape == (2, 64)
    assert (mask.sum(dim=1) == 32).all()


def test_to_from_blocks_roundtrip():
    torch.manual_seed(0)
    h = torch.randn(2, 16, 64, 8)  # (B, H, W, d) with T=4
    hb = to_blocks(h, 4)
    assert hb.shape == (2, 64, 16, 8)  # M=64, T²=16
    assert torch.equal(from_blocks(hb, 16, 64, 4), h)


def test_model_forward_shape():
    torch.manual_seed(0)
    model = BlockScanReader(CFG)
    x = torch.randint(0, CFG["vocab_size"], (2, CFG["seq_len"]))
    logits = model(x)
    assert logits.shape == (2, CFG["seq_len"], CFG["vocab_size"])


def test_model_backward_loss():
    torch.manual_seed(0)
    model = BlockScanReader(CFG)
    x = torch.randint(0, CFG["vocab_size"], (2, CFG["seq_len"]))
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, CFG["vocab_size"]), x.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
