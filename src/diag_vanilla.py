"""Controlled comparison: a VANILLA transformer on the SAME repeated real corpus.

Standard per-token position, full self-attention, FFN on every token, BPE token
ids, masked-LM training — the same data slice, steps, batch and seed as
``diag_crossread.py``. If this learns real Chinese (loss well below the unigram
floor) while the block-scan reader stays at the unigram, the block design is the
cause. If this ALSO plateaus, the cause is in the shared data/mask/loss path.
Runs on the HPC ``devel`` partition (CPU, free); uses no GPU.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_bpe_ids
from src.data.tokenizer import BpeTokenizer


class Layer(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.ffn(self.norm2(x))


class VanillaTransformer(nn.Module):
    """Per-token position + full bidirectional attention + per-token FFN."""

    def __init__(self, vocab: int, d: int, layers: int, d_ff: int, heads: int, seq_len: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.layers = nn.ModuleList(Layer(d, d_ff, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(n, device=x.device))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def load_tokens(tok, dataset: str, text_field: str, n_tokens: int) -> torch.Tensor:
    """Stream a bounded real-token buffer (same slice as the cross-read diagnostic)."""
    buf: list[int] = []
    for tid in iter_bpe_ids(tok, dataset, "train", text_field, max_chars=n_tokens * 4):
        buf.append(tid)
        if len(buf) >= n_tokens:
            break
    return torch.tensor(buf, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=50_000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", default="/data/bpe_tokenizer.json")
    ap.add_argument("--data", default="Skywork/SkyPile-150B")
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()

    tok = BpeTokenizer.load(args.vocab)
    print(f"streaming {args.tokens} real Chinese tokens...", flush=True)
    data = load_tokens(tok, args.data, args.text_field, args.tokens)
    print(f"loaded {data.numel()} tokens (vocab {tok.vocab_size})", flush=True)

    torch.manual_seed(0)
    rng = random.Random(0)
    model = VanillaTransformer(tok.vocab_size, args.d, args.layers, args.d * 4, 4, args.seq)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    n = data.numel()
    k = max(1, round(args.seq * 0.15))
    early: list[float] = []
    late: list[float] = []
    min_loss, min_step = float("inf"), 0
    for step in range(args.steps):
        starts = torch.randint(0, n - args.seq, (args.batch,))
        x = torch.stack([data[s : s + args.seq] for s in starts])
        xm = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        for i in range(args.batch):
            pos = torch.tensor(rng.sample(range(args.seq), k))
            xm[i, pos] = 1
            mask[i, pos] = True
        loss = F.cross_entropy(model(xm)[mask], x[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = loss.item()
        (early if step < 200 else late).append(v)
        if v < min_loss:
            min_loss, min_step = v, step
        if step % 200 == 0:
            print(f"  vanilla step {step}: {v:.4f}", flush=True)
    e, l = sum(early) / len(early), sum(late) / len(late)
    print(f"vanilla: early {e:.4f} -> late {l:.4f} | min {min_loss:.4f} @ step {min_step}", flush=True)


if __name__ == "__main__":
    main()
