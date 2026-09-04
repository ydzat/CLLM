"""Controlled comparison: a VANILLA BERT-style transformer on the SAME data.

Standard per-token position, full self-attention, FFN on every token, masked-LM
training. If this learns real Chinese (loss < 6.6) while the block-scan reader
stays at unigram, the bug is in the block-scan architecture. If this ALSO stays
at unigram, the bug is in the data/training pipeline, not the architecture.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_text_hf
from src.data.dataloader import batchify
from src.data.tokenizer import CharTokenizer


class Layer(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class VanillaTransformer(nn.Module):
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
            h = h + layer(h)
        return self.head(self.norm(h))


def mask_tokens(x: torch.Tensor, ratio: float, mask_id: int, rng: random.Random):
    b, n = x.shape
    xm = x.clone()
    m = torch.zeros(b, n, dtype=torch.bool, device=x.device)
    k = max(1, round(n * ratio))
    for i in range(b):
        idx = torch.tensor(rng.sample(range(n), k), device=x.device)
        xm[i, idx] = mask_id
        m[i, idx] = True
    return xm, m


def main() -> None:
    # Same params as configs/verify_med.yaml (d=128, L=2) for a fair comparison.
    d, layers, d_ff, heads, seq_len, vocab, batch = 128, 2, 512, 8, 1024, 8000, 8
    model = VanillaTransformer(vocab, d, layers, d_ff, heads, seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(0)

    tok = CharTokenizer.load("data/vocab.json") if Path("data/vocab.json").exists() else None
    # Use the HPC vocab path if present.
    import os

    vocab_path = os.environ.get("VOCAB", "/data/vocab.json")
    if not Path(vocab_path).exists():
        vocab_path = "data/vocab.json"
    tok = CharTokenizer.load(vocab_path)

    def char_ids():
        for ch in iter_text_hf("Skywork/SkyPile-150B", "train", "text", None):
            yield tok.to_id(ch)

    batches = batchify(char_ids(), batch, seq_len, tok.pad_id)

    print(f"vanilla transformer d={d} L={layers} — unigram floor ~6.6")
    for step, x in enumerate(batches):
        if step >= 400:
            break
        xm, m = mask_tokens(x, 0.15, 1, rng)
        logits = model(xm)
        loss = F.cross_entropy(logits[m], x[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
