"""CPU diagnostic: does the token -> all-block cross-read break the unigram plateau?

Streams one bounded real-Chinese token slice, then trains the block-scan reader
twice (``cross_read`` off and on) on the SAME slice, steps and seed, and prints
both loss trajectories. Runs on the HPC ``devel`` partition (CPU, free, 1 h);
it uses no GPU and consumes no GPU quota.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_bpe_ids
from src.data.tokenizer import BpeTokenizer
from src.model.model import BlockScanReader


def load_tokens(tok, dataset: str, text_field: str, n_tokens: int) -> torch.Tensor:
    """Stream a bounded real-token buffer (same slice for every arm)."""
    buf: list[int] = []
    for tid in iter_bpe_ids(tok, dataset, "train", text_field, max_chars=n_tokens * 4):
        buf.append(tid)
        if len(buf) >= n_tokens:
            break
    return torch.tensor(buf, dtype=torch.long)


def run(cfg: dict, data: torch.Tensor, steps: int, batch: int, seq: int, seed: int, label: str):
    """Train one arm; return (early mean loss, late mean loss)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = BlockScanReader(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    n = data.numel()
    k = max(1, round(seq * 0.15))
    early: list[float] = []
    late: list[float] = []
    for step in range(steps):
        starts = torch.randint(0, n - seq, (batch,))
        x = torch.stack([data[s : s + seq] for s in starts])
        xm = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        for i in range(batch):
            pos = torch.tensor(rng.sample(range(seq), k))
            xm[i, pos] = 1
            mask[i, pos] = True
        loss = F.cross_entropy(model(xm)[mask], x[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        (early if step < 200 else late).append(loss.item())
        if step % 200 == 0:
            print(f"  {label} step {step}: {loss.item():.4f}", flush=True)
    e, l = sum(early) / len(early), sum(late) / len(late)
    print(f"{label}: early {e:.4f} -> late {l:.4f}", flush=True)
    return e, l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=200_000)
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

    base = dict(
        d=args.d,
        layers=args.layers,
        d_ff=args.d * 4,
        heads=4,
        block_size=4,
        width=64,
        seq_len=args.seq,
        vocab_size=tok.vocab_size,
        alpha=0.5,
    )
    results = {}
    for label, cr in [("cross_read=OFF", False), ("cross_read=ON ", True)]:
        cfg = dict(base, cross_read=cr)
        results[label] = run(cfg, data, args.steps, args.batch, args.seq, 0, label)
    print("=== SUMMARY ===", flush=True)
    for label, (e, l) in results.items():
        print(f"{label}: early {e:.4f} late {l:.4f} delta {e - l:+.4f}", flush=True)


if __name__ == "__main__":
    main()
