"""Training loop: token-level masked language modeling (BERT/LLaDA style).

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Allow `python src/train.py` (run from the repo root) to import the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import batchify
from src.data.packing import block_linear_indices, mask_whole_blocks, num_blocks
from src.data.tokenizer import BpeTokenizer
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


def mask_tokens(
    x: torch.Tensor,
    mask_ratio: float,
    mask_id: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a random ``mask_ratio`` fraction of individual tokens.

    Unlike whole-block masking, each token is masked independently, so a block
    keeps most of its content and the slot output reconstructs masked tokens
    from their in-block neighbours (rearrangement, not full generation).
    Returns ``(x_masked, mask)`` with ``mask`` ``(B, N)`` bool.
    """
    b, n = x.shape
    x_masked = x.clone()
    mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
    k = max(1, round(n * mask_ratio))
    for i in range(b):
        idx = torch.tensor(rng.sample(range(n), k), device=x.device)
        x_masked[i, idx] = mask_id
        mask[i, idx] = True
    return x_masked, mask


def make_batches(args, mcfg, dcfg):
    """Yield ``(batch_size, seq_len)`` LongTensor batches.

    With ``--data``: stream a HuggingFace dataset, map chars to ids via the
    vocab at ``--vocab``, and batchify (finite — ends when the stream ends).
    Without ``--data``: infinite random batches for the smoke test.
    """
    if args.data:
        from src.data.corpus import iter_bpe_ids

        tok = BpeTokenizer.load(args.vocab)

        def token_ids():
            yield from iter_bpe_ids(tok, args.data, "train", args.text_field, args.max_chars)

        yield from batchify(token_ids(), dcfg["batch_size"], mcfg["seq_len"], tok.pad_id)
    else:
        while True:
            yield torch.randint(3, mcfg["vocab_size"], (dcfg["batch_size"], mcfg["seq_len"]))


def build_scheduler(opt: torch.optim.Optimizer, tcfg: dict):
    """Linear warmup then cosine decay to ``min_lr`` (LLaMA/GPT-3 recipe)."""
    steps = tcfg["steps"]
    warmup_steps = tcfg.get("warmup_steps", 0)
    min_lr = tcfg.get("min_lr", tcfg["lr"] * 0.1)
    if warmup_steps > 0:
        warmup = LinearLR(opt, start_factor=1 / warmup_steps, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=max(1, steps - warmup_steps), eta_min=min_lr)
        return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(opt, T_max=steps, eta_min=min_lr)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched,
    step: int,
    cfg: dict,
) -> None:
    """Save model + optimizer + scheduler state so training can resume."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--data", default=None, help="HF dataset name (omit for random smoke data)")
    parser.add_argument("--vocab", default=None, help="path to vocab JSON (required with --data)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="override training.steps from config")
    parser.add_argument("--save-dir", default="checkpoints", help="where to write checkpoints")
    parser.add_argument("--save-interval", type=int, default=1000, help="save every N steps")
    parser.add_argument("--mask-ratio", type=float, default=None, help="fixed mask ratio (omit for U(0,1) sampling)")
    args = parser.parse_args()

    if args.data and not args.vocab:
        parser.error("--vocab is required when --data is set")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    if args.steps is not None:
        tcfg["steps"] = args.steps

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # ~10x faster on H100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = BlockScanReader(mcfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.95), weight_decay=0.1
    )  # LLaDA/GPT-3 recipe: beta2=0.95, wd=0.1
    sched = build_scheduler(opt, tcfg)
    rng = random.Random(0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for step, x in enumerate(make_batches(args, mcfg, dcfg)):
        if step >= tcfg["steps"]:
            break
        x = x.to(device)
        mask_ratio = args.mask_ratio if args.mask_ratio is not None else rng.random()
        x_masked, mask = mask_tokens(x, mask_ratio, mask_id=1, rng=rng)
        logits = model(x_masked)  # (B, N, V)
        loss = F.cross_entropy(logits[mask], x[mask])  # masked positions only
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        print(f"step {step}: loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
        if (step + 1) % args.save_interval == 0:
            ckpt_path = save_dir / f"step_{step + 1}.pt"
            save_checkpoint(ckpt_path, model, opt, sched, step + 1, cfg)
            print(f"saved {ckpt_path}")

    final_path = save_dir / "final.pt"
    save_checkpoint(final_path, model, opt, sched, tcfg["steps"], cfg)
    print(f"saved {final_path}")
    print("train complete")


if __name__ == "__main__":
    main()
