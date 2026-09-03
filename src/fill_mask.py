"""Fill-mask evaluation: predict masked blocks in a real sentence.

Shows what the model learned on its actual training task (block MLM with
variable mask ratio), independently of unconditional generation.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.packing import block_linear_indices, mask_whole_blocks
from src.data.tokenizer import CharTokenizer
from src.model.model import BlockScanReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlockScanReader(mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = CharTokenizer.load(args.vocab)
    n = mcfg["seq_len"]
    ids = tok.encode(args.text)[:n]
    x = torch.full((1, n), tok.pad_id, dtype=torch.long, device=device)
    x[0, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    rng = random.Random(0)
    blocks = mask_whole_blocks(n, mcfg["block_size"], args.mask_ratio, rng)
    blk_idx = torch.tensor(
        block_linear_indices(n, mcfg["width"], mcfg["block_size"]), device=device
    )
    sel = torch.isin(blk_idx, torch.tensor(sorted(blocks), device=device))
    x_masked = x.clone()
    x_masked[0, sel] = tok.mask_id

    with torch.no_grad():
        logits = model(x_masked)
        topk_idx = logits[0].topk(args.topk, dim=-1).indices  # (N, topk)

    correct = 0
    total = 0
    print(f"=== masked positions (actual vs top-{args.topk} predictions) ===")
    for i in range(n):
        if sel[i].item() and int(x[0, i].item()) != tok.pad_id:
            actual = tok.decode([int(x[0, i].item())])
            preds = [tok.decode([int(j)]) for j in topk_idx[i].tolist()]
            hit = actual in preds
            correct += int(hit)
            total += 1
            print(f"pos {i:4d}: actual={actual!r}  top{args.topk}={preds}  {'HIT' if hit else 'miss'}")
    if total:
        print(f"\naccuracy@top{args.topk}: {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()
