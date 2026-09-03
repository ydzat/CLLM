"""Load a trained checkpoint and generate text by iterative mask-predict."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow `python src/generate.py` (run from the repo root) to import `src`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tokenizer import CharTokenizer
from src.model.model import BlockScanReader
from src.sampling import sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=16, help="mask-predict rounds K")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = BlockScanReader(mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint from step {ckpt['step']}")

    tok = CharTokenizer.load(args.vocab)

    for i in range(args.num_samples):
        ids = sample(model, mcfg["seq_len"], mask_id=tok.mask_id, steps=args.rounds, device=device)
        text = tok.decode(ids.tolist())
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
