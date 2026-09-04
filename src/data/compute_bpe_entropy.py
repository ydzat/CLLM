"""Compute BPE token unigram + bigram (conditional) entropy of the corpus.

Tells us where the model's loss floor is: if the training loss equals the
unigram entropy, the model is stuck at the marginal (not learning bigram);
if it approaches the conditional entropy, it's learning context.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_bpe_ids
from src.data.tokenizer import BpeTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max-chars", type=int, default=10_000_000)
    args = parser.parse_args()

    tok = BpeTokenizer.load(args.vocab)
    uni: Counter[int] = Counter()
    bi: Counter[tuple[int, int]] = Counter()
    prev = None
    total = 0
    for tid in iter_bpe_ids(tok, args.data, "train", "text", args.max_chars):
        uni[tid] += 1
        if prev is not None:
            bi[(prev, tid)] += 1
        prev = tid
        total += 1

    hu = -sum((n / total) * math.log(n / total) for n in uni.values())
    nbi = sum(bi.values())
    hj = -sum((n / nbi) * math.log(n / nbi) for n in bi.values())
    hc = hj - hu
    print(f"BPE unigram H(t)        = {hu:.4f} nats ({hu/math.log(2):.3f} bits)")
    print(f"BPE bigram joint H      = {hj:.4f} nats")
    print(f"BPE conditional H(t2|t1) = {hc:.4f} nats  <- achievable loss")


if __name__ == "__main__":
    main()
