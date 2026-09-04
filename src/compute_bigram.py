"""Compute unigram AND bigram (conditional) char entropy of the corpus.

Decisive: if the conditional entropy H(c2|c1) is ~6.4 (close to unigram 6.6),
the model is FINE — char-level Chinese on this corpus genuinely has weak
bigram structure, so plateauing near unigram is expected. If H(c2|c1) is ~3,
there is a real bug (a bigram model should reach ~3, not 6.6).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_text_hf
from src.data.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max-chars", type=int, default=10_000_000)
    args = parser.parse_args()

    tok = CharTokenizer.load(args.vocab)
    uni: Counter[int] = Counter()
    bi: Counter[tuple[int, int]] = Counter()
    prev = None
    total = 0
    for ch in iter_text_hf(args.data, "train", "text", args.max_chars):
        c = tok.to_id(ch)
        uni[c] += 1
        if prev is not None:
            bi[(prev, c)] += 1
        prev = c
        total += 1

    hu = -sum((n / total) * math.log(n / total) for n in uni.values())
    nbi = sum(bi.values())
    hj = -sum((n / nbi) * math.log(n / nbi) for n in bi.values())
    hc = hj - hu
    print(f"unigram H(c)      = {hu:.4f} nats ({hu/math.log(2):.3f} bits)")
    print(f"bigram joint H    = {hj:.4f} nats")
    print(f"conditional H(c2|c1) = {hc:.4f} nats  <- achievable loss for a perfect bigram model")


if __name__ == "__main__":
    main()
