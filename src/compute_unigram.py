"""Compute the unigram (marginal) char entropy of a streamed corpus.

Answers: is the training loss ~6.6 equal to the corpus's unigram entropy
(the model learned nothing beyond char frequency), or is it below it?
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
    counter: Counter[int] = Counter()
    total = 0
    for ch in iter_text_hf(args.data, "train", "text", args.max_chars):
        counter[tok.to_id(ch)] += 1
        total += 1

    h = 0.0
    for c, count in counter.items():
        p = count / total
        h -= p * math.log(p)
    print(f"total chars: {total}")
    print(f"distinct ids: {len(counter)}")
    print(f"unigram entropy: {h:.4f} nats = {h / math.log(2):.4f} bits/char")


if __name__ == "__main__":
    main()
