"""Build a char-level vocabulary from a corpus sample.

On HPC: stream a sample (~100M chars) from a HuggingFace dataset and write the
vocab. Locally: build from hardcoded samples (zero download) to verify the code.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow `python src/data/build_vocab.py` (run from the repo root) to import `src`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.corpus import iter_text_from_samples, iter_text_hf
from src.data.tokenizer import SPECIAL_TOKENS, CharTokenizer

LOCAL_SAMPLES = [
    "中文字符级预训练语料，块扫描阅读器测试。",
    "中国人在阅读书籍时可以扫过一块一块的内容。",
    "自然语言处理模型需要大量高质量语料。",
]


def build_vocab(chars, vocab_size: int) -> CharTokenizer:
    """Count characters and build a tokenizer from the top-``vocab_size`` chars."""
    counts = Counter(chars)
    top = [c for c, _ in counts.most_common(vocab_size - len(SPECIAL_TOKENS))]
    return CharTokenizer(chars=top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-chars", type=int, default=100_000_000)
    parser.add_argument("--out", default="data/vocab.json")
    parser.add_argument("--dataset", default=None, help="HF dataset name; omit to use builtin samples")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    if args.dataset:
        chars = iter_text_hf(args.dataset, "train", args.text_field, args.max_chars)
    else:
        chars = iter_text_from_samples(LOCAL_SAMPLES)

    tok = build_vocab(chars, args.vocab_size)
    tok.save(args.out)
    print(f"wrote {tok.vocab_size} entries to {args.out}")


if __name__ == "__main__":
    main()
