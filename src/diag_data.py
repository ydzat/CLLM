"""Diagnostic: inspect the actual training data (decode a batch + char stats).

If the decoded text is sensible Chinese and the char distribution is Zipfian,
the data is fine and the bug is elsewhere. If it's garbage, the data pipeline
or tokenizer is broken.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import iter_text_hf
from src.data.dataloader import batchify
from src.data.tokenizer import CharTokenizer


def main() -> None:
    vocab_path = os.environ.get("VOCAB", "/data/vocab.json")
    if not Path(vocab_path).exists():
        vocab_path = "data/vocab.json"
    tok = CharTokenizer.load(vocab_path)

    def char_ids():
        for ch in iter_text_hf("Skywork/SkyPile-150B", "train", "text", None):
            yield tok.to_id(ch)

    batches = batchify(char_ids(), 8, 1024, tok.pad_id)
    x = next(batches)
    print("batch shape:", tuple(x.shape))

    seq = x[0].tolist()
    text = tok.decode(seq)
    print("decoded first 150 chars:", repr(text[:150]))

    c = Counter(seq)
    print("distinct ids in first seq:", len(c))
    print("top 8 ids:", c.most_common(8))
    print("PAD(0) count:", c.get(0, 0), "UNK(2) count:", c.get(2, 0), "MASK(1) count:", c.get(1, 0))

    # bigram check: do adjacent chars correlate (is there structure to learn)?
    pairs = Counter(zip(seq, seq[1:]))
    print("distinct bigrams in first seq:", len(pairs))
    print("top 5 bigrams:", [tok.decode([a, b]) for (a, b), _ in pairs.most_common(5)])


if __name__ == "__main__":
    main()
