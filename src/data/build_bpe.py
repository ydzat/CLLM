"""Train a BPE (subword) tokenizer on a streamed corpus and save it.

Replaces char-level tokenization (decision 0001 → superseded by 0014): BPE
merges frequent Chinese character sequences into subword tokens, which have
lower conditional entropy and are learnable at realistic model scale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-chars", type=int, default=50_000_000)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Whitespace pre-tokenizer: Chinese has no spaces between words, so a
    # "word" is a whole run of characters, and BPE merges frequent adjacent
    # char pairs within it. (A per-char "isolated" split would give one-token
    # words and prevent any merge.)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[MASK]", "[UNK]"],
        min_frequency=2,
    )

    from datasets import load_dataset

    ds = load_dataset(args.data, split="train", streaming=True)

    def text_iter():
        total = 0
        for example in ds:
            text = example["text"]
            yield text
            total += len(text)
            if total >= args.max_chars:
                break

    tokenizer.train_from_iterator(text_iter(), trainer=trainer)
    tokenizer.save(args.out)
    print(f"saved BPE tokenizer -> {args.out}")
    print(f"vocab size: {tokenizer.get_vocab_size()}")
    for tok in ["[PAD]", "[MASK]", "[UNK]"]:
        print(f"  {tok} = {tokenizer.token_to_id(tok)}")
    # sanity: encode a sample
    ids = tokenizer.encode("深度学习是一个分支").ids
    print(f"encode '深度学习是一个分支' -> {ids} ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
