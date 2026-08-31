"""Streaming corpus loader.

On HPC, iterate characters from a HuggingFace dataset with ``streaming=True``
(no full download). Locally, iterate characters from hardcoded samples so the
pipeline can be smoke-tested without downloading any data. The ``datasets``
import is lazy so this module imports fine without it installed.
"""
from __future__ import annotations

from typing import Iterator


def iter_text_from_samples(samples: list[str]) -> Iterator[str]:
    """Yield characters from in-memory samples (local smoke test, zero download)."""
    for sample in samples:
        yield from sample


def iter_text_hf(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    max_chars: int | None = None,
) -> Iterator[str]:
    """Yield characters from a HuggingFace dataset, streamed.

    ``streaming=True`` avoids downloading the dataset; rows are fetched and
    discarded as consumed. ``max_chars`` caps the stream (for vocab building or
    a bounded smoke run).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    total = 0
    for example in ds:
        for ch in example[text_field]:
            yield ch
            total += 1
            if max_chars is not None and total >= max_chars:
                return
