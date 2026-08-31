"""Batch char-id streams into fixed-length 2D-grid tensors.

Device-agnostic: yields CPU ``torch.LongTensor``; the caller moves them to the
training device. Shuffling and multi-worker streaming belong on HPC, not here.
"""
from __future__ import annotations

from typing import Iterator

import torch


def batchify(
    char_ids: Iterator[int],
    batch_size: int,
    seq_len: int,
    pad_id: int,
) -> Iterator[torch.Tensor]:
    """Group a stream of char ids into ``(batch_size, seq_len)`` tensors.

    The stream is chunked into ``seq_len`` runs; a trailing partial chunk is
    padded with ``pad_id``. Consecutive chunks form a batch; a trailing partial
    batch is padded with all-``pad_id`` rows to ``batch_size``.
    """
    buf: list[int] = []
    batch: list[list[int]] = []
    for cid in char_ids:
        buf.append(cid)
        if len(buf) == seq_len:
            batch.append(buf)
            buf = []
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []
    if buf:
        buf.extend([pad_id] * (seq_len - len(buf)))
        batch.append(buf)
    if batch:
        while len(batch) < batch_size:
            batch.append([pad_id] * seq_len)
        yield torch.tensor(batch, dtype=torch.long)
