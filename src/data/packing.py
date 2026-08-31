"""2D grid packing and whole-block masking for the block-scan reader."""
from __future__ import annotations

import random


def grid_coords(length: int, width: int) -> tuple[list[int], list[int]]:
    """Row and column of each flat index, laid out ``width`` chars per line."""
    return [i // width for i in range(length)], [i % width for i in range(length)]


def block_coords(
    rows: list[int], cols: list[int], block_size: int
) -> list[tuple[int, int]]:
    """Block coordinate ``(u, v) = (row // T, col // T)`` for each token."""
    return [(r // block_size, c // block_size) for r, c in zip(rows, cols)]


def num_blocks(length: int, block_size: int) -> int:
    """Number of blocks covering ``length`` tokens at ``T x T`` blocks."""
    return (length + block_size * block_size - 1) // (block_size * block_size)


def mask_whole_blocks(
    length: int,
    block_size: int,
    mask_ratio: float,
    rng: random.Random | None = None,
) -> set[int]:
    """Return the set of block linear indices to mask.

    ``round(mask_ratio * M)`` whole blocks are chosen. Masking applies to every
    token in a chosen block, never to a partial block.
    """
    rng = rng or random.Random()
    m = num_blocks(length, block_size)
    k = max(1, round(m * mask_ratio))
    return set(rng.sample(range(m), k))
