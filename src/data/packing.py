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


def block_linear_indices(length: int, width: int, block_size: int) -> list[int]:
    """Linear block index ``b = u*C + v`` for each flat token position.

    Matches the ordering produced by ``src.model.pool.to_blocks``: blocks are
    ordered ``(u, v)`` row-major over the ``H x W`` grid.
    """
    c_blocks = width // block_size
    out = []
    for i in range(length):
        r, c = i // width, i % width
        out.append((r // block_size) * c_blocks + (c // block_size))
    return out


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
