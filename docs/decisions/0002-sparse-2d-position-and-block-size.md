# 0002 — Sparse 2D position and block size

## Problem

Replace precise (per-token) positional encoding with a "sparse" one, exploiting that Chinese meaning tolerates coarse order. What layout, and what granularity?

## Decision

Text is laid out as a fixed-width 2D grid (line width `W`), partitioned into `T×T` blocks. Position is a **learnable block-granular embedding** `P[u_i, v_i]` shared by every token in the block; there is no position signal inside a block, so tokens within a block are a permutation-invariant set.

## Alternatives considered

- **Per-token 1D RoPE**: the baseline; precise but linear, and the thing being replaced.
- **Per-token 2D RoPE** (cf. arXiv:2607.16072): 2D but still precise per token.
- **No position at all**: drops order entirely; collapses even cross-block structure.

## Consequences

- Position space shrinks from `N` to `N/T²` positions.
- `T` is the single knob trading order precision against compute efficiency; `T=1` degenerates to per-token 2D.
- Hypothesis H1 (Chinese tolerates coarser `T` than English at equal quality) is a direct, falsifiable function of `T`.
- Order-only contrasts inside one block (狗追猫 vs 猫追狗) become indistinguishable; the cost grows with `T`.
