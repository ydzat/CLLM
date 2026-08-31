# 0004 — Conditional FFN (top-k routing)

## Problem

The FFN dominates per-token FLOPs (`O(N·d²)`), yet the reading model says compute should be uneven: easy blocks are skimmed, hard blocks are read precisely. A uniform FFN applied to every token contradicts this.

## Decision

A per-block router `s_b = W_r z_b` selects the top-`k` blocks each layer; the FFN (and intra-block Set attention) runs **only** on those blocks (fraction `α = k/M`). Gradient through the selection uses a straight-through estimator.

## Alternatives considered

- **Uniform FFN**: the baseline; spends equal compute on all blocks.
- **Token-level Mixture-of-Experts**: finer routing granularity, but more overhead and no block structure.

## Consequences

- Exact FLOP cap: FFN cost is `O(8α·N·d²)`, a speed lever at every context length (unlike sparse attention, which only helps at long context).
- This is the mechanism that makes "FFN is the bottleneck" an opportunity rather than a wall.
- Main risk is router collapse (all-or-nothing selection) and training instability; a load-balancing auxiliary loss is the standard mitigation (Mixture-of-Depths, arXiv:2404.02258).
- Skipped blocks keep only their coarse block state; the router can re-select them in later layers.
