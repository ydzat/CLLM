# Decision 0008 — Pre-norm LayerNorm for numerical stability

## Problem

The layer sum-pools `T²` tokens into a block state and broadcasts it back with residual connections but no normalization. The pool sum multiplies the hidden-state magnitude by ~`T²` (16 for `T=4`) per layer; over `L=12` layers the state grows as `T^(2L) ≈ 1.7×10^7`, and the cross-entropy loss at step 0 is ~`3.7×10^7` instead of ~`ln(V) ≈ 9`. The model is untrainable at the full `d=512, L=12` configuration (only the tiny `L=1, d=64` smoke config escaped it).

## Decision

Apply pre-norm LayerNorm (LN) at every sublayer boundary: before global block attention, after the broadcast residual, before the router, and before intra-block Set attention and the FFN. Sum-pooling is kept (not replaced by mean-pooling), because the DeepSets universality argument in [0003](0003-content-based-intra-block-interaction.md) relies on it.

## Alternatives considered

- **Mean-pooling (`1/T² Σ`)** — removes the growth at the source but diverges from the documented sum-pool and weakens the DeepSets argument.
- **Post-norm (LN after the residual)** — the classic transformer choice, but pre-norm is more stable for deep stacks and is the current default.

## Consequences

- Hidden states stay `O(1)` across layers; loss at step 0 is ~9.04, matching `ln(V)`.
- LayerNorm adds `d` parameters and a small per-token cost per sublayer, negligible next to the FFN.
- Sum-pool still concentrates information (the LN gain/bias rescales it); this does not change the permutation-invariance guarantee, which the existing test verifies.
