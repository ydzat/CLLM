# 0005 — Discrete-diffusion generation

## Problem

How should an order-tolerant model generate text? Autoregressive left-to-right decoding imposes exactly the order the model was designed to be insensitive to.

## Decision

Generate by **non-autoregressive discrete diffusion** (iterative mask-predict): start from all-`[MASK]`, predict all tokens in parallel, keep high-confidence predictions, re-mask the rest, repeat `K ≈ 8–16` rounds.

## Alternatives considered

- **Autoregressive decoding**: imposes an artificial left-to-right order; contradicts [0002](0002-sparse-2d-position-and-block-size.md) and [0003](0003-content-based-intra-block-interaction.md).
- **Block-parallel autoregressive**: generate block-by-block, parallel within a block; still imposes an inter-block order.

## Consequences

- `O(K)` parallel passes replace `O(N)` sequential steps.
- The graph-generation field reached the same conclusion for order-free data: autoregressive node ordering is a harmful bias, and discrete diffusion (DiGress, arXiv:2209.14734; GraphARM, arXiv:2307.08849) is the adopted mechanism. Text-side, Mask-Predict (arXiv:1904.09324) and LLaDA (arXiv:2502.09992) validate it for language.
- Hypothesis H3: Chinese tolerates parallel fill-in better than English, since its meaning is order-tolerant.
