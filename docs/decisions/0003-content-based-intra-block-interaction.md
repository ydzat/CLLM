# 0003 — Content-based intra-block interaction

## Problem

Is cross-token interaction inside a block necessary, and if so, of what kind? Can a block be collapsed by sum-pooling alone?

## Decision

Interaction is **necessary**, and it must be **content-based** (permutation-invariant) — Set attention in the encoder's read step and slot attention at the output — never positional.

## Alternatives considered

- **Pure sum-pool + per-token FFN**: insufficient. A role assignment ("施事=我, 动作=吃, 受事=饭") is a relation/selection over tokens; per-token FFN carries no cross-token information and sum-pool is symmetric, so neither can reconstruct which token fills which role.
- **Positional attention within a block**: reconstructs roles by order, which contradicts the order-tolerance premise ([0002](0002-sparse-2d-position-and-block-size.md)).

## Consequences

- Content-based attention computes soft role assignments from token meaning, which is exactly the "reasonable combination" reconstruction.
- The model is blind to order-only contrasts within a block (狗追猫 vs 猫追狗 are the same set); they are resolved by semantic prior, mirroring human auto-correction.
- The cost is measurable (accuracy on order-ambiguous pairs) and grows with `T`; it is an evaluation target, not an accident.
