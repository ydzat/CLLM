# Decision 0010 — Per-slot positional output + gated attention

## Problem

The first two training runs plateaued at loss 6.6 nats, exactly the unigram entropy of Chinese (9.56 bits/char ≈ 6.63 nats). Root cause: the implementation dropped the intra-block position (`PE_local`) from the original design, and a fully-masked block's `T²` tokens are identical (`embed[MASK]` + the same block position). Every layer op (sum-pool, block-attention broadcast, set attention, FFN) preserves this symmetry, so the slot attention computes softmax over identical keys → uniform attention → identical output for all `T²` slots. The model can emit only ONE character per masked block, so its loss floor is the unigram distribution.

## Decision

1. **Per-slot positional output** (`slot_bias`, shape `[T², vocab]`, init zeros): add it directly to the slot logits. Slot `j` gets a learned positional prior over characters, so even a fully-masked block yields `T²` distinct outputs. Content (via slot attention) refines the prior when the block is unmasked.
2. **Gated attention** (borrowed from Qwen3.6): `out * sigmoid(gate(x))` on both block attention and set attention, where `gate` is a per-dim linear projection of the input.

## Alternatives considered

- **Intra-block position at the input** (`PE_local` added to tokens): washes out after ~12 layers — the sum-pool amplifies the block state ~16× and the broadcast swamps the 0.02-scale position (verified: token diff decays 0.096 → 2.4e-07).
- **Mean-pool instead of sum-pool**: reduces amplification but diverges from the DeepSets sum-pool argument ([0003](0003-content-based-intra-block-interaction.md), [0008](0008-pre-norm-layernorm.md)).
- **Bigger model / more data**: does not address the structural symmetry limit.

## Consequences

- Loss can now drop below unigram entropy (verified on structured data: 9.15 → 2.55, breaking the "1 char per 16 slots" floor of 3.67).
- Relaxes [0002](0002-sparse-2d-position-and-block-size.md)/[0003](0003-content-based-intra-block-interaction.md): intra-block *interaction* stays permutation-invariant, but the *output* is now positional (slots know their position). This is the minimal relaxation needed for reconstruction.
- Requires retraining; the previous checkpoints are superseded.
