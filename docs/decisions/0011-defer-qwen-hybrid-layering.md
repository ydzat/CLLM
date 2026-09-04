# Decision 0011 — Defer Qwen hybrid layering (Gated DeltaNet)

## Problem

Qwen3.6-27B's headline result rests on a 3:1 "hybrid layering" (3 Gated DeltaNet linear-attention layers : 1 Gated Attention layer). Should the block-scan reader adopt it? Qwen ships the same hybrid down to 0.8B (verified in `config.json`), but never below that.

## Decision

Do not add hybrid layering now. The block attention is already `O(M) = O(64)` (64 block vectors), not `O(N²)`; linear attention exists to cut `O(N²)`, a problem this model does not have at `N=1024`. Evidence against it at our scale: pure linear attention scores ppl 17.42 vs softmax 13.15 at 70M params, and hybrid models lag a pure transformer until ~1.5T training tokens. Revisit only when the sequence grows to where block count `M = N/T²` makes block attention `O(M²)` a real cost.

## Alternatives considered

- **Adopt hybrid now**: adds DeltaNet kernels + stability risk while cutting a negligible fraction of compute (the FFN dominates at short context).
- **Adopt pure linear attention**: strictly worse at small scale.
- **Borrow only the gating** (chosen, [0010](0010-per-slot-positional-output.md)): the one Qwen idea that is small-scale-safe and applies regardless of sequence length.

## Consequences

- Block attention stays dense over `M` blocks; the model keeps full recall within a 1024-char window.
- "Sparse attention" remains a long-context concern, not a current one.
- The Qwen borrowing is limited to gating (shipped) and the general "divide computation" idea, which the model already realizes as block-attention (coarse) + set-attention (fine) + conditional FFN (selective).
