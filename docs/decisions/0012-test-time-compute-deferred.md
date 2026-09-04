# Decision 0012 — Test-time compute maps to K rounds; defer the rest

## Problem

Modern LLMs split along a "fast vs reasoning" axis = test-time compute (spend more inference compute for better output). Can the block-scan reader add a "reasoning" capability?

## Decision

Chain-of-thought reasoning does not transfer — it is autoregressive (generate a left-to-right thinking trace), and the block-scan reader is non-autoregressive. The block-scan reader's test-time compute knob is the diffusion round count `K`: small `K` = fast, large `K` = deeper iterative refinement. Keep `K` as the single knob for now. Defer adaptive `K` (spend more rounds on uncertain blocks) and self-consistency (multi-sample voting) until generation is validated.

## Alternatives considered

- **CoT-style reasoning**: inapplicable to non-autoregressive masked-predict generation.
- **Adaptive K now**: premature — generation quality is not yet confirmed, so there is nothing to optimize.
- **Self-consistency now**: cheap but pointless before single-sample generation is coherent.

## Consequences

- `K` (8–16, configurable) is the explicit fast/reasoning switch, analogous to thinking/non-thinking in autoregressive models.
- After generation is validated, the two deferred additions (adaptive `K`, multi-sample voting) are the natural test-time-compute extensions, matching the industry trend of sparse compute + test-time compute.
