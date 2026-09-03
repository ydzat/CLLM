# Decision 0009 — Variable mask ratio (LLaDA-style) for generation

## Problem

The first trained checkpoint produced only `，` (comma) characters when asked to generate text. Training masked a fixed 15% of whole blocks; unconditional generation starts from an all-`[MASK]` grid. A model trained at a single mask ratio never sees 100% masking, so the all-`[MASK]` input is out of distribution and the model collapses to the most frequent token. Fixed-ratio masked LM (BERT-style) can predict masked tokens but cannot generate; this contradicts [0005](0005-discrete-diffusion-generation.md)'s claim of iterative mask-predict generation.

## Decision

Sample the block-mask ratio per batch as `r ~ U(0,1)` and mask `round(r · M)` whole blocks. The model is therefore trained across the full masking range, making the all-`[MASK]` starting point of generation in-distribution. This is exactly the LLaDA / Mask-Predict training recipe, with blocks in place of tokens. `mask_whole_blocks` already clamps the count to `≥ 1` block (`max(1, round(m·r))`), so `r = 0` is safe.

## Alternatives considered

- **Keep fixed 15% + prompt-completion demo only** — the model can only ever fill ~15% of a sequence, so true unconditional generation is impossible without retraining.
- **Partial-unmasking schedules** (e.g. unmask on a fixed schedule) — these fix the *sampling* loop, not the *training* distribution; a 15%-trained model still collapses when the mask fraction exceeds what it saw.
- **Autoregressive decoding from the same weights** — contradicts the non-autoregressive premise ([0005](0005-discrete-diffusion-generation.md)).

## Consequences

- One-line change in the training loop (`rng.random()`), plus removing the fixed `mask_ratio` config field.
- The model now sees all mask ratios, so iterative mask-predict generation is in-distribution.
- Requires retraining (~55 min on H100 + queue). The previous checkpoint (fixed 15%) is superseded.
- Loss is now averaged over a varying number of masked tokens per step (noisier at small `r`), which AdamW absorbs over many steps.
