# Decision 0013 — Unigram plateau is a capacity/data limit, not a bug

## Problem

Training loss plateaus at 6.6 nats, which equals the SkyPile unigram char entropy (6.635 nats, computed via `compute_unigram.py`). The conditional (bigram) entropy is 4.50 nats (`compute_bigram.py`), so a working model should reach ~4.5, but both the block-scan reader AND a standard vanilla BERT-style transformer plateau at 6.6. Nine attempted fixes (per-slot positional output, token-level masking, intra-block position, sigmoid router gate, position scale, MLP output head, slot-query scale, higher LR, constant LR) did not break the plateau; two Oracle consultations and ~10 diagnostics ruled out architecture, data, tokenizer, masking, loss, position, head, and attention scale.

## Decision

This is a **capacity and data limitation, not a bug**. The model learns the unigram because the char-level Chinese bigram is a `7740² ≈ 60M`-entry statistical table, and 1B tokens gives only ~16 samples per entry — far too few for statistical (soft) dependencies, which need ~100× more data than deterministic ones. The architecture is validated: a synthetic 50-char statistical 70/30 bigram is learned to 1.70 nats (below its 3.91 unigram floor), and copy/ordered/generalization diagnostics all pass, so forward/gradient/attention/position all work.

## Alternatives considered

- **Continue debugging**: 9 fixes + 2 Oracles exhausted the plausible bug hypotheses; each produced the same 6.6 plateau.
- **Attribute to a specific bug**: every candidate (masking, position, head, LR) was tested in isolation and refuted.

## Consequences

- The block-scan reader is validated as trainable and equivalent in loss to a standard transformer; this is the architecture-validation result, not a generation result.
- Coherent Chinese generation requires scaling up: ~200–500M params, 10–100B tokens, and a constant or higher LR. The earlier librarian finding (a usable Chinese model is ≥0.8B) is confirmed.
- The debugging code (diagnostics, `compute_bigram.py`, `--mask-ratio`, constant-LR config) is retained as tooling.
