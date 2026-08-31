# 0007 — Data pipeline: stream on HPC, zero-download locally

## Problem

How does char-level corpus data flow into training, given (a) the real corpus (SkyPile + FineWeb-Edu + Stack + OpenWebMath) totals ~600GB+, (b) local dev is a low-spec Windows box, and (c) training happens on HPC?

## Decision

- **HPC**: stream the corpus via HuggingFace `datasets` with `streaming=True` — rows are fetched and discarded as consumed, so the raw corpus is **never written to disk**. Char-level tokenization is trivial (no BPE training), so streaming + on-the-fly char→id mapping is IO-bound, not compute-bound.
- **Local**: **never download corpus**. Verify the pipeline end-to-end with a tiny hardcoded sample (a few Chinese sentences), zero data download.
- **Vocab**: built once, on HPC, from a streamed sample (~100M chars), saved as JSON, reused for all training.
- **Cache (optional)**: tokenized+packed ids can be cached to `$HPCWORK` for faster later epochs; not required for v1.

## Alternatives considered

- **Download raw corpus to disk first**: rejected — ~600GB, and streaming removes the need.
- **Download a small local subset**: rejected — still tens of GB, and local dev only needs to verify *code*, not *data*.
- **On-the-fly BPE tokenizer training**: n/a — char-level tokenization needs no tokenizer training.

## Consequences

- Local dev downloads zero data (the `datasets` import is lazy and not even installed locally).
- HPC stores no raw corpus; it streams and optionally caches tokenized ids on `$HPCWORK`.
- Vocab must be built before training: a one-time HPC job over ~100M streamed chars.
- The lazy `datasets` import means `corpus.py` and `build_vocab.py` import fine without `datasets` installed (only `iter_text_hf` needs it, and it imports inside the function).
