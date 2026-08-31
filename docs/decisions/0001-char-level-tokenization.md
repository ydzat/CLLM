# 0001 — Char-level tokenization

## Problem

How should Chinese text be split into model input units? Chinese has no whitespace-delimited words; "词" is a derived segmentation, not a native unit marked in text.

## Decision

Tokenize at the **character** level: one Chinese character = one token (`V` ≈ 8000).

## Alternatives considered

- **Subword BPE** (used by Qwen/MiniCPM): smaller sequences, but imposes a learned segmentation that bakes in word-boundary assumptions and English-oriented merges.
- **Word-level** (requires jieba/THULAC segmentation): external dependency, propagates segmentation errors, and contradicts the premise that word boundaries are derived.

## Consequences

- Maximum information density per token — directly the "information density" property being exploited.
- No external segmentation dependency; the model must discover word-like units itself, consistent with "词是后训练出来的".
- Small, fixed vocabulary simplifies the embedding and output heads.
- Loses explicit multi-character word signal; this is recovered by content-based intra-block interaction ([0003](0003-content-based-intra-block-interaction.md)).
