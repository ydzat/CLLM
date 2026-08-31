# 0006 — Pretraining dataset choice

## Problem

Which corpus does the block-scan reader train on? The model is Chinese-motivated (char-level, 2D block-scan reading), and the goal is the **best final Chinese model performance** — an experiment, not a Chinese-vs-English comparison paper.

## Decision

Train on a **Chinese-dominant** mix (~2B tokens total), sampled from open corpora:

| Slice | Share | Corpus |
|---|---|---|
| Chinese web | 50% | SkyPile-150B subset |
| Chinese books + wiki | 15% | WuDao/MNBVC book portions + zhwiki |
| English web | 15% | FineWeb-Edu |
| Code | 12% | The Stack v2 |
| Math | 8% | OpenWebMath / Proof-Pile-2 |

**Why this mix**: Qwen and DeepSeek disclose no public corpus (only DeepSeek-V2 gave a ratio — Chinese ≈ English + 12%); they disclose only *categories* (web, Chinese web, books, code, math, all with code/math upweighted). Since the goal is Chinese performance, the mix is Chinese-dominant (~65%), not balanced. English + code + math are kept because (a) Chinese text is full of English terms/borrowings, (b) code and math capability transfers, and (c) both Qwen and DeepSeek explicitly upweighted code/math.

## Alternatives considered

- **mC4 bilingual (controlled zh/en)**: rejected — that split existed to make a *comparison* experiment; this project is not one.
- **50/50 balanced zh/en**: rejected — unnecessary English for a Chinese model.
- **Pure Chinese (no English)**: rejected — loses code/math capability and English-terminology robustness.

## Consequences

- Chinese-dominant → best expected Chinese downstream performance.
- No English control group → the sparse-position hypothesis is evaluated as "block-scan model beats a standard transformer on **Chinese** benchmarks", not "Chinese beats English".
- Data pipeline needs: SkyPile, FineWeb-Edu, The Stack v2, OpenWebMath, zhwiki — all fetchable via HuggingFace `datasets`; ~2–5GB text before char-level tokenization.
- 60M params at ~2B tokens is near Chinchilla-optimal (~1.2B), so quantity is not the bottleneck; quality filtering (dedup, boilerplate removal) matters more than corpus choice.
- SkyPile is noisy relative to books; the Chinese book slice is deliberately kept to give the block-scan reader long-form, well-formed text.
