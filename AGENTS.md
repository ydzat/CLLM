# AGENTS.md

CLLM is a research project building a **block-scan reader** — a non-autoregressive language model that reads text as a 2D grid of blocks with sparse position, motivated by Chinese reading (high information density, block scanning, order tolerance). Read [docs/architecture.md](docs/architecture.md) before changing `src/`; rationale lives in [docs/decisions/](docs/decisions/README.md).

## Repository layout

```
src/           block-scan model, routing, sampling, data pipeline
specs/         what-to-build: spec.md (requirements + acceptance) + tasks.md per feature
docs/          architecture (design), development (tooling), decisions (rationale)
scripts/       SLURM job + Apptainer build
tests/         CPU smoke tests
configs/       YAML config (model, data, training)
```

## Commands

```sh
pytest tests/                          # CPU smoke tests (tiny model: L=1, d=64, batch=2)
python src/train.py --config configs/dev.yaml   # local CPU sanity run (NOT training)
# training runs on HPC only, via scripts/train.slurm (see docs/development.md)
```

## Conventions

- Python 3.11; PyTorch. Follow the model in [docs/architecture.md](docs/architecture.md); do not diverge without updating it in the same change.
- Every non-trivial design choice has a record in [docs/decisions/](docs/decisions/README.md) — Problem / Decision / Alternatives / Consequences.
- Document **current state, not change history**. No "previously / now / decided to" in durable prose; change stories go in commits.
- Char-level tokenization only — [decisions/0001](docs/decisions/0001-char-level-tokenization.md).
- Position is block-granular 2D only; tokens within a block are permutation-invariant — [decisions/0002](docs/decisions/0002-sparse-2d-position-and-block-size.md).
- Intra-block interaction is content-based (permutation-invariant), never positional — [decisions/0003](docs/decisions/0003-content-based-intra-block-interaction.md).
- FFN is conditional (top-k routed), never uniformly applied to every token — [decisions/0004](docs/decisions/0004-conditional-ffn-moe-routing.md).
- Generation is discrete diffusion (iterative mask-predict), never autoregressive — [decisions/0005](docs/decisions/0005-discrete-diffusion-generation.md).
- One home per fact: a fact lives in exactly one doc; everywhere else links to it. Grep a distinctive phrase before duplicating it.
- Concrete prose, no metaphors. Name exact tensors, shapes, and types. No reasoning transcripts in comments or docs.
- No `as any`-style type escapes; no silent `except:`; name what an empty `catch` swallows.
- Training/checkpoints/data never enter git (see `.gitignore`); they live on HPC `$HPCWORK`/`$WORK`.

## Editing these instructions

Edit this file. Keep each rule 1–3 lines and link its home; condense when clarity survives.
