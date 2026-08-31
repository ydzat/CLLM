# Tasks — core block-scan reader model

Ordered, each atomic and independently verifiable. `[P]` = parallelizable. Read [spec.md](spec.md) and [architecture](../../docs/architecture.md) first.

## Phase 1 — Skeleton and data

- **T1** — Create `src/` package layout, `requirements.txt`, `configs/dev.yaml` (tiny config: `L=1, d=64, batch=2, T=4, W=64`), `tests/` dir. *Accept*: `pytest tests/` collects (even if empty).
- **T2** `[P]` — Implement char-level tokenizer + vocab (`src/data/tokenizer.py`): char ↔ id mapping, `[MASK]`, `[PAD]`, vocab size 8000. *Accept*: round-trips a sample sentence; unknown chars map to `[PAD]`.
- **T3** `[P]` — Implement 2D packing + block mask (`src/data/packing.py`): `pack(ids, W) -> (r, c)`, block assignment `b(i)`, whole-block masking. *Accept*: `N=1024, T=4` yields `M=64` blocks; 15% mask covers whole blocks only.

## Phase 2 — Model

- **T4** — Implement embedding + sparse position (`src/model/position.py`): `P[u,v]` with `R·C = N/T²` entries. *Accept*: `T=4` gives position tensor shape `(R·C, d)`, not `(N, d)` (AC-1).
- **T5** — Implement pooling + broadcast (`src/model/pool.py`): sum-pool → MLP; index-add broadcast. *Accept*: permuting block tokens leaves `z_b` unchanged (AC-3).
- **T6** — Implement global block attention (`src/model/block_attn.py`): dense over `M` blocks (mask optional). *Accept*: shape `(B, M, d)` in/out.
- **T7** — Implement router + conditional FFN (`src/model/routing.py`): top-k with straight-through, `α` fraction. *Accept*: counter shows exactly `⌈α·M⌉` blocks run FFN (AC-2).
- **T8** — Implement intra-block Set attention + slot attention output (`src/model/set_attn.py`, `src/model/slots.py`). *Accept*: `logit_{b,j}` shape `(B, M, T², V)`.
- **T9** — Assemble the full block (`src/model/block.py`) + stack (`src/model/model.py`). *Accept*: forward on tiny config returns loss.

## Phase 3 — Training and generation

- **T10** — Training loop (`src/train.py`): block-MLM loss, masked-block-only CE (AC-4), optimizer, step logging. *Accept*: `python src/train.py --config configs/dev.yaml` runs 3 steps on CPU.
- **T11** — Sampling (`src/sampling.py`): iterative mask-predict, confidence keep, `K` rounds (AC-5). *Accept*: returns full grid `≤ K` rounds.

## Phase 4 — Tests and HPC

- **T12** `[P]` — CPU smoke tests (`tests/`): position shape, permutation-invariance, router count, masked-grad, sampling termination. *Accept*: `pytest tests/` green.
- **T13** `[P]` — SLURM + Apptainer scripts (`scripts/train.slurm`, `scripts/build_image.def`). *Accept*: script references `c23g` + `--nv` + image path (AC-7).
- **T14** — HPC validation: build image, run one short `c23g` job, confirm CUDA available + loss decreases. *Accept*: one logged run completes.
