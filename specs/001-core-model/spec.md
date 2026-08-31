# Spec — core block-scan reader model

## Summary

The core model reads character-level Chinese text as a 2D grid of `T×T` blocks with block-granular position, routes the FFN to a fraction of blocks, and generates by iterative mask-predict. This spec fixes **what** the model must do and how it is verified; **how** it is implemented lives in [architecture](../../docs/architecture.md) and [decisions](../../docs/decisions/README.md).

## Goals

1. A trainable non-autoregressive model whose input is a 2D block grid with sparse (block-granular) position.
2. A conditional-computation path: the FFN runs on only `α` of blocks per layer.
3. A discrete-diffusion sampler that produces an ordered output from a permutation-invariant encoding.
4. A smoke-testable codebase that runs on CPU (Windows) and trains on HPC (H100/Apptainer).

## Functional requirements

- **FR-1** — The model accepts a character sequence and lays it out as an `H×W` grid partitioned into `T×T` blocks.
- **FR-2** — Position is a learnable embedding indexed by block `(u,v)`; every token in a block shares it. No per-token position exists.
- **FR-3** — Each layer pools block tokens to a block state, applies global block attention, broadcasts back, routes a subset of blocks, and runs intra-block Set attention + FFN only on routed blocks.
- **FR-4** — Training masks whole blocks (15%) and predicts masked blocks' characters at their slots; the loss is cross-entropy over masked blocks only.
- **FR-5** — Generation is iterative mask-predict: parallel prediction, confidence-threshold keep, re-mask, repeat `K` rounds.
- **FR-6** — A tiny configuration (`L=1, d=64, batch=2`) runs forward + backward + one sampling round on CPU without error.

## Acceptance criteria

- **AC-1** — Given `T=4` and `N` characters, the model instantiates `⌈N/T²⌉` block positions, not `N` (sparse position is real, verified by a unit test on the position tensor shape).
- **AC-2** — Given routing fraction `α`, exactly `⌈α·M⌉` blocks execute the FFN per layer; skipped blocks forward only their block state (verified by a counter in a unit test).
- **AC-3** — Given a block of tokens, permuting its internal order does not change the block state `z_b` (permutation-invariance, verified by a unit test).
- **AC-4** — Given a masked block, gradient flows only through masked-block slot logits (verified by checking non-masked positions have zero grad).
- **AC-5** — Given `K` sampling rounds on a tiny model, the sampler returns a full grid (all tokens unmasked) and terminates in `≤ K` rounds.
- **AC-6** — `pytest tests/` passes on CPU; `python src/train.py --config configs/dev.yaml` completes 3 steps without error.
- **AC-7** — `scripts/train.slurm` submits to partition `c23g` and runs inside the Apptainer image (`--nv`), training and logging loss per step.

## Non-goals (explicit)

- No instruction tuning or RLHF.
- No decoding-time search (beam / contrastive).
- No multi-modal input or vision encoder.
- No production inference optimization (KV-cache, quantization, kernel fusion) in this feature.
- No long-context (> 4K) evaluation until the core model trains.

## Key entities

Character `x_i`, token state `h_i ∈ R^d`, block state `z_b ∈ R^d`, block `(u,v)`, slot `q_j`, router score `s_b`. Schemas (config, checkpoint, data) are specified in `contracts/` when implemented.
