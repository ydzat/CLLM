# Decision 0013 — Unigram plateau is a transient local minimum; escaping it needs a high stable LR (WSD)

## Problem

Training loss plateaus at the corpus unigram entropy — **6.635 nats** for char-level SkyPile, **8.37 nats** for the 30K BPE vocab — while the conditional (bigram) entropy is far lower (**4.50** char / **5.26** BPE). A masked position's top predictions stay at the most frequent tokens regardless of context. The plateau is identical for the block-scan reader and for a vanilla BERT-style transformer, across char and BPE tokenisation, and across 60M and 300M models, so it is not architectural.

## Decision

The plateau is a **transient unigram local minimum** — a documented phase of transformer training (uniform → unigram → bigram; induction heads form late — Edelman et al. 2024, arXiv:2402.11004). Escaping it is a **phase transition**: loss sits flat at the unigram entropy for thousands of steps, then drops sharply once content-addressed attention forms. The escape requires the learning rate to stay **high and non-decaying** through that window.

Training uses the **WSD schedule** (`src/train.py: build_scheduler`): linear warmup → a long constant "stable" phase at `lr` → linear decay to `min_lr` over the final `decay_steps`. Defaults: `lr=1e-3`, `min_lr=1e-4`, `decay_steps = steps // 10`, `warmup_steps=2000`.

Resuming must reset the optimizer's LR. `load_state_dict` restores the checkpoint's stale (decayed) `param_groups["lr"]` **and** `param_groups["initial_lr"]`, and PyTorch schedulers read `initial_lr`; `--resume` sets both to the config `lr` before building the schedule, otherwise the WSD "stable" phase silently runs at the old floor.

## Alternatives considered

- **Capacity/data limit** (60M char-level too small; scale to 200–500M + 10–100B tokens): refuted — the vanilla transformer and the 300M BPE model plateau identically, and a synthetic heterogeneous bigram escapes unigram with a d=128/L=2 model.
- **Cosine decay** (LLaMA/GPT-3 recipe): decays the LR continuously and reaches the floor before the escape point; the char-level run with cosine → 3e-5 never left unigram.
- **Constant LR**: escapes, but has no decay phase for convergence.
- **Unigram-bias head init** (Meister et al. 2022): tested; does not break the plateau.
- **Smaller local attention window** (Makkuva et al. 2024): deferred; the block-scan saccade attention is already local.

## Consequences

- The block-scan reader is validated: it trains identically to a vanilla transformer and learns the deterministic and homogeneous synthetic tasks; the plateau is a training-schedule artifact, not a model defect.
- A resumed run is expected to sit flat at the unigram entropy and then drop; the monitor watches for that phase transition (and for the `lr` line, which must read `1.00e-03` in the stable phase).
- The diagnostic suite (copy / ordered-reconstruction / heterogeneous-bigram, `compute_bigram.py`, `--resume`, `--mask-ratio`) is retained as tooling.
