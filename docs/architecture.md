# Architecture — the block-scan reader

## Summary

The block-scan reader is a non-autoregressive language model that reads a Chinese text as a 2D grid of `T×T` blocks rather than a 1D token stream. Position is sparse: defined only at block granularity, with tokens inside a block treated as a permutation-invariant set. Compute is uneven: a top-k router applies the FFN only to high-semantic-load blocks. Generation is discrete diffusion (iterative mask-predict), so output is a reconstructed "reasonable combination", not a left-to-right stream. The one knob that trades order precision against compute is the block size `T`.

## Table of contents

- [Two-level state](#two-level-state)
- [Input and 2D layout](#input-and-2d-layout)
- [Blocks](#blocks)
- [Embedding and sparse position](#embedding-and-sparse-position)
- [Layer](#layer)
- [Output: slot attention](#output-slot-attention)
- [Training: block MLM](#training-block-mlm)
- [Generation: discrete diffusion](#generation-discrete-diffusion)
- [Core tradeoff: block size T](#core-tradeoff-block-size-t)
- [Complexity](#complexity)
- [Dev Note](#dev-note)

## Two-level state

The model holds two persistent states per layer:

- Token state `h_i ∈ R^d` for each of `N` characters.
- Block state `z_b ∈ R^d` for each of `M = ⌈N/T²⌉` blocks, derived from token state by pooling.

The block state is the unit of global (cross-block) computation; the token state is materialized for fine-grained (within-block) computation, conditionally.

## Input and 2D layout

Character-level tokens `x = (x_1, …, x_N)`, `x_i ∈ V`, laid out in lines of fixed width `W`:

```
r_i = ⌊i / W⌋,  c_i = i mod W
```

Height `H = ⌈N/W⌉`, padded to `H×W`.

## Blocks

`T×T` tiles partition the grid:

```
u_i = ⌊r_i / T⌋,  v_i = ⌊c_i / T⌋,  b(i) = (u_i, v_i)
```

Block `(u,v)` contains up to `T²` characters. `M = R·C` blocks on the coarse grid `R×C`.

## Embedding and sparse position

```
h_i^(0) = Embed(x_i) + P[u_i, v_i],   P ∈ R^{R×C×d}  (learnable)
```

`P[u_i, v_i]` is the **only** positional signal. It is block-granular (`R·C = N/T²` positions, not `N`), and every token in a block shares it, so tokens inside a block are permutation-invariant. `T=1` degenerates to per-token 2D position; larger `T` is sparser.

## Layer

Each of `L` layers runs five steps. Steps (a)–(d) are the cheap "scan" (unconditional); step (e) is the expensive "read" (conditional).

**(a) Pool (coarse scan, DeepSets)**

```
z_b = MLP_pool( Σ_{i∈b} h_i^(ℓ−1) )  ∈ R^d
```

**(b) Global block attention (jump, fixed sparse mask)**

```
S(b) = { b' : max(|u−u'|,|v−v'|) ≤ 1 } ∪ { b' : v'=v, u'−u ∈ {±2,±3,±4} } ∪ { b' : u'=u, v'−v ∈ {±2,±3} }
z_b ← z_b + Attn( z_b ; { z_{b'} : b' ∈ S(b) } )
```

The three terms are the 3×3 neighbourhood, vertical skip, and horizontal skip. `|S| ≈ 19`. When `M` is small (≤ a few hundred) this is computed as dense attention over all blocks; the mask is an ablation knob.

**(c) Broadcast**

```
h_i^(ℓ) = h_i^(ℓ−1) + z_{b(i)}
```

**(d) Router**

```
s_b = W_r z_b ∈ R,   K = top-k_b { s_b }
```

**(e) Conditional read (only b ∈ K, fraction α = k/M)**

```
h_i^(ℓ) ← h_i^(ℓ) + SetAttn( h_i^(ℓ) ; { h_j^(ℓ) : j ∈ b } )   // content attention, no positional bias
h_i^(ℓ) ← h_i^(ℓ) + FFN( h_i^(ℓ) )
```

Skipped blocks are only broadcast-updated; they never run the FFN. See [0003](decisions/0003-content-based-intra-block-interaction.md) and [0004](decisions/0004-conditional-ffn-moe-routing.md).

## Output: slot attention

Encoding is a set; output is an ordered reconstruction. Each slot `q_j` (`j = 0..T²−1`) selects its character from the block by content:

```
a_{j,i} = softmax_i( q_jᵀ W_k h_i / √d )
logit_{b,j} = W_o Σ_i a_{j,i} W_v h_i
```

Slots are the "reasonable combination" positions; which character fills a slot is decided by content, not by position.

## Training: block MLM

Mask a random 15% of whole blocks; predict each masked block's characters at its slots. Loss is cross-entropy over masked blocks. Whole-block masking forces reconstruction of an entire block from its coarse position and neighbour context.

## Generation: discrete diffusion

Iterative mask-predict: initialise the grid to `[MASK]`, predict all tokens in one parallel forward pass, keep high-confidence predictions, re-mask the rest, repeat for `K ≈ 8–16` rounds. `O(K)` forwards replace `O(N)` autoregressive steps. See [0005](decisions/0005-discrete-diffusion-generation.md).

## Core tradeoff: block size T

Larger `T` → fewer blocks (`M = N/T²`), more pooling/skip efficiency, and more order loss inside a block (order-only contrasts such as 狗追猫 vs 猫追狗 fall into one block and are resolved by prior, not position). Smaller `T` → the reverse. `T` is the single knob for the central hypothesis H1: Chinese tolerates coarser position than English at equal quality.

## Complexity

Per layer, dense transformer vs block-scan reader:

| Part | Dense | Block-scan |
|---|---|---|
| Attention | `O(N²·d)` | `O(M·|S|·d) + O(αN·T²) ≈ O(N)` |
| FFN | `O(8N·d²)` | `O(8α·N·d²)` |
| Pool + broadcast | — | `O(N·d)` |
| Generation | `O(N)` steps | `O(K)` passes |

## Dev Note

None.
