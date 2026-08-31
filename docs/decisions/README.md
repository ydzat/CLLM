# Decisions — index

Each file records one non-trivial design choice in the fixed form **Problem / Decision / Alternatives considered / Consequences**. A decision states the shipped (or committed) choice in present tense and what was given up; it never reads as a "should" or a migration plan. Add one file per new decision; number sequentially.

| # | Decision | One-line |
|---|---|---|
| [0001](0001-char-level-tokenization.md) | Char-level tokenization | 中文"词"是派生单位，字符级让模型自发现词边界，信息密度最大 |
| [0002](0002-sparse-2d-position-and-block-size.md) | 2D grid + block-granular position | 位置只在块粒度定义，块内置换不变；T 是"顺序精度↔算力"单一旋钮 |
| [0003](0003-content-based-intra-block-interaction.md) | Content-based intra-block interaction | 块内交互必要，但必须内容型（置换不变），不能用位置型 |
| [0004](0004-conditional-ffn-moe-routing.md) | Conditional FFN (top-k routing) | FFN 只跑在语义重载的块上，是短上下文里的真·速度杠杆 |
| [0005](0005-discrete-diffusion-generation.md) | Discrete-diffusion generation | 非自回归迭代掩码填充，O(K) 前向替代 O(N) 步；图生成领域的结论 |
| [0006](0006-dataset-choice.md) | Pretraining dataset | 中文为主(~65%) + 英文/代码/数学；SkyPile + FineWeb-Edu + Stack + OpenWebMath；追求中文性能而非对照 |
