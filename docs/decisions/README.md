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
| [0007](0007-data-pipeline.md) | Data pipeline | HPC 流式读取(streaming=True，不落盘)；本机零下载用内置样例；词表一次性构建 |
| [0008](0008-pre-norm-layernorm.md) | Pre-norm LayerNorm | sum-pool 每层放大 T²，无归一化会指数爆炸；每子层加 pre-norm LN |
| [0009](0009-variable-mask-ratio.md) | Variable mask ratio | 固定 15% 掩码无法生成（100% 掩码 OOD 坍缩到逗号）；改为每 batch 采样 r~U(0,1) |
| [0010](0010-per-slot-positional-output.md) | Per-slot positional output + gated attention | 丢块内位置→掩码块 16 token 同构→slot 只能输出 1 字符→卡 unigram 熵；加 per-slot 输出偏置 + Qwen 门控 |
| [0011](0011-defer-qwen-hybrid-layering.md) | Defer Qwen hybrid layering | 块注意力已 O(M)=O(64)，线性注意力解决的是我们不存在的 O(N²)；小规模下线性注意力更差，留待长上下文 |
| [0012](0012-test-time-compute-deferred.md) | Test-time compute = K rounds | CoT 套不上非自回归；K 轮迭代即"思考开关"；自适应 K + 多样本投票留待生成验证后 |
