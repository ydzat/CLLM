# CLLM — 块扫描阅读器（Block-Scan Reader）

[English](README.md) · [中文](README.zh-CN.md)

一个非自回归语言模型：把文本当作 `T×T` 块的二维网格来读，而不是一维 token 流。位置是稀疏的（块粒度、块内置换不变），FFN 是条件化的（top-k 路由到部分块），生成是离散扩散（迭代掩码预测）。动机来自中文阅读：信息密度高、逐块扫视、对顺序宽容。

> **状态** — spec 驱动开发。~60M 参数核心模型、训练循环、采样、流式数据管线均已实现并通过 CPU 冒烟测试。训练在 RWTH CLAIX（H100）上进行，见 [development](docs/development.md)。

## 是什么

块扫描阅读器把文本当作"类图结构"而非"类字符串"。固定行宽 `W` 的重排把字符铺成 `H×W` 网格，再切成 `T×T` 块。一个块粒度位置向量由块内所有 token 共享；块内部，token 构成置换不变集合。top-k 路由器只对语义重载的块施加 FFN。有序输出由 slot attention 从无序块集合重建，通过迭代掩码预测生成。

完整设计（数学、张量、复杂度）在 [architecture.md](docs/architecture.md)；每个非平凡选择都有记录 [decisions/](docs/decisions/README.md)。

## 关键设计点

| # | 选择 | 出处 |
|---|------|------|
| 1 | 字符级分词——"词"是派生单位，由模型自发现 | [0001](docs/decisions/0001-char-level-tokenization.md) |
| 2 | 二维网格 + 块粒度稀疏位置；`T` 是"顺序精度↔算力"旋钮 | [0002](docs/decisions/0002-sparse-2d-position-and-block-size.md) |
| 3 | 块内交互是内容型的（置换不变），绝不用位置型 | [0003](docs/decisions/0003-content-based-intra-block-interaction.md) |
| 4 | 条件化 FFN（top-k 路由）——FFN 只跑在 `α` 比例的块上 | [0004](docs/decisions/0004-conditional-ffn-moe-routing.md) |
| 5 | 生成是离散扩散——迭代掩码预测，`O(K)` 次前向 | [0005](docs/decisions/0005-discrete-diffusion-generation.md) |
| 6 | 预训练数据——中文为主（~65%）+ 英文/代码/数学 | [0006](docs/decisions/0006-dataset-choice.md) |
| 7 | 数据管线——HPC 流式读取，本机零下载 | [0007](docs/decisions/0007-data-pipeline.md) |

## 仓库结构

```
specs/        要建什么：每个特性的 spec.md + tasks.md（先 spec 后码）
docs/         架构（设计）、开发（工具链）、决策（理由）
src/          模型、数据管线、训练、采样
tests/        CPU 冒烟测试（极小模型：L=1, d=64, batch=2）
configs/      YAML 配置（模型、数据、训练）
scripts/      SLURM 作业 + Apptainer 构建
```

## 快速开始

**本地——仅 CPU 冒烟测试（从不训练）：**

```bash
pip install -r requirements.txt
pytest tests/                                  # 极小模型冒烟测试
python src/train.py --config configs/dev.yaml  # 3 步 sanity 运行
```

**训练——仅 HPC。** 见 [development.md](docs/development.md) 的 Git 同步 → Apptainer → SLURM 流程。`scripts/train.slurm` 提交冒烟作业；`scripts/train_real.slurm` 在流式 SkyPile 上训练完整模型。

## 文档索引

- **做什么 / 为什么**（需求 + 验收）：[specs/001-core-model/spec.md](specs/001-core-model/spec.md)
- **怎么做**（设计、数学）：[docs/architecture.md](docs/architecture.md)
- **每个选择的理由**（问题 / 决策 / 备选 / 后果）：[docs/decisions/README.md](docs/decisions/README.md)
- **工具链 + 工作流**：[docs/development.md](docs/development.md)

## 许可证

MIT。
