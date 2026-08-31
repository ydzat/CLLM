# 002 · 技术栈与开发-训练工作流

> 状态：v1.0（§10 已回填 RWTH CLAIX 实测参数）
> 日期：2026-08-31
> 约束：开发机 Windows 11（低配，不训练，仅基础运行调试）；训练机 RWTH HPC（Linux，仅支持 Apptainer，不支持 Docker）。

---

## 1. 总体架构

```
┌─────────────────────────┐        git push / pull          ┌──────────────────────────┐
│  Windows 11（开发机）    │ ──────────────────────────────► │  RWTH HPC（Linux）        │
│  写代码 / 单元测试        │ ◄────────────────────────────── │  SLURM + Apptainer        │
│  CPU 冒烟测试            │      rsync 回传 checkpoint/log    │  训练 / 评测 / 数据预处理  │
└─────────────────────────┘                                 └──────────────────────────┘
```

**核心原则**：本机只做「能抓 bug 的最小验证」，一切重活（训练、大数据预处理）都在 HPC。

---

## 2. 开发机（Windows 11）

职责：写代码、单元测试、**CPU 冒烟测试**（极小模型跑通前向/反向/采样，抓 shape/语法/逻辑 bug），**不训练**。

- **Python**：3.11（与 HPC 一致）
- **PyTorch**：CPU 版，`pip install torch --index-url https://download.pytorch.org/whl/cpu`（仅冒烟测试用，不要求 GPU）
- **环境管理**：venv 或 miniconda（建议 miniconda，与 HPC 内环境一致性好）
- **编辑器**：VS Code
- **其他**：git（必装）、rsync（WSL 内或 `winget install`，用于回传 checkpoint/log）

冒烟测试的标准：模型层数 L=1、d=64、块 T=4、batch=2，跑 3 步前向+反向+采样，确认无报错即可。

---

## 3. HPC 训练环境（Linux + Apptainer）

职责：训练、评测、数据预处理。

- **容器运行时**：Apptainer（Singularity 的继任者，RWTH 官方容器方案，系统自带、无需 module load）。**「appcontainer」不是 RWTH 术语**——RWTH FAQ 明确「不支持 Docker，用 Apptainer」；「AppContainer」是 Microsoft Windows 的沙箱概念，是你记混了。
- **镜像格式**：单个 `.sif` 文件。
- **GPU**：待确认（§10）。

---

## 4. 代码同步工作流（核心）

### 主方案：Git（单一事实来源）

1. 本地 `git init`（当前 `D:\Workspace\CLLM` 尚未是 git 仓库）。
2. 推到 **GitHub/GitLab 私有仓库**（或 RWTH 提供的 GitLab，如有）。
3. HPC login 节点 `git clone`，之后每次 `git pull`。
4. 版本化 + 可回滚 + 天然同步，避免「两台机器哪个是新的」的混乱。

**同步节奏**：

```
本地写代码 → CPU 冒烟测试通过 → git commit + push → HPC git pull → SLURM 提交小规模验证 → 结果回传
```

### 辅助方案：VS Code Remote-SSH（可选，强烈推荐）

- 用 VS Code 直接 SSH 连 HPC login 节点，**在 HPC 上编辑/调试**，省掉来回同步。
- 适合「需要 GPU/完整环境才能调试」的场景（本机低配跑不动时）。
- 与 Git 不冲突：在 HPC 上改完，直接在 HPC 端 commit/push。

### 回传：rsync

- 训练 checkpoint / 日志 / 评测结果用 `rsync -av hpc:/path/to/ckpt ./local/` 拉回本地分析。

---

## 5. 容器方案（Apptainer，无 Docker）

### 5.1 基础镜像选择

| 来源 | 镜像 | 说明 |
|---|---|---|
| NVIDIA NGC | `nvcr.io/nvidia/pytorch:26.07-py3` | **推荐**：CUDA 13.3.1 + PyTorch 2.13，Ubuntu 24.04，Hopper 优化 |
| NVIDIA NGC（稳） | `nvcr.io/nvidia/pytorch:26.05-py3` | 落后一个月、更稳（CUDA 13.2.1 + PyTorch 2.12） |
| Docker Hub | `pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime` | 通用、体积较小 |

H100（Hopper）下 NGC 26.x 均可。RWTH 自己也提供预构建 PyTorch 容器 module（`module spider PyTorch` 查看），追求零配置可优先用它。

### 5.2 构建方式（关键：Windows 上不能直接 build Apptainer）

Apptainer 构建需要 Linux 环境。三条路：

1. **在 HPC 上直接 build（推荐）**：`apptainer build torch.sif docker://nvcr.io/nvidia/pytorch:2x.y-py3`（需 HPC 提供 build 权限或 build 节点；RWTH 政策待确认）。
2. 在 HPC 上用 definition 文件 build：把 `pip install -r requirements.txt` 写进 `.def`，一步到位烤进镜像。
3. ❌ 本地 WSL2 + Apptainer 构建再传 `.sif`——理论上可行但 WSL2 里 Apptainer 支持不稳，不推荐。

### 5.3 运行

```bash
# 挂载代码目录 + 数据目录，启用 GPU
apptainer exec --nv \
  -B /path/to/code:/code -B /path/to/data:/data \
  torch.sif python /code/train.py
```

或 `apptainer run --nv torch.sif ...`。`--nv` 是关键（透传 NVIDIA 驱动）。

### 5.4 依赖装哪

- **烤进镜像**：`requirements.txt` 在 build 时 pip install，镜像自包含、可复现。缺点：改依赖要重建镜像。
- **挂载 venv**：HPC 上 `apptainer exec` 里建 venv 到 `$WORK`，运行时挂载。迭代快，但可复现性差。
- **建议**：核心依赖（torch、transformers、cuda）烤进镜像；项目自己的小改动用挂载的代码目录（无需重建镜像）。

---

## 6. 依赖与版本

- **Python**：3.11
- **PyTorch**：2.x + CUDA 12.x（与 HPC GPU 匹配，§10 定）
- **核心库**：transformers、tokenizers、datasets（沿用 dllm / nanoLLaDA 的技术选型）
- **复用组件**（见 001 §11）：dllm（训练循环）、lucidrains/slot-attention、nanoLLaDA（60M 参考）
- **实验追踪**：wandb（若 HPC 有外网）或 tensorboard + 本地日志（无网兜底）
- **版本锁定**：`requirements.txt` 精确 pin；PyTorch/CUDA 版本记录在镜像 tag 上

---

## 7. SLURM 作业

```
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
module purge
apptainer exec --nv $HPCWORK/pytorch-26.07-py3.sif python train.py
```

两阶段：
1. **小规模验证**（1 GPU、短时、小数据）：验证代码在 HPC 真环境跑通。
2. **正式训练**（多 GPU，如需）：60M 模型单卡即可，多卡用 DDP。

---

## 8. 数据管线位置

- **语料**：SkyPile / Wudao / CCI 子集（中文）。
- **预处理**（字符切分 + 2D 打包 + 整块掩码）：**全部在 HPC 上跑**，本机低配不做。
- **存储**：HPC `$WORK` / scratch 目录（具体路径 §10 回填）。预处理产物（打包好的 tensor）落盘复用，避免每次重算。

---

## 9. 目录结构（建议）

```
CLLM/
├── doc/                  # 设计/技术文档（001、002、…）
├── src/
│   ├── model.py          # 块扫描阅读器（核心，~300 行）
│   ├── routing.py        # MoD 路由（~30 行）
│   ├── sampling.py       # 迭代掩码采样（抄 nanoLLaDA/LLaDA）
│   └── data.py           # 字符级 2D 打包 + 整块掩码
├── scripts/
│   ├── build_image.def   # Apptainer 定义文件
│   └── train.slurm       # SLURM 作业脚本
├── tests/                # 冒烟测试
├── requirements.txt
└── README.md
```

---

## 10. RWTH CLAIX 已确认事实（2026-08 核实）

### 10.1 集群
- **活跃**：CLAIX-2023（通用，Intel Sapphire Rapids + H100）、CLAIX-2025（AMD Turin + H100，仅 NHR/WestAI 项目）。
- **已退役**：CLAIX-2018（2024-05-31 退役）、CLAIX-2016。「CLAIX-2018-MIG」不存在——MIG 只出现在 GPU login 节点和 JupyterHub `c23i` 分区。
- **目标集群：CLAIX-2023**（普通项目可达）。

### 10.2 GPU 与 CUDA
- 只有 **H100**：CLAIX-2023-ML = 4× H100 **96GB** HBM2e/节点；CLAIX-2025-ML = 4× H100 **80GB** HBM3/节点。**无 A100、无 L40S**。

### 10.3 容器
- **Apptainer**（Singularity 继任者），系统自带、无需 module load。**Docker 明确禁止**。「appcontainer」不是 RWTH 术语（=你记混的 Windows AppContainer 沙箱）。

### 10.4 SLURM 分区
- CLAIX-2023：`c23ms`(默认) / `c23mm` / `c23ml` / **`c23g`(GPU)** / `devel`(免费、1 小时、无需项目)。
- CLAIX-2025：`c25ms` / `c25ml` / `c25g`。
- GPU 计费：1 GPU 时 = 24 核时；1 GPU 限 24 核 + 122GB。

### 10.5 存储
- `$HOME` /home/ **250GB**（备份+快照）；`$WORK` /work/ **250GB**（快照无备份）；`$HPCWORK` /hpcwork/ **1000GB**（Lustre）；`$BEEOND` 节点本地 SSD 临时。
- **镜像 + 数据集放 `$HPCWORK`，作业工作目录用 `$WORK`，代码/配置放 `$HOME`。** 查配额 `r_quota`。

### 10.6 网络
- 出站互联网基本可用：PyPI（pip）✅、GitHub（git/GHCR）✅、NGC（nvcr.io）✅——RWTH 文档均有示例。
- **Anaconda 默认源被防火墙封禁**（2024-09 起，licensing）→ 用 **conda-forge / Miniforge**。
- wandb：无官方说明，预计可用；稳妥做法 `WANDB_MODE=offline` 批作业 + login 节点 `wandb sync`。

### 10.7 镜像构建与作业（可直接照抄）

```bash
# login 节点构建镜像
module purge && cd $HPCWORK
apptainer pull pytorch-26.07-py3.sif docker://nvcr.io/nvidia/pytorch:26.07-py3
apptainer exec --nv pytorch-26.07-py3.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```bash
# train.slurm
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
module purge
apptainer exec --nv $HPCWORK/pytorch-26.07-py3.sif python train.py
```

### 10.8 参考资料
- Apptainer：`help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/e6f146d0d9c04d35aeb98da8d261e38b/`
- 分区：`help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/9108f4a6f43c40a3a168919afd36839d/`
- 存储：`help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/da307ec2c60940b29bd42ac483fc3ea7/`
- CLAIX-2023 计算节点：`help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/fbd107191cf14c4b8307f44f545cf68a/`
- NGC PyTorch 26.07：`catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/26.07-py3/tags`
