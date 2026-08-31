# Development — setup and workflow

## Summary

CLLM develops on a low-spec Windows 11 machine (edit + CPU smoke tests only) and trains on the RWTH Aachen CLAIX cluster (Linux + SLURM + Apptainer). Code is synced via Git; training, data preprocessing, and evaluation run on HPC. Containers use Apptainer (Docker is not allowed on CLAIX).

## Local setup (Windows 11)

- Python 3.11, `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU torch, smoke tests only).
- `pip install -r requirements.txt` (core deps: torch, transformers, tokenizers, datasets, wandb — see `requirements.txt`).
- Smoke test = tiny model (`L=1, d=64, batch=2`), 3 steps forward + backward + sampling, catches shape/syntax/logic bugs before shipping to HPC. It is not training.

## HPC (RWTH CLAIX-2023)

- Cluster: CLAIX-2023 (Intel Sapphire Rapids + H100). CLAIX-2025 is NHR/WestAI-only. CLAIX-2018 retired 2024-05-31.
- GPU: 4× H100 96GB HBM2e per ML node; partition `c23g` (`devel` = free, 1h max, for testing).
- Container: **Apptainer** (not Docker; "appcontainer" is not an RWTH term). System-provided, no module load.

### Build the image (on a login node)

```bash
module purge && cd $HPCWORK
apptainer pull pytorch-26.07-py3.sif docker://nvcr.io/nvidia/pytorch:26.07-py3
apptainer exec --nv pytorch-26.07-py3.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Submit a job

```bash
# scripts/train.slurm
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
module purge
apptainer exec --nv $HPCWORK/pytorch-26.07-py3.sif python train.py
```

## Storage

- `$HOME` /home/ 250GB (backup+snapshots): code, configs, important results.
- `$WORK` /work/ 250GB (snapshots, no backup): job working dir.
- `$HPCWORK` /hpcwork/ 1000GB (Lustre): **SIF image + datasets + checkpoints**.
- `$BEEOND`: node-local SSD, temporary. Check quotas with `r_quota`.

## Code sync workflow

```
本地写代码 → pytest 冒烟测试通过 → git commit + push → HPC git pull → SLURM 小规模验证 → 结果 rsync 回传
```

Git is the single source of truth. VS Code Remote-SSH edits directly on the login node for GPU-required debugging. Checkpoints/logs return via `rsync -av hpc:/path .`.

## Network

PyPI, GitHub, and NGC (`nvcr.io`) are reachable from CLAIX. Anaconda default channels are firewall-blocked (use `conda-forge`/Miniforge). wandb: prefer `WANDB_MODE=offline` in batch jobs, `wandb sync` from the login node.

## Reference

RWTH docs: Apptainer `help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/e6f146d0d9c04d35aeb98da8d261e38b/` · partitions `…/article/9108f4a6f43c40a3a168919afd36839d/` · storage `…/article/da307ec2c60940b29bd42ac483fc3ea7/`. NGC PyTorch 26.07: `catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/26.07-py3/tags`.
