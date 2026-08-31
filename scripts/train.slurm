#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --job-name=cllm-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Block-scan reader training on one H100 (RWTH CLAIX-2023, Apptainer).
# Code lives at $HOME/cllm (git clone); the image lives at $HPCWORK (docs/development.md).
set -euo pipefail

module purge
mkdir -p logs

apptainer exec --nv \
  -B "$HOME/cllm:/code" \
  "$HPCWORK/cllm.sif" \
  python /code/src/train.py --config /code/configs/dev.yaml
