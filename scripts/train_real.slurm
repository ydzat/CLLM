#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=cllm-train-real
#SBATCH --output=logs/train_real_%j.out
#SBATCH --error=logs/train_real_%j.err

# Real training: stream SkyPile (no raw download), char-level, 60M model.
# Requires: $HPCWORK/cllm.sif (image) + $HPCWORK/vocab.json (built once).
set -euo pipefail

module purge
mkdir -p logs

# HF cache must live on the /data bind-mount ($HPCWORK → /data inside the
# container), NOT the host path $HPCWORK — that is read-only inside Apptainer.
export HF_HOME="/data/hf-cache"
export HF_DATASETS_CACHE="/data/hf-cache/datasets"
# Read-only HF token (gated datasets); stored outside git in ~/.hf_token.
export HF_TOKEN="$(cat ~/.hf_token)"

apptainer exec --nv \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/train.py \
    --config /code/configs/train.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --text-field text \
    --save-dir /data/checkpoints \
    --save-interval 1000
