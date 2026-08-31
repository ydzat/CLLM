#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --job-name=cllm-train-real
#SBATCH --output=logs/train_real_%j.out
#SBATCH --error=logs/train_real_%j.err

# Real training: stream SkyPile (no raw download), char-level, 60M model.
# Requires: $HPCWORK/cllm.sif (image) + $HPCWORK/vocab.json (built once).
set -euo pipefail

module purge
mkdir -p logs

# Keep HuggingFace cache off HOME (quota) — stream cache lives on HPCWORK.
export HF_HOME="$HPCWORK/hf-cache"
export HF_DATASETS_CACHE="$HPCWORK/hf-cache/datasets"

apptainer exec --nv \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/train.py \
    --config /code/configs/train.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --text-field text
