#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --job-name=cllm-train-300m
#SBATCH --output=logs/train_300m_%j.out
#SBATCH --error=logs/train_300m_%j.err

# Scale-up training: ~300M model, ~3.3B tokens, streamed SkyPile.
set -euo pipefail
module purge
mkdir -p logs

export HF_HOME="/data/hf-cache"
export HF_DATASETS_CACHE="/data/hf-cache/datasets"
export HF_TOKEN="$(cat ~/.hf_token)"

apptainer exec --nv \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/train.py \
    --config /code/configs/train_300m.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/bpe_tokenizer.json \
    --text-field text \
    --save-dir /data/checkpoints \
    --save-interval 5000 \
    --mask-ratio 0.15
