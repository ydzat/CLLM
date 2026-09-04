#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=cllm-verify-tokenmask
#SBATCH --output=logs/verify_tokenmask_%j.out
#SBATCH --error=logs/verify_tokenmask_%j.err

# Verify token-level masking drops loss below unigram entropy (~6.6).
# Small model (dev config) + real SkyPile data + 500 steps, on CPU.
set -euo pipefail
module purge
mkdir -p logs

export HF_HOME="/data/hf-cache"
export HF_DATASETS_CACHE="/data/hf-cache/datasets"
export HF_TOKEN="$(cat ~/.hf_token)"

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/train.py \
    --config /code/configs/dev.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --text-field text \
    --steps 300 \
    --mask-ratio 0.15
