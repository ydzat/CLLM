#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=cllm-verify-med
#SBATCH --output=logs/verify_med_%j.out
#SBATCH --error=logs/verify_med_%j.err

# Medium config (d=128/L=2) on real SkyPile — does a bigger model break unigram?
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
    --config /code/configs/verify_med.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --text-field text \
    --steps 400 \
    --mask-ratio 0.15
