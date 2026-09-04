#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-unigram
#SBATCH --output=logs/unigram_%j.out
#SBATCH --error=logs/unigram_%j.err

# Compute SkyPile unigram char entropy (compare against the 6.6 loss plateau).
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
  python /code/src/compute_unigram.py \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --max-chars 10000000
