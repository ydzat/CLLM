#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-bigram
#SBATCH --output=logs/bigram_%j.out
#SBATCH --error=logs/bigram_%j.err

# Compute bigram entropy — is the 6.6 plateau actually the data's floor?
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
  python /code/src/compute_bigram.py \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --max-chars 10000000
