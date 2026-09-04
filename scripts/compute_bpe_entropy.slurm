#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-bpe-entropy
#SBATCH --output=logs/bpe_entropy_%j.out
#SBATCH --error=logs/bpe_entropy_%j.err

# Compute BPE unigram + conditional entropy.
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
  python /code/src/data/compute_bpe_entropy.py \
    --data Skywork/SkyPile-150B \
    --vocab /data/bpe_tokenizer.json \
    --max-chars 10000000
