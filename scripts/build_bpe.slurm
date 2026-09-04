#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=cllm-build-bpe
#SBATCH --output=logs/build_bpe_%j.out
#SBATCH --error=logs/build_bpe_%j.err

# Train BPE tokenizer on streamed SkyPile (one-time).
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
  python /code/src/data/build_bpe.py \
    --data Skywork/SkyPile-150B \
    --out /data/bpe_tokenizer.json \
    --vocab-size 30000 \
    --max-chars 50000000
