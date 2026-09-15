#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=cllm-diag-crossread
#SBATCH --output=logs/diag_crossread_%j.out
#SBATCH --error=logs/diag_crossread_%j.err

# CPU-only (devel is free, ~1 min queue): compare cross_read OFF vs ON on real
# Chinese. No GPU, no GPU quota. Proves whether the token -> all-block read
# breaks the unigram plateau on real data.
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
  python /code/src/diag_crossread.py \
    --vocab /data/bpe_tokenizer.json \
    --data Skywork/SkyPile-150B \
    --text-field text \
    --steps 1200 \
    --batch 4 \
    --seq 256 \
    --tokens 200000
