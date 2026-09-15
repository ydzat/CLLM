#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=cllm-diag-vanilla
#SBATCH --output=logs/diag_vanilla_%j.out
#SBATCH --error=logs/diag_vanilla_%j.err

# CPU-only (devel is free): vanilla per-token-position transformer on the SAME
# repeated real-Chinese slice as diag_crossread (50k tokens x 6000 steps), BPE
# ids, identical mask/loss/optimizer. If vanilla learns and block-scan does not,
# the block design is the cause; if both plateau, the shared path is.
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
  python /code/src/diag_vanilla.py \
    --vocab /data/bpe_tokenizer.json \
    --data Skywork/SkyPile-150B \
    --text-field text \
    --steps 6000 \
    --batch 4 \
    --seq 256 \
    --tokens 50000
