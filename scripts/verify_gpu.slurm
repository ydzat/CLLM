#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=cllm-verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

# 1-step GPU verification: full config + real SkyPile data, one
# forward + backward + optimizer step. Proves the GPU training path end-to-end
# before committing a ~20h queue slot to the full 30000-step job.
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
    --config /code/configs/train.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/vocab.json \
    --text-field text \
    --steps 1
