#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=cllm-verify-cpu
#SBATCH --output=logs/verify_cpu_%j.out
#SBATCH --error=logs/verify_cpu_%j.err

# 1-step CPU verification (devel is free + fast): full config (d=512, L=12) +
# real SkyPile data, one forward + backward + optimizer step. Proves the full
# training path (data -> model -> loss -> backward) on CPU, where the queue is
# ~1 min instead of ~13 h on the GPU partition. GPU execution is already proven
# by the earlier smoke test (device: cuda + 3 steps); what remains unverified is
# full config + real data together, which this runs.
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
    --config /code/configs/train.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/bpe_tokenizer.json \
    --text-field text \
    --steps 1 \
    --save-dir /data/checkpoints \
    --save-interval 1
