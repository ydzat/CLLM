#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# Text generation from a trained checkpoint (iterative mask-predict).
set -euo pipefail
module purge
mkdir -p logs

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/generate.py \
    --config /code/configs/train.yaml \
    --checkpoint /data/checkpoints/final.pt \
    --vocab /data/vocab.json \
    --num-samples 3 \
    --rounds 16
