#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=cllm-train-300m-resume
#SBATCH --output=logs/train_300m_resume_%j.out
#SBATCH --error=logs/train_300m_resume_%j.err

# Resume the 300M model from step_70000 (stuck at unigram) with the WSD
# schedule (warmup -> stable 1e-3 -> decay) to ESCAPE the unigram local minimum.
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
    --config /code/configs/train_300m.yaml \
    --data Skywork/SkyPile-150B \
    --vocab /data/bpe_tokenizer.json \
    --text-field text \
    --save-dir /data/checkpoints \
    --save-interval 5000 \
    --mask-ratio 0.15 \
    --resume /data/checkpoints/step_70000.pt
