#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=cllm-fillmask
#SBATCH --output=logs/fillmask_%j.out
#SBATCH --error=logs/fillmask_%j.err

# Fill-mask diagnostic: mask 15% of a real sentence, show model predictions.
set -euo pipefail
module purge
mkdir -p logs

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/fill_mask.py \
    --config /code/configs/train.yaml \
    --checkpoint /data/checkpoints/final.pt \
    --vocab /data/vocab.json \
    --text "深度学习是机器学习的一个分支，它通过多层神经网络自动学习数据的特征表示。近年来随着计算能力的提升和大规模数据的积累，深度学习在图像识别语音识别和自然语言处理等领域取得了显著进展。这些技术进步推动了人工智能的发展，为各行各业带来了新的机遇和挑战。" \
    --mask-ratio 0.15 \
    --topk 3
