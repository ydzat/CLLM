#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=cllm-preflight
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

# Preflight: verify the compute node can reach HF and stream SkyPile with the
# container cache path, BEFORE burning a GPU-queue slot on the real job.
set -euo pipefail

module purge
mkdir -p logs

# Same cache-path convention as train_real.slurm (mounted /data, not host $HPCWORK).
export HF_HOME="/data/hf-cache"
export HF_DATASETS_CACHE="/data/hf-cache/datasets"
export HF_TOKEN="$(cat ~/.hf_token)"

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python -c "
from datasets import load_dataset
ds = load_dataset('Skywork/SkyPile-150B', split='train', streaming=True)
ex = next(iter(ds))
print('OK fields:', list(ex.keys()))
print('text_len:', len(ex['text']))
print('sample:', ex['text'][:60])
"
