#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-diag-order
#SBATCH --output=logs/diag_order_%j.out
#SBATCH --error=logs/diag_order_%j.err

# Diagnostic: can the model reconstruct ordered chars within a block?
set -euo pipefail
module purge
mkdir -p logs

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/diag_order.py
