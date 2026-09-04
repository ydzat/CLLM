#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-diag-context
#SBATCH --output=logs/diag_context_%j.out
#SBATCH --error=logs/diag_context_%j.err

# Diagnostic: can the slot attention copy a char from block context?
set -euo pipefail
module purge
mkdir -p logs

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/diag_context.py
