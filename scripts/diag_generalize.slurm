#!/bin/bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cllm-diag-gen
#SBATCH --output=logs/diag_generalize_%j.out
#SBATCH --error=logs/diag_generalize_%j.err

# Diagnostic: can the model generalize (learn a copy rule from streaming data)?
set -euo pipefail
module purge
mkdir -p logs

apptainer exec \
  -B "$HOME/cllm:/code" \
  -B "$HPCWORK:/data" \
  "$HPCWORK/cllm.sif" \
  python /code/src/diag_generalize.py
