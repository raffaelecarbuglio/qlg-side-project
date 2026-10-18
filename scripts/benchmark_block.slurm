#!/usr/bin/env bash
#SBATCH --partition=levante
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Submit from the repository with its Python environment and native variables set.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.compare_preparation \
  --run-root "${1:?Supply an unused run directory}" \
  --output "${2:?Supply a report JSON path}"
