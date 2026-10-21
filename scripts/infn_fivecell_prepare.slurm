#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm-fivecell-prep-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
: "${QLG_RUN:?Set a durable shared run directory}"
: "${QLG_TWO_K:?Set 1, 2 or 3}"
: "${QLG_DL:?Set the auxiliary cutoff}"
source scripts/infn_fivecell_env.sh
mkdir -p "$QLG_RUN"
python -m benchmarks.prepare_fivecell \
  --run "$QLG_RUN" \
  --output "$QLG_RUN/prep-k${QLG_TWO_K}-dl${QLG_DL}-shard${SLURM_ARRAY_TASK_ID}.json" \
  --two-k "$QLG_TWO_K" --dl "$QLG_DL" \
  --shards 16 --shard "$SLURM_ARRAY_TASK_ID"
