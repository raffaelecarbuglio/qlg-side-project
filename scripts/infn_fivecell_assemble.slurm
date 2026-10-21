#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-fivecell-assemble-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
: "${QLG_RUN:?Set a durable shared run directory}"
: "${QLG_TWO_K:?Set 1, 2 or 3}"
: "${QLG_DL:?Set the auxiliary cutoff}"
source scripts/infn_fivecell_env.sh
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["QLG_RUN"])
two_k, dl = int(os.environ["QLG_TWO_K"]), int(os.environ["QLG_DL"])
rows = [json.loads((run / f"prep-k{two_k}-dl{dl}-shard{i}.json").read_text())
        for i in range(16)]
if any(r["status"] != "completed" or r["shard"] != i or
       r["two_k"] != two_k or r["dl"] != dl for i, r in enumerate(rows)):
    raise RuntimeError("Some preparation shards are incomplete")
if len({r["cache_id"] for r in rows}) != 1 or len({r["union_sha256"] for r in rows}) != 1:
    raise RuntimeError("Incompatible preparation inputs")
if sum(r["selected"] for r in rows) != rows[0]["union_vertices"]:
    raise RuntimeError("Incomplete union of ordered vertex labels")
print(f"All {rows[0]['union_vertices']} vertices were assigned to 16 completed shards",
      flush=True)
PY
python -m benchmarks.check_fivecell --run "$QLG_RUN" \
  --output "$QLG_RUN/assembled-k${QLG_TWO_K}-dl${QLG_DL}.json" \
  --two-k "$QLG_TWO_K" --dl "$QLG_DL"
