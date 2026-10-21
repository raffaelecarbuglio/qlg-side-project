#!/usr/bin/env bash
#SBATCH --account=default
#SBATCH --partition=levante
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-fivecell-smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
: "${QLG_RUN:?Set a durable shared run directory}"
source scripts/infn_fivecell_env.sh
python -m benchmarks.check_fivecell --run "$QLG_RUN/smoke" \
  --output "$QLG_RUN/smoke-l0.json" --two-k 1 --dl 0
python - <<'PY'
import json
import os
import numpy as np
from pathlib import Path

reference = json.loads(Path("benchmarks/results/2026-10-19-fivecell.json").read_text())
candidate = json.loads((Path(os.environ["QLG_RUN"]) / "smoke-l0.json").read_text())
assert candidate["status"] == "completed"
for key in ("bf_checks", "results"):
    for row in candidate[key]:
        old = next(x for x in reference[key] if x["ordering"] == row["ordering"]
                   and (key == "bf_checks" or x["dl"] == 0 and x["two_k"] == 1))
        a, b = np.asarray(row["entries"]), np.asarray(old["entries"])
        error = np.max(np.abs(a - b)) / np.max(np.abs(b))
        if error > 1e-8:
            raise AssertionError(f"Native L=0 regression failed: {key} "
                                 f"{row['ordering']}: {error}")
print("Both BF and EPRL L=0 frames match the published coefficients", flush=True)
PY
