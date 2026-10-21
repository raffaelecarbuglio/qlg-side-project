# Five-cell continuation on INFN Magellano

The supplied *HPC Clusters — Magellano HPC documentation* lists Levante as
20 dual-socket AMD EPYC 9654 nodes, each with 192 physical cores, 1.5 TiB of
RAM and local NVMe. Its Slurm partition is `levante`. `$SCRATCH_LOCAL` is
deleted after each job; home storage is shared and persistent. The jobs below
keep the vertex cache and JSON reports in a shared `QLG_RUN`, and stage the
backend and its 3j/6j tables to local NVMe per job.

## Inputs and limits

The K=1/2, L=12 scan has completed in both frames. The local K=1, L=12
pilot measured five *new* vertex tensors at 39.6–90.2 seconds each
(median 51.3 s) with two OpenMP threads and a reference BLAS. K=3/2 has
7,919 admissible assignments and 367 required ordered labels per frame.
These are workload measurements, not cluster timing predictions; the Levante
compiler, BLAS, CPU and scheduling can change costs.

Each array has 16 independent single-process jobs with two OpenMP threads
and an 8 GiB memory request per job. The 16 shards cover the **union** of
the two frame label sets; no tensor is assigned to two shards. After they
finish, a separate 16 GiB job contracts all finite spin assignments, compares
two exact contraction orders and performs the K=1/2 independent checks.
Actual peak memory and wall time must be checked on the first Levante pilot
before enlarging the job.

## One-time setup

On Levante, build the pinned `sl2cfoam-next` revision as described in
[setup.md](setup.md), including Wigner tables and `build-info.json`.
Build with the intended CPU target; the original local `-march=native`
binary should not be copied to a different CPU without verification. Use
`rho=gamma*(j+1)`, `MPI=0`, `OMP=1` and VERYHIGH accuracy. Create a
Python environment with the module version that will be loaded by the jobs:

```bash
module load python
python -m venv /absolute/shared/path/to/qlg-venv
source /absolute/shared/path/to/qlg-venv/bin/activate
python -m pip install -e . sympy opt_einsum==3.4.0 cotengra==0.8.0 autoray==0.8.0
```

Copy [infn.env.example](../configs/infn.env.example) to
`configs/infn.env` and set the shared environment and backend paths.
The untracked `infn.env` can load extra modules required by the native
library. Confirm that the native dependencies load on a compute node.

From the repository directory, set shared paths and run the short
K=1/2, L=0 regression before a production array:

```bash
export QLG_ENV_FILE="$PWD/configs/infn.env"
export QLG_RUN="$HOME/qlg-runs/fivecell"
sbatch scripts/infn_fivecell_smoke.slurm
```

The smoke job checks all 32 BF/EPRL coefficients in both frames against
the published reference (relative tolerance 1e-8) and checks the independent
finite sums. If the native build differs from the local binary, start an
unused `QLG_RUN`: strict cache identities reject mixing builds.

## Finite-cutoff job

For K=1, L=12, submit preparation and then assembly:

```bash
export QLG_TWO_K=2 QLG_DL=12
sbatch scripts/infn_fivecell_prepare.slurm
# After all 16 array members have completed successfully:
sbatch scripts/infn_fivecell_assemble.slurm
```

For K=3/2 repeat with `QLG_TWO_K=3` and first `QLG_DL=8`, then 12.
The assembly script refuses missing or mismatched shard reports.
Completed tensors remain in `$QLG_RUN/dl_XXX` even if a member is
interrupted; rerun that member with `sbatch --array=N
scripts/infn_fivecell_prepare.slurm`. Examine
`$QLG_RUN/assembled-kK-dlL.json` and compare the full 32-component
boundary vectors across K, L and the two frames.

The local checkpoint archives can be inspected independently; they are
reusable on Levante only if backend binary, tables, wrapper and NumPy
version produce exactly the same cache identity. No cluster jobs have been
submitted from this repository yet.
