# qlg-side-project

Numerical Lorentzian EPRL spin-foam calculations, starting from the published
Delta4 benchmark and working toward controlled refinement studies.

**Implemented:** native vertex amplitudes, the full admissible internal-label
sum for the Delta4 complex, auxiliary-cutoff scans, checkpoints, reference
comparisons, plots and Slurm arrays. The Python interface calls
[sl2cfoam-next](https://github.com/qg-cpt-marseille/sl2cfoam-next)'s C library;
Julia is not required.

**Current scope:** finite-complex calculations. There is no implemented continuum
limit or general coarse-graining algorithm. The `refinement-plan` command checks
same-boundary subdivision combinatorics; it does not compute their amplitudes.

## First run

Python >= 3.11 and a configured native backend are required. Follow
[docs/setup.md](docs/setup.md) for the native installation and environment variables.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make test
make estimate
```

With `SL2CFOAM_LIBRARY` and `SL2CFOAM_TABLES` set:

```bash
make smoke
```

This computes Delta4 at gamma=1.2, boundary spins j=1 and shell cutoffs
0, 1, 2, then compares the three uniform boundary intertwiner states i=0,1,2
with the published numerical table. Agreement is a test to perform, not a
result assumed by the code. No reference values enter the amplitude computation.

Rerunning resumes completed work. A changed config, source code, backend or
lookup table requires a new run directory.

## INFN / Slurm

Prepare on a login node; execute numerical work on compute nodes. Paths to the
virtual environment, backend, tables and run directory must be visible there.

```bash
qlg prepare configs/benchmark.json runs/benchmark
qlg submit runs/benchmark --jobs 2 --threads 4 --mem 4G --time 01:00:00 --dry-run
qlg submit runs/benchmark --jobs 2 --threads 4 --mem 4G --time 01:00:00
# After all array tasks finish:
qlg analyze runs/benchmark --reference
```

Add `--partition NAME` and `--account NAME` if your allocation requires them.
Resources above are initial requests, not measured requirements. Benchmark the
smoke configuration first. One array task handles one shell cutoff; OpenMP uses
the requested cores, while BLAS uses one thread. MPI builds are not supported
by this worker.

For a workstation:

```bash
qlg prepare configs/smoke.json runs/local
qlg run runs/local --jobs 1 --threads 2
qlg analyze runs/local --reference
```

Reports appear in `RUN/analysis/`: CSV, JSON, PNG and PDF. Incomplete scans fail
analysis unless `--allow-partial` is explicit. Failed reference checks return a
nonzero exit status. Relative changes are cutoff diagnostics, not error bounds.

## Inputs and conventions

- All spins and intertwiners in config files are **twice** their physical values.
- `configs/vertex.json` supports arbitrary admissible spins at one vertex.
- Delta4 uses homogeneous boundary spins. Its twelve boundary intertwiners may
  differ: replace `two_i` by a list of twelve labels, grouped three per vertex.
- Internal spins and intertwiners of Delta4 are summed over their entire allowed
  ranges. Fixing boundary data does not replace EPRL with a symmetry-reduced
  internal model.
- `shells` controls the auxiliary-spin cutoff Delta l, not lattice refinement.
- `max_tensor_mb` limits output-tensor size only; native workspace needs more RAM.

See [the amplitude specification](docs/model.md),
[validation status](docs/validation.md) and
[the next refinement calculation](docs/refinement.md).
The [method comparison](docs/method-comparison.md) includes measured native
timings and a [reproducible benchmark](benchmarks/README.md).
An experimental [edgewise-refinement comparison](docs/refinement-benchmark.md)
evaluates the first 16-simplex block at small internal-spin cutoffs, comparing
exact contractions, partial spin conditioning and uniform sampling.
Further [support and compression experiments](docs/compression-experiments.md)
improve the exact small-block sum and test approximate fixed-spin sectors at
256 simplices; those larger results have not converged.
New [cutoff and fixed-order checks](docs/cutoff-and-refinement-checks.md)
complete the first block's spin sum at K=3 and calibrate compression against
exact full finite-cutoff sums on 20, 25 and 30 simplices.
[Restartable preparation and shell tests](docs/preparation-and-shell-tests.md)
complete K=3 at auxiliary shell cutoffs through 8, compare CPU layouts and
verify restart on the expensive case. Boundary amplitude ratios stabilize
faster than raw amplitudes; a naive fixed-j refinement changes boundary area.
[Matched-boundary tests](docs/matched-boundary-tests.md) now implement exact
SU(2) graph-subdivision embeddings, validate half-integer gluing against BF,
and compare every coarse intertwiner coefficient in fixed j=1/2 and j=1
sectors. The added boundary nodes are degenerate; this is a diagnostic of a
kinematical embedding, not a dynamical vacuum or a continuum calculation.
The same report extends the original complete K=3 shell scan through cutoff
10: the two ratios change by 0.55% and 1.15% from cutoff 8, with the second
remaining nonmonotonic over the scan. The extended matched-boundary campaign
is recorded as partial: 31 of 32 comparisons were confirmed before the execution
service disconnected; shell-12 completion is unverified.

## References

- Donà & Frisoni, [How-to compute EPRL spin foam amplitudes](https://arxiv.org/abs/2202.04360).
- Gozzini, [A high-performance code for EPRL spin foam amplitudes](https://arxiv.org/abs/2107.13952).
- Asante, Dittrich & Steinhaus, [Spin foams, refinement limit and renormalization](https://arxiv.org/abs/2211.09578).

Numerical reference data are attributed in `qlg/reference/source.json`.
Upstream native libraries retain their own licenses; they are not vendored here.
