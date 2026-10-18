# Restartable preparation, shell cutoffs and boundary checks

This campaign tests the full 16-simplex edgewise block with boundary j=1,
gamma=1.2, unit edge weight, internal face weight 2j+1 and all internal
intertwiners. K=3 completes the internal spin sum for this block. Shell cutoff
Delta l is a separate numerical approximation. Native accuracy is very-high.

`block_scan` prepares every required **ordered full native vertex** once,
then assembles the three homogeneous canonical boundary states i=0,1,2.
It reuses the atomic NumPy checkpoints already used by the production runner.
The cache identity includes gamma, native accuracy, backend/build identity,
lookup-table hashes, the wrapper source hash and NumPy version. The tensor key
also includes the shell cutoff and ordered spin tuple. Changing process/thread
counts or the tested boundary intertwiner does not invalidate these tensors.
Changing numerical conventions or backend does. Native booster workspaces are
isolated per process and discarded between invocations; only complete atomic
vertex archives are resumed. The last interrupted vertex may need recomputation.

## What is being compared

The CPU benchmark uses K=2, Delta l=3, with six process/thread layouts and two
fresh-cache repetitions in opposite order. It checks every native vertex
against the first serial run, and compares all three fully contracted boundary
amplitudes. BLAS uses one thread throughout. The timed preparation includes
worker startup, native initialization, loading lookup tables, vertex computation
and atomic checkpoint writes. Assembly and exact contractions are separate.
These timings are local, not INFN results. Requested core-seconds count allocated
CPU capacity times preparation wall time; they are not measured CPU consumption.

The shell scan uses K=3. Each boundary amplitude is contracted with two selected
exact paths. Signed ratios between boundary amplitudes and the normalized
three-component amplitude vector diagnose state dependence of shell effects.
The three states are a tiny part of the boundary Hilbert space. Their normalized
vector is neither a full physical observable nor a probability distribution.
Changes between consecutive cutoffs do not bound the uncomputed shell tail.

## CPU results and recovery checks

The local host reported Intel Xeon Platinum 8573C, nine visible virtual CPUs,
and an eight-core CPU quota. The same pinned native binary and input data
were used throughout. The two measured preparation times and their median:

| Processes | Threads/process | Two cold times (s) | Median (s) | Requested core-seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 70.48, 64.40 | 67.44 | 67.44 |
| 1 | 2 | 43.00, 54.89 | 48.94 | 97.89 |
| 1 | 4 | 30.56, 23.49 | 27.02 | 108.10 |
| 2 | 1 | 74.86, 75.29 | 75.08 | 150.15 |
| 2 | 2 | 60.51, 59.47 | 59.99 | 239.96 |
| 4 | 1 | 95.09, 107.41 | 101.25 | 405.00 |

One process with four threads was fastest, approximately 2.50 times the
serial speed. Serial execution consumed the fewest requested core-seconds.
More processes lost sharing of native booster preparation; their total native
computation increased. This conclusion applies to this small full finite-K
block, binary and machine. Two repetitions expose timing variation but do not
establish a universal ranking, particularly on Levante or larger complexes.

Across all twelve cold runs, the largest amplitude relative difference was
5.69e-14. The largest native tensor difference, divided by that tensor's largest
entry, was 1.04e-14. Entries close to cancellation zero are checked with an
absolute tolerance of 1e-14 times the tensor maximum, alongside an entrywise
relative tolerance of 1e-10 and a whole-tensor maximum-relative-error check.
A preliminary comparison with a fixed absolute tolerance of 1e-25 flagged
an approximately 1.19e-25 difference in a tensor with maximum around 2.5e-9.
The scale-aware check was then applied to every saved cold output without
repeating its timed native preparation. The final controller records this
validation and distinguishes loaded cold reports from newly executed cases.

A fresh two-process/two-thread invocation reused all 243 baseline vertices,
computed none, and took 0.620 s for the preparation stage. This comparison
includes lookup-table loading and archive reads but excludes factor assembly
and contraction on both cold and reused paths. All three final amplitudes
still agreed with the baseline.

The three additional native/unit checks passed: complete required vertex sets
(32,243,574 at K=1,2,3), thread/process agreement and cross-layout reuse, and
SIGTERM termination followed by recovery. The termination check waits for
at least three complete archives, kills the controller, resumes, verifies
that exactly the missing entries are computed and that every preserved
archive's SHA-256 is unchanged.

Raw [CPU measurements](../benchmarks/results/2026-10-18-preparation.json)
and the [boundary audit](../benchmarks/results/2026-10-18-boundary-audit.json)
include numerical values and provenance.

## Boundary audit

The same original boundary triangle has 1,4,16,64 subtriangles at edgewise
q=1,2,4,8. The script independently checks their auxiliary geometric areas
and that they partition each of the ten original triangles. If all fine spins
stay j=1, the additive area of such a coarse surface grows by precisely those
factors, in fixed units of 8*pi*gamma*l_P^2. Thus the current homogeneous
probes do not keep that geometric boundary area fixed across refinement.

There is another useful small-spin diagnostic. A coarse j=1 triangle has area
sqrt(2), while four nonzero fine-spin area eigenvalues sum to at least
4*sqrt(3)/2 > sqrt(2). Consequently one cannot represent its subdivision by
four equal, nonzero, definite fine spins at the same physical scale. This is
not an obstruction to a continuum limit: it diagnoses this particular boundary
ansatz. States involving j=0, superpositions, a larger-area semiclassical
boundary, and an appropriate map between boundary Hilbert spaces require their
own construction. Matching areas alone would still leave shapes and
correlations unspecified. The area spectrum is from Rovelli,
[arXiv:1102.3660v5](https://arxiv.org/abs/1102.3660), equations 37-38.

The audit does not supply a physical metric to the auxiliary coordinates or
implement a boundary embedding/renormalization map. It provides a concrete
failure of the naive fixed-j comparison, not a numerical continuum result.

## Reproduce

Install the native backend and optional `benchmarks/requirements.txt`, then:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest benchmarks.test_block_scan -v
python -m benchmarks.compare_preparation --run-root runs/preparation-new \
  --output benchmarks/results/preparation-new.json
python -m benchmarks.check_boundary \
  --output benchmarks/results/boundary-audit-new.json
```

Use an unused directory for a cold CPU benchmark. `compare_preparation --resume`
can revalidate complete cold reports and execute missing cases; use the same
hardware, software and inputs when resuming a timing campaign. A partially
prepared cold timing case requires a fresh directory. To resume a shell scan,
reuse its `--run` directory; complete vertices are retained even after timeout
or interruption. Reports should have distinct names if preserving each attempt.
`--processes P --threads T` requests P*T CPU threads for preparation. Assembly
and contraction run in a separate single-thread process. Native workspace
control is 128 MB per thread, which is not a total process memory limit.

For a Levante pilot, submit from the repository with the same Python/native
environment configured:

```bash
sbatch scripts/benchmark_block.slurm runs/levante-preparation-new \
  benchmarks/results/levante-preparation-new.json
```

The script requests one node, four CPUs and 8 GB, then repeats the same six
layouts. It has been syntax-checked locally; no INFN job has been submitted.
The supplied hardware PDF lists 192 physical cores, 1.5 TiB RAM and local
NVMe per Levante node. This pilot uses only a small allocation, and should
precede changing the layout for a larger allocation or multiple sockets/nodes.
