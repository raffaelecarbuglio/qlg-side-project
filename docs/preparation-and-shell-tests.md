# Restartable preparation, shell cutoffs and boundary checks

Threading and checkpoint reuse make higher shell cutoffs feasible on the
full 16-simplex block. The tested amplitude ratios vary much less than the
raw amplitudes, while the boundary audit shows that fixed-j refinements
change the area of an inherited coarse boundary surface.

This campaign tests the full 16-simplex edgewise block with boundary j=1,
gamma=1.2, unit edge weight, internal face weight 2j+1 and all internal
intertwiners. K=3 completes the internal spin sum for this block. Shell cutoff
Delta l is a separate numerical approximation. Native accuracy is very-high.

`block_scan` prepares every required **ordered full native vertex** once,
then assembles the three homogeneous canonical boundary states i=0,1,2.
It reuses the atomic NumPy checkpoints already used by the production runner.
The cache identity includes gamma, native accuracy, backend/build identity,
lookup-table hashes, the wrapper source hash and NumPy version in a local
cache manifest. The tensor key
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
In each state's build statistics, the inherited `native_seconds` field measures
prepared-backend archive reads. Actual native computation is timed under
`workers`. These timings are local, not INFN results. Requested core-seconds count allocated
CPU capacity times preparation wall time; they are not measured CPU consumption.

The shell scan uses K=3. Each boundary amplitude is contracted with two selected
exact paths. Signed ratios between boundary amplitudes and the normalized
three-component amplitude vector diagnose state dependence of shell effects.
The three states are a tiny part of the boundary Hilbert space. Their normalized
vector is neither a full physical observable nor a probability distribution.
Changes between consecutive cutoffs do not bound the uncomputed shell tail.

## CPU results and recovery checks

The local benchmark used up to four requested CPU threads. The same pinned
native binary and input data were used throughout. Public reports retain
numerical inputs, measurements and source revisions; host diagnostics and
binary/build fingerprints are kept out of those reports. Metadata was narrowed
after the numerical runs without changing any numerical value or timing.
`numeric_source_revision` identifies the public code used for those calculations;
the current drivers export limited metadata by default. The two measured
preparation times and their median:

| Processes | Threads/process | Two cold times (s) | Median (s) | Requested core-seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 70.48, 64.40 | 67.44 | 67.44 |
| 1 | 2 | 43.00, 54.89 | 48.94 | 97.89 |
| 1 | 4 | 30.56, 23.49 | 27.02 | 108.10 |
| 2 | 1 | 74.86, 75.29 | 75.08 | 150.15 |
| 2 | 2 | 60.51, 59.47 | 59.99 | 239.96 |
| 4 | 1 | 95.09, 107.41 | 101.25 | 405.00 |

One process with four threads was fastest, approximately 2.50 times the
serial speed. Multi-process assignments used round-robin shards of the
sorted ordered-spin tuples; other task schedules were not benchmarked. Serial execution consumed the fewest requested core-seconds.
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

## Complete spin sums through shell cutoff 8

All six cutoffs 0,2,4,5,6,8 completed, with K=3 and the same 16-simplex
complex. A0,A1,A2 denote the three homogeneous boundary amplitudes, not
three refinement levels. There are 18 complete finite-shell amplitudes,
each checked with two selected exact contraction paths. The largest
relative disagreement between those paths was 2.11e-15.

| Shell cutoff | ln abs(A1) | A1/A0 | A2/A0 |
| ---: | ---: | ---: | ---: |
| 0 | -350.9128369 | -2.744244297e-06 | 1.382879033e-06 |
| 2 | -341.7932683 | -2.238983096e-06 | 7.479805971e-07 |
| 4 | -340.3325061 | -2.047372783e-06 | 6.338664893e-07 |
| 5 | -340.0720603 | -1.998063987e-06 | 6.175899169e-07 |
| 6 | -339.9081343 | -1.968233529e-06 | 6.135236557e-07 |
| 8 | -339.7145869 | -1.940431981e-06 | 6.183116132e-07 |

The raw i=1 amplitude grows by 21.355% from shell 6 to 8. The signed
ratios A1/A0 and A2/A0 change by -1.413% and +0.780%, respectively.
Removing one common amplitude scale therefore cancels much of the observed
drift, but there is still state dependence. A2/A0 decreases from shell 4
to 5 to 6, then increases at 8; a monotonic tail extrapolation is not justified
by these data. These differences are observations, not shell-tail error bounds
or continuum measurements. No subpercent truncation error for both ratios has
been established, and only three boundary states have been probed.

The previously timed-out K=3, shell-6 case completed locally with four threads
in 387.3 s of preparation. Earlier single-thread timings were obtained on a
different host; this is a completed calculation, not a measured cross-host
speedup factor. Shell-scan timings are individual descriptive measurements;
short regression checks overlapped the beginning of this scan.

Shell 8 reached the imposed 900 s preparation limit after saving 551 vertices.
The incomplete attempt has no amplitude estimate. A second invocation reused
all 551 archives and computed exactly the missing 23, then contracted all three
states. Its preparation took 476.5 s; both attempts together took 1376.6 s.
The restart preserves completed vertices, but rebuilds auxiliary booster data
for the missing ones. This additional work makes the remaining vertex count
a poor predictor of restart runtime. Preserving completed auxiliary data is
a potential future optimization, not part of the current checkpoint mechanism.
The 900 s limit was an experimental budget, not a hardware impossibility.

The raw [initial scan](../benchmarks/results/2026-10-18-shell-states.json)
and [successful shell-8 recovery](../benchmarks/results/2026-10-18-shell8-resume.json)
retain both outcomes, worker counters, timings, signs, logarithmic scales,
normalization diagnostics, exact contraction paths and public source revisions.

The completed physical spin sum also matters for the choice of boundary
probe. At shell cutoff 0, the i=2 amplitude is negative at K=1 in the
[earlier data](../benchmarks/results/2026-10-18-fixed-order.json), with
ln(abs(A2))=-352.9256373. At K=3 it is positive, with
ln(abs(A2))=-351.5981751. All new shell comparisons use the exact K=3
spin endpoint, rather than assuming one small-K result controls every state.

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
ansatz at fixed physical area. A limit involving growing physical boundaries
would need its own scaling prescription and observables. States involving j=0, superpositions, a larger-area semiclassical
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
python -m benchmarks.block_scan --run runs/shell-states-new --cutoff 3 \
  --dl 0 2 4 5 6 8 --processes 1 --threads 4 --timeout 900 \
  --output benchmarks/results/shell-states-new.json
```

If shell 8 reaches the budget, keep its first report and resume explicitly:

```bash
python -m benchmarks.block_scan --run runs/shell-states-new --cutoff 3 \
  --dl 8 --processes 1 --threads 4 --timeout 900 \
  --output benchmarks/results/shell8-resumed-new.json
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
