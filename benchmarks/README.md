# Native vertex comparison

From the repository root, after [installing the backend](../docs/setup.md):

```bash
python -m benchmarks.compare_vertices --output benchmarks/results/local.json
```

This compares two evaluations of the **same Lorentzian EPRL Delta4 amplitude**:

- `full`: compute all vertex entries, then select the requested boundary labels.
- `slice`: request only those boundary labels through the upstream
  `sl2cfoam_vertex_range` API. Both internal intertwiner axes remain complete.

Defaults: gamma=1.2, j=1,2, Delta l=0,2, uniform boundary states i=0,1,2,
three repetitions, one OpenMP thread, one BLAS thread, very-high native accuracy.
Use `--two-i 0` to compare a single boundary state or `--two-j 8` for j=4.
All quantum-number arguments represent twice the physical label.

Each repetition runs each method in a fresh subprocess with a separate temporary
native data directory. A `cold` calculation starts with the same lookup tables
and no booster cache. A `warm` calculation reuses its own booster files but
recomputes all vertices. Method order alternates between repetitions. No cached
vertex values or reference values enter either calculation.

The benchmark checks j=1 against the published table at relative tolerance
1e-5, and checks all method/repetition results against one another at 1e-10.
It records individual timings, vertex and contraction costs, returned tensor
bytes, process peak RSS, library/table/script hashes and CPU information.

Times include vertex evaluation, copying results into NumPy, and Delta4
contractions. They exclude process startup, native initialization, table hashing,
and the checks against reference data. `returned_bytes` is the sum of returned
array sizes, **not** total memory or memory allocations. Peak RSS includes Python,
native workspace and lookup tables; the warm reading retains the process's
earlier peak. These are local measurements, not INFN benchmarks or continuum
convergence tests.

See [the comparison and recommendation](../docs/method-comparison.md).

## Edgewise-refinement comparison

This separate experiment evaluates the full admissible finite-cutoff sum on a
16-simplex block, with boundary j=1 and canonical i=0,1,2. It does not require
or implement a continuum limit. Install its optional dependencies and run:

```bash
python -m pip install -r benchmarks/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m benchmarks.compare_refinement --repeats 15 \
  --output benchmarks/results/edgewise-i1.json
python -m benchmarks.compare_refinement --case 1 0 --boundary-i 0 --repeats 15 \
  --skip-larger-plans --output benchmarks/results/edgewise-i0.json
python -m benchmarks.compare_refinement --case 1 0 --boundary-i 2 --repeats 15 \
  --skip-larger-plans --output benchmarks/results/edgewise-i2.json
```

Unlike `compare_vertices`, `--case K DL` and `--boundary-i` use physical integer
labels, not doubled labels. Cases default to (K,DL)=(1,0),(1,2),(2,0).
The first block's parity constraints exclude half-integer internal spins;
later-level cost plans include half-integer labels where appropriate.

Each case runs in a fresh subprocess with a fresh native booster directory.
One native instance is initialized per process. Preparation includes native
vertices, basis maps and factor assembly; native initialization and convention
checks precede that timer. Path search/compilation and repeated contractions
are reported separately. The sampling timer includes random-index lookup,
factor selection and exact contractions, but excludes generating draws.
All repeated routes share the same precomputed physical factors. No synthetic
tensor amplitudes enter the measured contraction comparison.

The larger 256-simplex level is analyzed using **shapes only**. Reported
operation counts and intermediate sizes there are planner estimates. Slicing
those trees is also symbolic; their many individual tasks are not executed.
KaHyPar timing-limited searches can find different paths on another run; raw
paths are saved. Single-thread execution is required for these comparisons.

See [results, normalization conventions and limitations](../docs/refinement-benchmark.md).

## Exact support and approximate compression

```bash
python -m pip install -r benchmarks/compression-requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_DISABLE_JIT=1
python -m benchmarks.compare_support --output benchmarks/results/support-local.json
python -m unittest benchmarks.test_compression -v
python -m benchmarks.compare_compression --output benchmarks/results/compression-local.json
```

The support comparison preserves the full finite-cutoff amplitude. Compression
calibration uses five full 16-simplex cases plus a fixed-spin sector; the two
256-simplex cases are **fixed-spin sectors only**, with all internal intertwiners
included before truncation. `--small-only` omits the larger cases. Labels use
physical integer spins, as in `compare_refinement`.

`--chi` sets retained dimensions for the small-block calibration; the larger
scan uses chi=2,3,4,6,8 at local gauge radius 1 and chi=4,6,8 at radius 2.
`--max-entries` caps the proposed output of a single contraction (default
32,000,000); it does not cap all tensors or QR/SVD workspace memory. Resource
limits are recorded as failures, not converted to amplitude estimates.

The decomposition backend's first-use cost is reported separately for small
cases. Native preparation, support reduction, optimizer searches and compression
times have separate fields. Compression times include the maintained local
gauge transformations. Peak RSS is cumulative within each fresh worker process.
The optional Numba JIT is disabled to avoid a QR compilation incompatibility in
the tested environment; the native C EPRL backend remains enabled.

See [the measurements and their limits](../docs/compression-experiments.md).

## Cutoffs and fixed contraction orders

`python -m benchmarks.check_cutoffs` scans shell cutoffs and spin cutoffs
K=1,2,3 on the full 16-cell block. K=3 is the exact admissibility endpoint
for its boundary j=1; the auxiliary shell sum remains separate.

`python -m benchmarks.check_refinement` reuses each saved compression order
at six retained ranks, comparing with exact full spin sums on 16,20,25,30
cells. These are conforming local subdivisions for algorithm calibration,
with changed boundary graphs and no boundary-state embedding.

Run the new checks with
`python -m unittest benchmarks.test_refinement benchmarks.test_compression -v`.
Use the single-thread and JIT settings above. See the
[report and complete commands](../docs/cutoff-and-refinement-checks.md).

## Restartable preparation and boundary audit

`python -m benchmarks.block_scan --run runs/block-new --output benchmarks/results/block-new.json`
prepares full ordered vertices with atomic restart, then exactly contracts the
three uniform boundary intertwiner states. Use `--dl 0 2 4 6`, `--cutoff 3`,
`--processes P` and `--threads T`; BLAS must be single-threaded. Different
boundary states and process/thread layouts can reuse the same vertex cache.

`python -m benchmarks.compare_preparation --run-root runs/preparation-new --output benchmarks/results/preparation-new.json`
compares six fresh-cache CPU layouts twice at K=2, shell cutoff 3, checking
all tensor entries and all three amplitudes. `benchmarks.test_block_scan`
checks actual process termination and restart, thread/process agreement and
complete vertex sets. `benchmarks.check_boundary --output PATH` audits the
fixed-j boundary area across edgewise levels. See the
[results and reproduction instructions](../docs/preparation-and-shell-tests.md).

## Matched coarse and fine boundary states

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest benchmarks.test_matched_boundary -v
python -m benchmarks.matched_boundary --run runs/matched-new \
  --output benchmarks/results/matched-new.json --dl 0 2 4 8
```

This computes complete coarse intertwiner functionals at fixed boundary
j=1/2 and j=1 (32 and 243 coefficients), using two exact graph-subdivision
embeddings on a local 2-cell split and the actual 16-cell block. It includes
all admissible bounded internal spins, including half-integers, and every
internal intertwiner. BF amplitudes independently check the full gluing and
normalization before EPRL interpretation. `--two-j` uses doubled spin labels;
`--kinds split` selects the smallest case. Native tensors are restartable.

Unused boundary links carry j=0 and inserted nodes have zero volume. The maps
preserve the original spin-network state as a holonomy pullback, but are not
derived from the dynamics. See the [measurements and limitations](../docs/matched-boundary-tests.md).

To reproduce the ratio figure without a tail fit:

```bash
python -m benchmarks.plot_shell_ratios \
  benchmarks/results/2026-10-18-shell-states.json \
  benchmarks/results/2026-10-18-shell8-resume.json \
  benchmarks/results/2026-10-18-shell10.json --output docs/figures/shell-ratios.svg
```

Only completed K=3 cases are plotted. The default view starts at shell 4;
`--minimum-dl 0` includes the earlier cutoffs.
