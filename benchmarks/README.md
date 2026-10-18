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
