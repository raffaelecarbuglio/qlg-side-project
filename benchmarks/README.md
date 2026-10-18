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
