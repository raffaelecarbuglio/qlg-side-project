# Cutoff checks and fixed-order compression

These tests complete the internal-spin sum on the 16-simplex block, separate
rank changes from contraction-order changes, and evaluate full finite-cutoff
spin sums on conforming 20, 25 and 30-simplex complexes. The larger complexes
provide exact references for approximation tests. They are not measurements
of a continuum limit.

## An exact endpoint for the first spin sum

All boundary triangles carry j=1. Each of the five internal triangles belongs
to a tetrahedron whose other three triangles are boundary triangles. A
four-valent SU(2) invariant requires each spin to be no larger than the sum
of the other three, so every internal spin is at most 3. The binary parity
constraints also force these five spins to be integers.

Consequently K=3 includes **every admissible internal spin** on this particular
complex and boundary state. It does not complete the auxiliary shell sum, nor
does it provide a spin bound for other complexes or boundary data.

The scans use gamma=1.2, canonical boundary i=1, unit edge weights, and one
factor 2j+1 per internal face. Boundary face weights are omitted, as in the
previous block calculations. Results are unnormalized boundary amplitudes.
Values are recorded as a sign and a natural logarithm to avoid underflow.

| Shell cutoff | ln(abs A), K=1 | ln(abs A), K=2 | ln(abs A), K=3 |
| ---: | ---: | ---: | ---: |
| 0 | -350.957241 | -350.935839 | -350.912837 |
| 1 | -344.510015 | -344.329959 | -344.322110 |
| 2 | -342.043829 | -341.798294 | -341.793268 |
| 3 | -341.065930 | — | -340.799692 |
| 4 | -340.600725 | -340.335145 | -340.332506 |
| 6 | -340.173886 | — | 600 s limit |

All completed amplitudes in this table are negative. A dash denotes an
uncomputed pair. The K=3, shell-cutoff-6 worker reached the configured
600-second limit; it supplies no amplitude estimate. Fifteen cutoff cases
completed and one reached this limit.

At shell cutoff 4, completing the spin sum from K=2 to K=3 changes
the magnitude by **0.264%**. With the spin sum complete, raising the
shell cutoff from 3 to 4 changes it by **59.55%**. There is no
shell-convergence plateau in these data. At K=1, raising the shell
cutoff from 4 to 6 still changes the magnitude by 53.2%.

For the complete K=3 spin sum at shell cutoff 4, native vertex
calculations took 186.2 s; the two exact contractions took
0.058 and 0.067 s. Native shell-sum preparation dominates this block
at this accuracy setting. This cost observation does not determine
the best algorithm on the full 256-cell graph.

Cutoff differences are diagnostics, not bounds on the missing shell tail.
In particular, changes of an unnormalized amplitude can include changes of
overall normalization. These tests do not determine convergence of normalized
physical observables. Shell cutoff, physical spin sum and lattice refinement
remain different operations.

## Compression with the order held fixed

For each case we construct one resolved tensor network, including its exact
equality trees for shared spin labels. Two contraction orders are searched at
reference rank 8. Each saved order is then reused unchanged at ranks
2,4,8,16,32,64. The input tensors, equality-tree arrangement and local QR gauge
radius 1 are also held fixed. Thus rank is the only changed approximation
setting within each scan. No numerical zero tolerance is used for support
pruning.

The table gives the larger absolute relative error of the two fixed orders:

| Simplices | K | Shell cutoff | Rank 8 | Rank 16 | Rank 32 | Rank 64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 0 | 0.00600 | 5.8e-15 | 7.7e-15 | 4.4e-16 |
| 16 | 1 | 2 | 0.611 | 5.6e-15 | 2.4e-14 | 3.8e-15 |
| 16 | 2 | 0 | 0.652 | 0.0271 | 5.0e-14 | 9.6e-15 |
| 20 | 2 | 0 | 0.250 | 0.0295 | 2.3e-15 | 3.3e-15 |
| 30 | 2 | 0 | 0.154 | 0.0198 | 0.00130 | 2.5e-14 |

At K=1 and shell cutoff 0, all tested 20,25,30-cell amplitudes recover the
exact result at rank 16. At K=2 the tested 30-cell block needs rank 64 to
recover roundoff agreement on both saved orders. Rank 32 already gives
better than 0.13% observed relative error there, but this is measured against
an exact answer; it is not a general error guarantee.

Errors can still be nonmonotone with fixed order. For example, on the 30-cell
K=2 block, one order gives absolute relative errors 1.010 and 1.076 at ranks
2 and 4. Retaining more local information does not order the errors after
signed contributions cancel. Different orders also remain relevant: on the
16-cell K=1, shell-cutoff-2 case, rank-8 errors are 4.44% and 61.1%.

There are 120 completed approximate contractions across ten cases. They
include boundary i=2 on the original block and shell cutoff 2 on a 20-cell
block. Every exact reference is checked using two independently selected
contraction paths. No compression attempt in this campaign reached the
16,000,000-entry intermediate cap. Timings are individual measurements, not
a repeated performance comparison. First-use overhead is included where it
occurs, so these data should not be used to rank algorithms by small timing
differences.

## What the intermediate complexes test

Starting from the 16-cell edgewise block, we successively bisect boundary
edges (1,3), (4,7), (8,12), inserting a midpoint and splitting every incident
four-simplex. The vertex numbering is that of `edgewise_simplex(2)`. This
produces nested conforming triangulations of the same auxiliary four-ball:

| Simplices | Internal triangles | Internal tetrahedra | Boundary tetrahedra |
| ---: | ---: | ---: | ---: |
| 16 | 5 | 20 | 40 |
| 20 | 8 | 27 | 46 |
| 25 | 12 | 36 | 53 |
| 30 | 16 | 45 | 60 |

All have no internal edges, and an exact parity-rank check forces all internal
spins to be integers. The builder rejects a complex with allowed half-integer
sectors rather than silently dropping them. The contractions include every
internal spin up to the stated K and every admissible internal intertwiner;
they are not fixed-spin sectors or a restricted geometric model.

Geometry checks verify tetrahedron incidence, positive cell volumes, total
volume, boundary placement and Euler characteristic. An independent test
checks containment and the volume partition inside every original parent
simplex. Another enumerates all 256 spin sectors of the 20-cell K=1 block
and compares their sum with the joint tensor contraction.

These locally refined graphs are algorithm-calibration examples, not a
replacement for the uniform edgewise sequence. Their boundary graphs differ;
no embedding of one physical boundary state into another has been specified.
They also avoid the nontrivial half-integer sectors that occur at 256 cells.
Success here therefore does not resolve the unconverged 256-cell sector
calculation or implement its full spin sum.

## Reproduce

With the native environment configured and optional dependencies installed:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_DISABLE_JIT=1
python -m benchmarks.check_cutoffs --output benchmarks/results/cutoffs-local.json
python -m benchmarks.check_cutoffs --cases 1:6 2:4 3:3 3:4 3:6 --timeout 600 \
  --output benchmarks/results/cutoffs-extended-local.json
python -m benchmarks.check_refinement --output benchmarks/results/fixed-order-local.json
python -m benchmarks.check_refinement --cases 1:2:0:1 3:2:0:1 1:1:2:1 \
  --output benchmarks/results/fixed-order-extended-local.json
python -m unittest benchmarks.test_refinement benchmarks.test_compression -v
```

Cutoff cases use `K:shell`; refinement cases use `steps:K:shell:boundary_i`.
Labels here are physical integers. Each numerical case runs in a fresh
native process and cache. The new native test is also isolated because the
pinned library cannot safely be reinitialized repeatedly in one process.
All six optional checks passed, including three existing compression checks.

The raw [cutoff grid](../benchmarks/results/2026-10-18-cutoffs.json),
[extended grid](../benchmarks/results/2026-10-18-cutoffs-extended.json),
[fixed-order scans](../benchmarks/results/2026-10-18-fixed-order.json), and
[extended scans](../benchmarks/results/2026-10-18-fixed-order-extended.json)
include actual paths, input shapes, signs, logarithmic scales, timings,
resource records and source hashes. Dependencies are pinned in the benchmark
requirements. Runs used the same native EPRL backend as the earlier studies,
very-high native accuracy, and single-thread execution. Optional Numba JIT
was disabled; native C calculations were enabled. These are local results,
not INFN measurements.
