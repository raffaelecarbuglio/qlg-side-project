# Exact support reduction and compression experiments

The experiments produced an improved **exact** 16-simplex evaluator, and actual
compressed calculations on a **fixed-spin sector** of 256 simplices. The latter
are not converged. There is no trustworthy full 256-simplex amplitude or new
renormalization map in this work.

## Improved exact algorithm

At boundary j=i=1, gamma=1.2, internal-spin cutoff K=2 and shell cutoff Delta l=0:

| Method | Median contraction time | Input factor storage |
| --- | ---: | ---: |
| Padded factors, greedy | 91.86 ms | 6.14 MB |
| Padded factors, random-greedy path | 50.09 ms | 6.14 MB |
| Exact support trimming, greedy | 7.20 ms | 1.13 MB |
| Support trimming plus conditioning one spin | **2.92 ms** | **1.13 MB** |

These are fifteen-repeat local measurements. The winning method conditions
spin index 3 in the sorted internal-face order and evaluates all three values
0,1,2, contracting every remaining spin and intertwiner exactly. It is about
31 times faster than the padded greedy calculation in this run and about twice
as fast as the previous 6.34 ms conditioned benchmark. The latter comparison
comes from separate runs, so it is indicative rather than a paired speed ratio.

The trimming algorithm removes an index value only if at least one incident
factor is identically zero for that value. It iterates to closure and uses
**no numerical tolerance**. It removes 37 impossible padded label slots across
the indices; it removes no admissible physics configurations. Preparation took
0.586 s, support trimming 0.013 s; path search is separate in the raw data.
The largest intermediate for the winning method is 69,984 bytes, not total RAM.
As before, cold end-to-end gains are smaller than contraction-only speedups.

Exact local contractions also work: the full K=1 block reduces completely with
generated intermediates no larger than 3,888 entries. At K=2 the same 10,000-entry
output cap leaves four factors, after 17 contractions. This cap does not restrict
the sizes of factors already present at input.

## Exact spin-parity structure at the next level

At every tetrahedron, the sum of its four spins must be an integer. Assign one
binary variable to each internal triangle: zero for integer spin and one for
half-integer spin. With integer boundary spins these constraints give:

| Simplices | Internal triangle labels | Independent parity equations | Free binary labels |
| ---: | ---: | ---: | ---: |
| 16 | 5 | 5 | 0 |
| 256 | 260 | 225 | **35** |

Binary Gaussian elimination computes these ranks exactly. The 35 internal edges
also provide 35 independent generators of allowed parity assignments; all are
checked against every parity equation. This is an exact representation of the
parity constraints, not yet a full contraction algorithm. Further triangle
inequalities and intertwiner constraints still apply.

This identifies why simply extending the first block's integer-only spin loop
would be wrong: half-integer sectors become possible at 256 simplices. A full
later-level implementation must retain them and validate their gluing phases.

## Compression calibrated against exact amplitudes

We used quimb 1.11.2 and cotengra 0.8.0 for SVD bond compression on arbitrary
networks. Hyper indices are resolved into trees of exact equality tensors.
The number chi is the largest retained bond dimension. Larger chi retains more
information, but does not itself bound the error in a signed amplitude.

The five full 16-simplex cases use the same boundary states and cutoffs as the
[previous comparison](refinement-benchmark.md). Every internal spin and
intertwiner is included before compression. Two path searches were tested at
each chi. The table gives the **larger observed relative error** of the two:

| Boundary i | K | Delta l | chi=2 | chi=4 | chi=8 | chi=16 | chi=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 7.5e-6 | 2.6e-12 | 1.4e-15 | 2.3e-15 | 3.2e-15 |
| 1 | 1 | 0 | 1.39 | 0.665 | 0.0673 | 6.0e-15 | 9.7e-15 |
| 2 | 1 | 0 | 2.45 | 2.66 | 0.0482 | 4.6e-14 | 8.2e-14 |
| 1 | 1 | 2 | 0.417 | 0.351 | 0.0462 | 1.0e-14 | 2.7e-15 |
| 1 | 2 | 0 | 0.863 | 1.42 | 0.844 | 0.0900 | 9.5e-15 |

Thus chi=32 reproduces every tested small-block amplitude to roundoff, while
smaller dimensions can change the sign or produce errors greater than 100%.
The error is not monotone in chi because paths and intermediate truncations
also change. The very small error for i=0 at small chi is not representative of
the other boundary states. Compression is slower than exact contraction on
these small blocks; its purpose here is calibration for larger graphs.

An additional fixed-j=1, 16-simplex sector check compares the separately built
sector to the corresponding term of the full spin sum, including its logarithmic
normalization. This avoids treating a sector as the full amplitude.

## Actual 256-simplex sector calculations

Here **every triangle is fixed at j=1**, while all 480 internal intertwiner
indices are summed, subject to compression. Boundary j=i=1 and Delta l=0.
This is a diagnostic of the contraction method, not the full spin-summed model.
Only exactly zero support was eligible for pruning; none was removed. Exact
local contractions reduced 256 factors to 176, with no generated factor larger
than 243 entries. The exact remaining contraction is still expensive.

The following are actual approximate contraction results. L is the natural
logarithm of the absolute value **after the recorded common input rescaling**.
Only comparisons within this table use that common scale; these are not
continuum observables.

| chi | Path search | Sign | L, local gauge radius 1 |
| ---: | ---: | ---: | ---: |
| 2 | 37 | - | -37.61 |
| 2 | 71 | - | -36.95 |
| 3 | 37 | + | -24.62 |
| 3 | 71 | + | -24.33 |
| 4 | 37 | - | -18.28 |
| 4 | 71 | + | -23.29 |
| 6 | 37 | - | -15.31 |
| 6 | 71 | - | -16.21 |

Larger retained spaces change the magnitude by many orders, and different
paths can change the sign. Enlarging the local gauge radius to 2 did not cure
this: at chi=6 the two searches gave signs + and -, with L=-13.15 and -14.67.
There is no evidence of convergence in this scan, and no exact reference at
256 simplices. These values must **not** be reported as predictions.

All four chi=8 attempts reached the configured cap of 32,000,000 entries in a
proposed intermediate and were stopped before that contraction. This is a cap
on one output array, not the process's total memory. Peak process RSS was about
1.6 GiB; QR/SVD workspaces and other tensors also occupy memory. Completed
large-sector contractions took roughly 2-23 s each, plus path search. These
times are not estimates for a complete spin sum or for converged accuracy.

## Reproducibility and interpretation

The [driver](../benchmarks/compare_compression.py) recorded 76 compression
attempts: 72 completed and four stopped at the entry cap. Raw paths, timings,
normalizations, resource-limit reasons and source hashes are stored in
[compression results](../benchmarks/results/2026-10-18-compression.json).
The [support comparison](../benchmarks/results/2026-10-18-support.json) records
150 timed exact evaluations. Three additional checks passed: preservation of
a hyper-index sum under exact reduction, parity-generator completeness, and
native sector/compression agreement with the full small block.

The runs used one OpenMP, BLAS and Numba thread. Numba JIT was disabled for
these optional benchmarks after an observed QR compilation incompatibility;
NumPy/SciPy linear algebra was used, and the native EPRL backend was unchanged.
The selected compressor uses explicit QR gauge transformations, avoiding a
rank-deficient virtual-gauge instability observed in an exploratory run.
Path searches can return other paths on another run; recorded paths are part
of the experimental record. Timings are local, not INFN measurements.

The exact trimming plus conditioning is the current measured default for the
small block. A small compressed dimension that works there is not justified at
the next level. The full 256-simplex spin sum, controlled open-boundary block
compression, boundary-state matching and an RG prescription remain unimplemented.
The auxiliary and physical spin cutoffs also remain unconverged.

The compression method follows
[Gray & Chan, PRX 14, 011009](https://arxiv.org/abs/2206.07044); software details
are in the [quimb guide](https://quimb.readthedocs.io/en/latest/examples/ex_htn_to_tn_2d.html).
These are established numerical techniques applied here to the EPRL test,
not a claim of a new quantum-gravity model or proof of a continuum limit.
