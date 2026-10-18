# Algorithms for edgewise refinement: measured results

The best tested approach is **exact tensor contraction with a variable number
of explicitly conditioned spin indices**. At the first 1->16 refinement, keep
all five spin sums inside the network for K=1; for K=2, explicitly sum three
chosen spins and contract the remaining network exactly for each assignment.
This is an application of standard cutset conditioning, not a new EPRL model.
No method has been established as optimal for the entire refinement sequence.

## Actual problem evaluated

The ordered edgewise subdivision of one four-simplex has:

| Divisions per original edge | Four-simplices | Internal triangles | Internal tetrahedra | Boundary tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 | 5 |
| 2 | 16 | 5 | 20 | 40 |
| 4 | 256 | 260 | 480 | 320 |

The actual amplitude benchmark is the **16-simplex block**. It uses the pinned
Lorentzian EPRL backend with gamma=1.2, its rho=gamma(j+1) convention, very-high
native accuracy, and auxiliary shell cutoff Delta l. Every boundary triangle
has j=1; every boundary tetrahedron has the stated canonical intertwiner i.
The computed boundary component is

```text
Z(K,Delta l) = sum_{internal j <= K} product_f (2j_f+1)
                * sum_{all admissible internal i} product_v A_v(j,i;Delta l).
```

Edge weights are one; external face weights are omitted as a boundary
normalization convention. All twenty internal intertwiner sums are complete.
For this particular first block, tetrahedra with three boundary faces force
each of its five internal spins to have integer parity. Thus 0,1,...,K is the
full allowed spin domain below integer K, not an integer-spin restriction on
the model. The same shortcut must **not** be assumed at later refinements.

The common intertwiner basis uses the sorted order of the four triangle labels
of each tetrahedron. Native vertex bases are transformed into it by overlaps of
normalized magnetic-index 4jm tensors, including their permutation phases.
Integer face spins make the possible (-1)^(2j) face-orientation factors unity.
Native vertices are cached only by their exact ordered ten-spin tuple. At
finite shell cutoff no unverified permutation symmetry is used.

The code rescales factors and records the common logarithmic scale to avoid
confusing underflow with convergence. Its normalized numerical values are
computational aids; amplitudes at different cutoffs must include that scale.

## Timings on 18 October 2026

AMD EPYC 9V74, one OpenMP and one BLAS thread. Entries are medians of 15
contractions using already constructed factors and already compiled paths.
These are local measurements, not INFN timings.

| Boundary i | K | Delta l | Greedy joint network | Random-path joint network | Enumerate all spin sectors | Best measured choice |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 0 | 1.01 ms | 0.62 ms | 3.86 ms | Joint network |
| 1 | 1 | 0 | 1.03 ms | 0.66 ms | 5.01 ms | Joint network |
| 1 | 1 | 2 | 1.04 ms | 0.67 ms | 4.01 ms | Joint network |
| 1 | 2 | 0 | 81.52 ms | 44.61 ms | 31.47 ms | Condition three spins: **6.34 ms** |
| 2 | 1 | 0 | 1.01 ms | 0.60 ms | 4.15 ms | Joint network |

The K=2 choice fixes spin indices (1,2,4) in the sorted internal-face order.
It evaluates all 3^3=27 assignments, with no sampling and no dropped terms.
It is about 12.9 times faster than the basic joint contraction, 7.0 times faster
than its random-path alternative and 5.0 times faster than sector enumeration.
The largest intermediate shrinks from 30,375,000 bytes for basic greedy to
75,000 bytes per conditioned task. This is **not total memory**: the input
factors occupy 6,135,320 bytes, and peak process RSS was about 180 MiB.

These speedups concern the **sums over the complex**, not individual native
vertices. Preparation took 0.18 s for K=1, Delta l=0; 6.16 s for K=1, Delta l=2;
and 0.56 s for K=2, Delta l=0. Preparation includes native evaluations, copying,
basis transformations and factor assembly. Native computation occupied 0.15,
6.13 and 0.43 s respectively. There were 32 unique native vertex tensors for
K=1 and 243 for K=2. Initialization and convention checks precede these timers.

Path optimization is also separate: about 26-28 ms for the random joint path,
and 23 ms for the winning K=2 conditioned strategy. Exhaustively benchmarking
every possible strategy has additional tuning cost. Cache a selected path by
topology and dimensions for subsequent shell and boundary scans. For a single
cold evaluation, preparation dominates and the end-to-end gain is much smaller
than 12.9 times; at K=2, preparation plus search plus one contraction is roughly
0.64 s for basic greedy versus 0.59 s for the selected conditioned method.

## Sampling comparison

Uniform sampling of the five spins was actually executed: four independent
trials at 128 samples and four at 1024 samples per case. Each draw recomputes
the exact intertwiner contraction, using the precomputed vertex factors but
without caching the sector answer. Monte Carlo retains signs.

Because exact sector answers are available, the population variance is known.
For uniform spin sampling, the predicted sample counts for **1% relative RMS
sampling error** are:

| Boundary i | K | Delta l | Samples |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 310,000 |
| 1 | 1 | 0 | 8,617,879 |
| 1 | 1 | 2 | 5,436,454 |
| 1 | 2 | 0 | 68,470,517 |
| 2 | 1 | 0 | 13,637,684 |

These counts are variance-based predictions, **not completed runs of millions
of samples**. They exclude native, shell-tail and spin-cutoff errors. For the
i=1 cases, 1024 actual samples took roughly 0.13-0.14 s, already much longer
than the complete exact sum while retaining large statistical uncertainty.

The boundary dependence matters: at K=1, Delta l=0, i=0 essentially only one
spin sector contributes; i=1 has cancellations among all 32 sectors. Uniformly
sampling both spins and admissible intertwiners is still worse: exact moment
calculations predict between 10^15 and 10^17 samples for 1% RMS error in these
cases. This estimator was assessed by its exact variance, not timed at those
sample counts. There are 3,713,051,664 admissible joint assignments at K=1 and
11,109,950,280 at K=2. Tensor contraction avoids enumerating them.

This rules against these **uniform estimators on these blocks**. It does not
rule out importance sampling, contour methods or Monte Carlo on larger graphs.

## What happens at 256 simplices?

Only **symbolic dense contraction plans** were computed at this level; no
256-simplex EPRL amplitude was evaluated. Tested optimizers were greedy,
64-trial random greedy, and KaHyPar hypergraph partitioning with subtree
reconfiguration (up to 32 trials, about 30 seconds plus the final trial).

Even for the fixed sector j=1 on every face, the lowest-operation-cost tested
plan predicted about 4.4e20 arithmetic operations and a largest intermediate
of 4.0e17 bytes. Slicing that particular tree down to intermediates below
64 MiB produced about 3.6e19 separate contractions. Treating all j<=1 at once
with dense padded factors was much more expensive.

These are costs of **the found plans**, not lower bounds or proof that every
exact algorithm is infeasible. They ignore exploitable zeros, representation
blocks and possible exact decompositions. They do show that increasing the
number of HPC cores is not enough to justify these dense plans.

My next candidate for the sequence is therefore representation-aware sparse
contractions with reusable local contractions, followed by an explicitly
controlled compression of block boundaries if exact methods stop scaling.
The latter is a renormalization research task, not implemented here. Uniform
Monte Carlo is not the default suggested by the first-block results.

## Checks and remaining limits

- The geometric checks cover cell volumes, boundary support, Euler
  characteristic and internal-spin parity. A separate containment check found
  every division-4 cell inside a division-2 cell, confirming nesting.
- Three direct magnetic-index K5 contractions agree with the native normalized
  BF 15j tensors within 1.7e-16 absolute difference. BF is used for a convention
  check only; all timing-table amplitudes use Lorentzian EPRL vertices.
- All nine exact routes in all five cases agree within 2.7e-15 relative
  difference; there were 675 timed exact evaluations. Random orthogonal changes
  of every internal intertwiner basis preserve the contracted result within
  4.5e-15 relative difference. These are internal checks, not an independent
  published reference amplitude for this new block.
- All eleven project tests passed with the real native Delta4 reference test.
- The shell cutoff is **not converged**: at i=1,K=1, changing Delta l=0 to 2
  changes the amplitude magnitude by a factor of roughly 7400. Roundoff-level
  agreement between algorithms does not control this error.
- Boundary states here are fixed spin-network components, not matched states
  representing the same physical boundary across refinement levels. A scalar
  block with fixed boundary data cannot substitute for the open-boundary tensor
  needed to glue blocks or construct a renormalization map.
- There is no comparison with a replacement implementation of the EPRL vertex
  integral. These experiments compare summation/contraction algorithms for a
  refined complex while retaining sl2cfoam-next as the vertex backend.

Raw results: [i=0](../benchmarks/results/2026-10-18-edgewise-i0.json),
[i=1 and larger-level plans](../benchmarks/results/2026-10-18-edgewise-i1.json),
[i=2](../benchmarks/results/2026-10-18-edgewise-i2.json).
Commands and timing boundaries are in [the benchmark guide](../benchmarks/README.md).

References: [edgewise subdivision](https://doi.org/10.1007/s004540010063),
[hyper-optimized contraction](https://arxiv.org/abs/2002.01935),
[spin-foam tensor algorithms](https://arxiv.org/abs/2406.19676), and
[bulk-spin Monte Carlo](https://arxiv.org/abs/2302.00072).
