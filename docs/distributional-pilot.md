# Boundary-map consistency, gluing and a small refinement pilot

This experiment completed the three locally scoped tests: composition of
boundary maps, finite-complex gluing with explicit normalization, and EPRL
amplitudes on a short refinement chain. It does not establish a continuum
limit or a full family of maps over all triangulations.

The motivation is Bruno, Colafranceschi, Mele and Rovelli,
[The Structure of the Continuum Limit of Spin Foams (2026)](https://arxiv.org/abs/2603.16999),
especially Sections 5.1–5.3. Their framework assumes a consistent family of
boundary embeddings and considers convergence of pairings with test states.
It does not prove distributional convergence of Lorentzian EPRL. The finite
tests here investigate ingredients of such a calculation.

## What is held fixed

Start with the boundary K5 graph of one four-simplex. Its ten links all carry
j=1/2 or all carry j=1; keep all five intertwiner indices open. These give
32 and 243 coefficients, respectively, so each calculation evaluates the
whole linear functional on the selected fixed-spin sector. It is not a
calculation on all boundary spins or graphs.

Bisect primal edges (0,1), then (2,3), then (0,5), where vertex 5 is the
first midpoint. Bisect every cell incident on the selected edge, producing
1, 2, 4 and 6 four-simplices. The corresponding internal triangle counts are
0, 0, 1 and 2; internal tetrahedron counts are 0, 1, 4 and 7. Barycentric
coordinates specify subdivisions, not a physical background metric.

An elementary map is defined on every old boundary link. At each split
tetrahedron choose the child retaining the first endpoint of the selected
edge as its root; choose the analogous child of a split triangle as carrier.
Connect roots through the carrier. Transport existing K5 routes through
these elementary paths, canceling immediate retracings. Unused links have
j=0, and inserted nodes carry normalized bivalent or trivial intertwiners.

These are kinematical pullbacks of the same original test states. They do
not describe a nondegenerate semiclassical geometry at each resolution:
the added boundary nodes have zero volume. Zero-spin extensions can be
legitimate embeddings in the paper's test-state framework; this fact does
not select them dynamically or establish the required global consistency.

## 1. Composition and inner products

Compare elementary holonomy transport through two and three successive
levels against transport along the final composed routes, at random SU(2)
holonomies. Also exchange the order of each of the 15 pairs of disjoint
original edges. Compare final cells, roots and paths using barycentric
coordinates, so new vertex numbering cannot manufacture a difference.

At all four levels and both spins, independently contract magnetic-index
spin networks and compare them with the coarse pullback at six random
holonomy assignments. Check the Gram matrix of all 32 or 243 coarse
intertwiner states through the local basis maps. Together with normalized
bivalent insertions and SU(2) orthogonality, this checks isometry in the
tested sectors. Inherited triangle areas are preserved exactly.

All these checks passed. They cover the stated histories, not arbitrary
overlapping subdivisions or all relations in the refinement order.

## 2. Finite gluing and normalization

Use gamma=1.2, very-high native accuracy, rho=gamma*(j+1), unit internal
edge weight, internal face weight 2j+1, and sqrt(2j+1) per boundary link.
All physical internal spins and intertwiners allowed by admissibility are
retained. The finite internal-spin bounds are proved by neighboring
tetrahedra; this evaluator refuses complexes without such bounds.

Independently assemble two normalized single-simplex amplitudes and join
them across their common tetrahedron. Evaluate the shared intertwiner
metric and orientation phases from explicit SU(2) epsilon matrices.
Each shared triangle needs an interface factor (2j+1)^(-1/2), because the
two original boundary links become one link on the joined boundary.

Compare the entire pulled-back functional with the direct two-cell
calculation. Run both BF controls and EPRL at shell cutoffs 0,2,4,8,
for both spins: ten comparisons in total. Deliberately omit the interface
factor as a negative control. This changes the result by a factor of 4
at j=1/2 and 9 at j=1, and is detected. The correctly normalized results
agree to numerical roundoff.

Full BF coarse/fine functionals also agree on 2,4,6 cells at both spins,
including absolute normalization. BF is a convention check, not a
substitute for EPRL. Joining across this tetrahedron tests finite tensor
gluing; it does not test the continuum cylinder or the conjectured
distributional gluing law in the paper.

## 3. Fresh EPRL pairings

The scan uses 2 spins x 4 triangulations x 4 shell cutoffs = 32 functional
evaluations, containing 4,400 signed coefficients. Two exact contraction
orders check every completed functional. No best-fit normalization is
applied between states or levels. Shell cutoffs improve a fixed-complex
amplitude; they are not spacetime refinement levels.

An important limitation emerged from the admissibility check: for these
particular transported states, every internal triangle on the 4- and
6-cell complexes is forced to carry the original j. This is a consequence
of the boundary labels, not an imposed restriction of the bulk model.
Internal intertwiners are still summed completely, but this pilot does
not explore fluctuating internal face spins. More general boundary probes
or triangulations are needed for that question.

The complete signed arrays, logarithmic scales, spin domains, preparation
records, composition checks and replay validation are in
[the numerical report](../benchmarks/results/2026-10-19-subdivision.json).

Maximum absolute coefficient at shell cutoff 8, with no rescaling:

| Four-simplices | j=1/2 | j=1 |
|---:|---:|---:|
| 1 | 3.352404e-07 | 2.341272e-07 |
| 2 | 2.682381e-14 | 5.933169e-15 |
| 4 | 3.380964e-28 | 1.143401e-29 |
| 6 | 4.262075e-42 | 2.220538e-44 |

Relative change of the full coefficient array from shell cutoff 4 to 8,
measured as max|W8-W4|/max|W4|:

| Four-simplices | j=1/2 | j=1 |
|---:|---:|---:|
| 1 | 2.356% | 4.180% |
| 2 | 3.793% | 6.244% |
| 4 | 7.304% | 13.892% |
| 6 | 11.033% | 21.113% |

Validation maxima:

| Check | Maximum discrepancy |
|---|---:|
| Magnetic-index boundary pullback (absolute) | 3.111e-15 |
| Sector Gram matrix (absolute) | 6.661e-15 |
| Full BF coarse/fine functional (relative max norm) | 5.218e-15 |
| Independent normalized two-cell gluing (relative max norm) | 4.250e-15 |
| Two exact EPRL contraction orders (relative max norm) | 2.179e-16 |

The existing eleven automated checks, including the published Delta4
reference comparison, also passed with the freshly rebuilt backend.
A complete replay using the published source and cached native vertices
reproduced all 32 amplitude arrays and scales without new native EPRL
vertex computations; the replay discrepancies are recorded in the JSON.

## Interpretation and next calculation

Absolute amplitudes are strongly suppressed as cells are added, while
their relative dependence on boundary states can change much less. Small
ratios or a high overlap therefore do not establish convergence to a
nonzero distribution. We neither fit out this suppression nor infer an
infinite-refinement zero limit from four finite triangulations.

Cutoff dependence increases along this chain. The differences between
shell cutoffs 4 and 8 are measured diagnostics, not bounds on the remaining
tail. This must be resolved before making quantitative continuum claims.

The useful next amplitude campaign would control that tail on this small
family and compare additional compatible refinement directions with the
same test states. A consistent extension of the boundary maps must be
established first. Arbitrary additional cells do not automatically supply
a well-defined probe of the paper's full refinement limit. A justified
renormalization prescription may also be needed; this experiment does not
provide one. The pinned alternative rho=gamma*j backend still lacks the
zero-spin case, so that convention has not been compared here.

The older interrupted shell-12 and 16-cell campaigns remain separately
incomplete; these are fresh calculations on different, smaller complexes.
No INFN job was submitted.

## Reproduce

Install the pinned backend as in [setup.md](setup.md), its lookup tables,
and `benchmarks/requirements.txt`. Set SL2CFOAM_LIBRARY and SL2CFOAM_TABLES.
The measured run used one process, two OpenMP threads and one BLAS thread:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m benchmarks.check_subdivision --run runs/subdivision-new \
  --output benchmarks/results/subdivision-new.json --dl 0 2 4 8
```

Completed native tensors are atomic checkpoints. Repeating the same command
reuses them and reconstructs the report. Inputs and backend identity are
checked before cache reuse. The code requires only the existing native
backend and benchmark dependencies; no new simulation package is introduced.
