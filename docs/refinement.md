# Refinement calculations

An experimental edgewise 1->16 calculation now exists in `benchmarks/`.
It implements explicit common-basis transformations, all admissible internal
intertwiners and small finite internal-spin cutoffs for boundary j=1.
See [its specification, measurements and limits](refinement-benchmark.md).
This is not a renormalization map or a matched-boundary continuum calculation.
The full spin-summed 256-simplex level has symbolic cost estimates only.
Additional [compression experiments](compression-experiments.md) evaluate
fixed-spin sectors there, but do not obtain converged amplitudes.
The [follow-up cutoff checks](cutoff-and-refinement-checks.md) reach the exact
K=3 endpoint of the first block's spin sum and test full finite-cutoff sums on
20,25,30-cell local subdivisions. Their changed boundary graphs serve as
algorithm tests; they do not define a boundary-state matching map.

The new [matched-boundary calculation](matched-boundary-tests.md) supplies
explicit kinematical embeddings for a local 1->2 split and the 1->16 block,
including half-integer gluing checks and full coarse intertwiner functionals.
Unused boundary links carry j=0; the inserted nodes have zero volume. This
is a controlled first comparison, not a dynamically chosen fine vacuum.
The experimental evaluator proves finite spin bounds for these complexes and
refuses the unbounded 1->5 case below.

## Separate common-boundary 1->5 calculation

A complementary comparison starts from a single four-simplex and inserts one
interior vertex. This produces five four-simplices with the same five external
tetrahedra. There are ten internal triangles and ten internal tetrahedra.

The code can already construct and check these triangulations:

```bash
qlg refinement-plan --steps 2
```

It produces 1, 5 and 9 simplices by successively subdividing one cell. This is
one refinement sequence, not all triangulations. The output explicitly marks
its amplitudes as unimplemented.

Before computing it, the physical amplitude needs the following specification:

1. An ordered frame and intertwiner basis for each shared tetrahedron. Local
   vertex bases need not agree. Derive the required permutation, recoupling and
   orientation factors; matching array dimensions is not sufficient.
2. A common external boundary basis, with its normalization, and the face/edge
   measure consistent with that basis. Fix any necessary gauge treatment.
3. A regulated sum over the ten internal spins, and a separate auxiliary-shell
   convergence study. Internal-spin sums can be unbounded; the finite Delta4
   range must not be reused for this topology.
4. A renormalization prescription. The two bare amplitudes are not required to
   be equal in a gravitational model. Specify how effective amplitudes change
   when internal degrees of freedom are eliminated. A free normalization fitted
   independently for every boundary state would erase the test.

A first extension should evaluate the smallest admissible nontrivial boundary
data and compare more than one boundary state. Benchmark native vertex costs,
reuse exactly matching tensors and optimize contraction order before committing
large resources. The ten internal spins make brute-force enumeration grow very
quickly. Uniform sampling or another estimator would require an independent
error study against exact sums at small cutoffs.

Only after several refinement levels can be computed should one ask whether
boundary amplitudes stabilize as both numerical controls and the retained
boundary information improve. A plateau at fixed truncation is insufficient.

General-complex amplitudes and the renormalization map remain research tasks.
The experimental edgewise benchmark derives its basis transformations explicitly;
it does not turn this combinatorial 1->5 command into an amplitude evaluator.
Reference: Asante, Dittrich & Steinhaus,
[Spin foams, refinement limit and renormalization](https://arxiv.org/abs/2211.09578).
