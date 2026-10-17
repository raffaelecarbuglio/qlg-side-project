# Next physical calculation: a common-boundary refinement

The intended next comparison starts from a single four-simplex and inserts one
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

This specification is an explicit research boundary of version 0.1: general
EPRL gluing and the renormalization map are not replaced with guessed formulas.
Reference: Asante, Dittrich & Steinhaus,
[Spin foams, refinement limit and renormalization](https://arxiv.org/abs/2211.09578).
