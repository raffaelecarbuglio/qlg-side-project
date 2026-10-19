# Common-boundary 3-to-3 diagnostic

This experiment compares two finite Lorentzian EPRL amplitudes on exactly
the same boundary. It is not a refinement sequence and does not establish
a continuum limit. Exact EPRL equality under this move is not a pass criterion.

## Results: 19 October 2026

All 20 EPRL evaluations completed: five shell cutoffs, two triangulations,
and two local orderings. Each retains all 512 boundary coefficients and
both internal-spin contributions. The
[full numerical report](../benchmarks/results/2026-10-19-pachner.json)
contains 10,240 total and 20,480 sector coefficients.

| Shell cutoff | Left/right difference | Ordering sensitivity | Change from previous cutoff |
|---|---:|---:|---:|
| 0 | 18.54% | 17.10% | -- |
| 2 | 20.07% | 6.13% | 80.04--82.37% |
| 4 | 20.18% | 3.13% | 17.74--20.29% |
| 8 | 20.52% | 1.39% | 5.20--6.88% |
| 12 | 20.66% | 0.67% | 1.25--1.96% |

All three columns use the symmetric maximum-coefficient relative difference
defined below. The cutoff column spans both sides and both local orderings.
The left/right value is the same for either common ordering, up to roundoff.
These are measured changes, not statistical error bars or tail bounds.

At shell 12, the largest absolute coefficients are 2.97546e-21 and
2.99560e-21 for the ascending left and right calculations. Their largest
absolute difference is 6.18991e-22. The distance between their unit-norm
vectors is 0.41827, so their difference is not merely an overall scale.
The shape change between shells 8 and 12 is 0.000795--0.004343, and the
shape sensitivity to local ordering is 0.004250. These dimensionless
distances refer only to this finite orthonormal boundary sector.

At the same cutoff, the x=3/2 contribution has Euclidean norm 13.08% of
the total on the ascending left triangulation (13.49% with descending
ordering). This is a signed amplitude contribution, not a probability.
It is included exactly and is not negligible. Comparing opposite local
orderings, which place the internal spin in matching native face slots,
gives left/right differences of 20.45--20.80%; the effect survives that
additional comparison.

The complete BF identity agrees to 9.44e-16 in the same relative metric.
Omitting the x=3/2 sector on the left produces a 50% BF discrepancy.
An independent cache-only replay reproduced all 30,720 EPRL total and
sector coefficients exactly, with zero new native EPRL evaluations.

**Conclusion:** this small experiment resolves nontrivial finite-cutoff
dependence on bulk connectivity. It improves on the earlier forced-spin
pilot and gives a concrete quantity to study in a future coarse-graining
construction. The 1% ordering target is met, but the 1% absolute-amplitude
cutoff target is not. The data therefore support a numerical indication
of a stable triangulation difference, without a certified infinite-shell
answer. They establish neither continuum convergence nor the superiority
of a refinement strategy. No INFN resources were used.

## Definition

The four-simplices are

```
left:  01234, 01235, 01245    internal triangle: 012
right: 01345, 02345, 12345    internal triangle: 345
```

Both sides have three four-simplices, three internal tetrahedra, and the
same nine boundary tetrahedra and eighteen boundary triangles. A constant
common factor per vertex therefore cancels in their relative comparison.
Spin-dependent weights still matter and are explicitly fixed below.

Every boundary triangle carries j=1/2. At each boundary tetrahedron, both
canonical intertwiner states i=0,1 are retained, so the calculation produces
the complete 512-component functional in this fixed-spin sector. These
include asymmetric boundary states; the comparison is not restricted to
one homogeneous intertwiner assignment. Homogeneous spins are a chosen
boundary sector, not a restriction on the internal EPRL sum. No extra
zero-spin links or boundary embedding maps are introduced.

The internal triangle has exactly two admissible values, x=1/2,3/2. Each
incident tetrahedron has three other spins equal to 1/2; angular momentum
addition gives x<=3/2 and half-integer parity. The corresponding internal
intertwiner spaces have dimensions 2 and 1 at each tetrahedron. Both
sectors and every internal intertwiner are summed. The report retains the
individual signed contributions before summation.

Conventions match the existing backend: gamma=1.2, rho=gamma(j+1), native
VERYHIGH accuracy, unit edge weight, internal face weight 2x+1, and
sqrt(2j+1) per boundary triangle. Native source revision is
`052e4346028870bd76f69a3034e6cae8defb8f7f`.
No overall factor is fitted to either triangulation or any boundary state.

## Validation and diagnostics

- The complete BF functionals must agree between both triangulations and
  both local vertex orderings. This validates the recoupling, orientation,
  gluing, boundary convention, and internal face weight for this example.
- A negative control omits the highest internal spin on one side. It must
  break the BF identity, showing that a missing physical sector is detected.
- Each sector is contracted using two exact tensor contraction orders.
- Native tensors are cached by their full ordered spin tuple and cutoff;
  no permutation symmetry at finite auxiliary cutoff is assumed.
- Ascending and descending local simplex orderings use the same canonical
  boundary basis. Reversal of five vertices is an even permutation. The
  comparison diagnoses finite-shell ordering/gauge-fixing sensitivity.

The auxiliary-spin cutoffs are Delta l=0,2,4,8,12. Delta l=0 is only the
lowest-shell diagnostic; it does not replace the subsequent shell sums.
No physical internal-spin cutoff is imposed.

For two coefficient vectors A and B, the main relative difference is

```
max(abs(A-B)) / max(max(abs(A)), max(abs(B)))
```

The JSON also records absolute differences, Euclidean relative differences,
norm ratios, and the distance between the two unit-norm vectors. Unit-norm
vectors diagnose dependence on boundary state; they do not replace the
absolute amplitudes or define a new normalization of the model.

A 1% last-cutoff change and ordering change were specified as practical
diagnostic targets. Neither is a bound on the omitted infinite shell tail.
Absolute amplitude stability and boundary-shape stability are reported
separately.

## Reproduce

After configuring the native backend and optional benchmark dependencies:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m benchmarks.check_pachner --run runs/pachner-new \
  --output benchmarks/results/pachner-new.json --two-j 1 --dl 0 2 4 8 12
```

Each finished native vertex is atomically saved. Rerunning with the same
run directory validates and reuses those tensors, then reconstructs all
boundary coefficients. The report is atomically updated after each
evaluation. Changing the native library or cache identity rejects reuse.
The raw report contains the boundary ordering, full coefficients and
sector coefficients, so every stated difference can be recalculated.

## Interpretation

The question is whether a change of bulk connectivity produces a resolved
change in boundary-state dependence. A difference would not show that
EPRL is inconsistent: gravitational amplitudes need not be invariant under
Pachner moves at a finite discretization. Agreement on this pair would
also not establish cylindrical consistency or a continuum limit.

This calculation is more informative than the previous forced-spin pilot
about the effect of internal spin fluctuations. It still has small,
homogeneous boundary spins and does not approximate a controlled
semiclassical geometry. It does not implement renormalization, construct
effective amplitudes, or select an optimal refinement sequence or algorithm.

For context, Lorentzian EPRL calculations on the three-vertex, one-internal-
face Delta3 graph appear in [Gozzini (2021)](https://arxiv.org/abs/2107.13952).
The broader distinction between refinement and constructing effective
amplitudes is discussed by
[Asante, Dittrich and Steinhaus (2022)](https://arxiv.org/abs/2211.09578).
