# Regulated five-cell effective boundary amplitude

This implements the first calculation in the
[fixed-boundary protocol](continuum-protocol.md): a one-to-five subdivision
of one four-simplex, with the complete j=1/2 boundary functional retained.
It computes finite regulated EPRL amplitudes. It does not remove the internal
spin regulator or obtain a continuum limit.

## Amplitude and regulators

An interior vertex 5 replaces 01234 by the five four-simplices
01235,01245,01345,02345,12345. The exterior five tetrahedra and ten triangles
are unchanged. All ten boundary spins equal 1/2, and both intertwiners at
each boundary tetrahedron remain open: 32 signed coefficients per evaluation.
The boundary tetrahedra and their canonical sorted-face bases are identical
for the single-cell and five-cell calculations.

All ten internal triangle spins run over half-integer values from zero
through K. Every internal tetrahedral intertwiner is summed over its entire
admissible range. K=1/2 gives 16 admissible spin assignments and K=1 gives
721. These are exhaustive within the stated regulator. The unregulated
spin domains are unbounded; neither cutoff is an admissibility endpoint.

The microscopic vertex is the full Lorentzian EPRL vertex approximated at
auxiliary shell cutoff L=Delta l. The numerical scan uses L=0,2,4,8,
gamma=1.2, rho=gamma(j+1), VERYHIGH native accuracy, unit edge measure,
internal face weight 2j+1 and sqrt(2j+1) per external triangle. The last
product is 32 for this boundary. No fitted scalar or per-state rescaling is
applied. Native source revision is
`052e4346028870bd76f69a3034e6cae8defb8f7f`.

Both ascending and descending local cell vertex orderings are evaluated,
and transformed to the same canonical boundary basis. Reversal of five
vertices is an even permutation. In the descending frame the exterior
tetrahedron is the gauge-fixed native tetrahedron; in the ascending frame
it is not. Their discrepancy is a finite-shell sensitivity diagnostic,
not a physical change of boundary state.

## Independent validation

For K=1/2 the finite BF sum is evaluated in two ways:

1. Native normalized BF vertices, canonical basis rotations, and optimized
   tensor contraction with every internal intertwiner summed.
2. BF vertices reconstructed directly from magnetic-index 4jm tensors,
   without the native 15j routine or vertex-basis rotations, followed by
   explicit loops over every internal intertwiner assignment. Face
   orientation phases are reconstructed by actual epsilon-matrix products.

The complete finite sums agree in both frames. Removing the internal face
weights causes a large discrepancy and is retained as a negative control.
No raw equality with the single BF vertex is required: the new interior
vertex introduces bubble/gauge issues, and a finite spin cutoff does not
preserve an unregulated topological identity.

The K=1/2 EPRL sums at every L are independently checked using explicit
internal-intertwiner loops and epsilon-matrix phases. Both physical cutoffs
also use greedy and optimal exact contraction paths, cached by their shapes.
These checks validate the finite sums and conventions. They do not provide
an independent evaluation of the native Lorentzian vertex integrations.

## Recorded data and interpretation

The [raw report](../benchmarks/results/2026-10-19-fivecell.json) retains all
32 coefficients at every K,L and frame, the single-cell reference at each L,
and contributions grouped by the maximum internal spin. In particular,
the K=1 contribution from max(j_internal)<=1/2 must reproduce the complete
K=1/2 result with the same frame and L. The report records contraction,
cutoff and ordering diagnostics, cancellation measures, and preparation costs.

Relative amplitude changes use

```
max(abs(A-B)) / max(max(abs(A)), max(abs(B)))
```

Unit-vector distances are separate shape diagnostics. Spin contributions
are signed amplitudes, not probabilities. Finite changes in K or L are
diagnostics rather than rigorous estimates of the missing infinite tails.
Only two physical K values are tested; they cannot establish convergence,
an asymptotic divergence exponent, or a renormalization prescription.

## Reproduce and restart

After configuring the native backend and benchmark dependencies:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m benchmarks.check_fivecell --run runs/fivecell-new \
  --output benchmarks/results/fivecell-new.json --two-k 1 2 --dl 0 2 4 8
```

`--two-k` uses doubled labels. The two frames are selected by
`--orders ascending descending`. Higher L values through 12 are accepted,
but they are not automatically run. Native vertices are cached by full
ordered spin tuples, auxiliary cutoff and strict backend identity. A vertex
is reusable only after its complete atomic archive has been written.
`RUN/progress.json` is updated after every prepared tensor; each complete
boundary functional is atomically recorded in the output report.
