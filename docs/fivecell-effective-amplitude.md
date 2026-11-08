# Regulated five-cell effective boundary amplitude

This implements the first calculation in the
[fixed-boundary protocol](continuum-protocol.md): a one-to-five subdivision
of one four-simplex, with the complete j=1/2 boundary functional retained.
It computes finite regulated EPRL amplitudes. It does not remove the internal
spin regulator or obtain a continuum limit.

## Completed results — 19 October 2026

All 16 five-cell evaluations completed: two physical spin cutoffs, four
auxiliary cutoffs, and two local frames. Each keeps all 32 boundary
coefficients. Four single-cell references are also recorded.

| Physical cutoff K | max(abs(W5)) at L=8, across frames | Relative change L=4 to 8 | Frame difference at L=8 |
|---|---:|---:|---:|
| 1/2 | 2.477e-34 to 2.519e-34 | 7.11% to 8.83% | 1.67% |
| 1 | 9.570e-34 to 9.841e-34 | 8.80% to 8.90% | 2.76% |

Increasing K from 1/2 to 1 at L=8 changes the coefficient vector by
74.12--74.43% in the symmetric maximum-coefficient metric below. Its
Euclidean norm grows by a factor 3.878--3.923. The distance between its
unit-norm directions is 0.06017--0.06021: the change is predominantly scale,
with a smaller change in boundary-state dependence. Neither that factor
nor the direction is used to redefine the amplitude.

The lower-spin shell in every K=1 result reproduces the complete K=1/2
functional exactly, confirming the nesting of the two physical cutoffs.
The added max(j_internal)=1 shell is therefore important. Two K values
do not establish a large-K divergence law or the convergence of any ratio.

Independent BF finite sums agree within 2.50e-15. EPRL explicit-intertwiner
checks at K=1/2 agree within 7.85e-17, and the two exact contraction paths
agree within 5.05e-17 across all EPRL evaluations. These relative errors
refer to the full finite coefficient vectors. They do not bound the native
vertex integration error or the missing spin tails.

The single-cell maximum at L=8 is 3.35240e-7. In the ascending frame the
five-cell/single-cell Euclidean norm ratio is 7.48250e-28 at K=1/2 and
2.93563e-27 at K=1. Their unit-direction distances are respectively
0.06180 and 0.001690. This very strong absolute suppression is retained.
It depends on the stated convention and incomplete internal-spin sums;
it does not establish a zero continuum limit. A coarse-reference absolute
tolerance cannot resolve the regulator dependence of these tiny amplitudes;
the report therefore also retains the relative changes at their own scale.

**Assessment:** the implementation and finite sums pass their checks.
The 1% relative auxiliary-cutoff and frame targets are not met, and
physical-spin convergence is not established. No nine-cell run or
normalization fit has been performed. The data deliver a regulated
effective amplitude and quantify the controls still required.

Native vertex preparation took 589.39 seconds in total on two OpenMP
threads, of which 526.03 seconds were at L=8. Each individual boundary
contraction/assembly took under one second in the recorded final pass.
These are local measurements, not an INFN benchmark. No INFN jobs were used.

All 612 ordered native tensors are supplied in the
[numerical archive](../benchmarks/results/2026-10-19-fivecell-vertices.tar.gz).
They contain arrays, identity digests and timings. A fresh extraction and
cache-only replay reproduced all 640 fine/coarse coefficients and 768
spin-shell coefficients exactly, with zero new native EPRL evaluations.
The archive checksum and code-file hashes are included in the raw report.

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

To inspect or restore the supplied numerical archive:

```bash
mkdir -p runs/fivecell-archived
tar -xzf benchmarks/results/2026-10-19-fivecell-vertices.tar.gz -C runs/fivecell-archived
```

Using that directory as `--run` validates the strict cache identity; a
different native build may require its own run directory. The stored arrays
can also be read independently with NumPy's `load(..., allow_pickle=False)`.

## October 21 extension: L=12 at K=1/2

The [complete L=12 report](../benchmarks/results/2026-10-21-fivecell-l12-report.json)
contains both orderings and all 32 boundary coefficients at K=1/2. The
largest absolute coefficient is 2.544003471e-34 (ascending) and
2.534996094e-34 (descending). Relative to L=8, the complete vectors change
by 1.08% and 2.29%, respectively, in the symmetric maximum norm defined
above. The two L=12 orderings differ by 0.354% in that norm. These are finite
auxiliary-cutoff comparisons at one fixed physical cutoff.

The backend source revision remains 052e4346. The rebuilt local binary uses
the Wigner source vendored in ARTS and the system reference BLAS. Its L=0
BF and EPRL vectors agree entry for entry with the previously published
native calculation; the [restart audit](../benchmarks/results/2026-10-21-fivecell-l0-regression.json)
records those values. This regression does not independently validate every
L=12 native integration.

The [L=12 checkpoint archive](../benchmarks/results/2026-10-21-fivecell-l12-checkpoint.tar.gz)
contains 14 complete, validated native tensors, their strict cache identity,
the build manifest and the numerical report. For a backend rebuilt with
exactly the same binary, lookup tables, wrapper and NumPy version:

```bash
python scripts/restore_fivecell.py \
  --archive benchmarks/results/2026-10-21-fivecell-l12-checkpoint.tar.gz \
  --run runs/fivecell-recovered
```

An incompatible build is rejected; the archive remains readable as data.
During a run, take further recoverable snapshots using:

```bash
python scripts/snapshot_fivecell.py --run runs/fivecell-recovered \
  --report runs/fivecell-recovered/report.json \
  --prefix benchmarks/results/fivecell-l12 --dl 12
```

The runner also accepts `--two-k 3` (K=3/2): its five-cell graph has
7,919 admissible spin assignments and 367 ordered native vertex labels per
frame. To measure representative native vertices without committing to the
full sum, use `python -m benchmarks.pilot_fivecell --run RUN --output
PILOT.json --two-k 3 --dl 12`. Pilot tensors share the full-run cache.
The [INFN continuation notes](infn-fivecell.md) record the completed
K=1 and K=3/2 pilots and the sharded Levante jobs.

## November 8 extension: L=12 at K=1 on Levante

The [completed run](../benchmarks/results/2026-11-08-fivecell-k1-l12-levante.json) records all 32 signed
boundary coefficients for both orderings, with 721 admissible internal spin
assignments per ordering. The maximum absolute coefficient is
`9.956657910e-34` (ascending) and `9.835044117e-34` (descending).

Using the symmetric maximum norm defined above, the complete coefficient
vectors change by **1.1647%** (ascending) and **2.6945%** (descending)
relative to the archived K=1, L=8 vectors. The difference between the two
orderings decreases from **2.7551%** at L=8 to **1.2214%** at L=12.

The reports specify the same native source revision and amplitude conventions,
but the local L=8 and Levante L=12 calculations used different native builds.
The observed changes therefore do not isolate the effect of L or bound the
remaining auxiliary-spin tail. This is a finite-cutoff comparison at fixed K
and fixed triangulation.
