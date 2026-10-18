# Validation

`make test` runs a small set of checks covering the risks in this implementation:

- Spin admissibility, half-integer parity and nonzero intertwiner offsets.
- Signed Delta4 contractions compared with an independently enumerated sum.
- A compiled C ABI fixture with unequal tensor dimensions, including layout
  and memory lifetime. This fixture is **not** a physics calculation.
- Atomic checkpoint reuse, incompatible-run rejection and worker locking.
- Incomplete scans and relative comparisons of amplitudes near 1e-38, with no
  absolute tolerance that would make an incorrect zero pass.
- Report generation and same-boundary 1->5 subdivision combinatorics.

If `SL2CFOAM_LIBRARY` and `SL2CFOAM_TABLES` are set, an additional test computes
the real Delta4 amplitude at Delta l=0 for j=1, gamma=1.2 and i=0,1,2. It compares
with the independent published table using relative tolerance 1e-5. Otherwise
this test is explicitly skipped. CI exercises the Python and ABI checks; it does
not install the native scientific libraries.

`make smoke` extends that native comparison to Delta l=0,1,2.
`configs/benchmark.json` extends the scan through Delta l=15. Reference failure
is an error, not an instruction to rescale results until they match.

On 18 October 2026 the pinned native backend was built locally and all ten tests
passed, including the real Delta4 reference test. Additional full-table versus
boundary-slice runs at j=1, Delta l=0,2 and i=0,1,2 agreed with the published
finite-cutoff table within 1.61e-9 relative difference. The j=2 comparison checks
agreement between two evaluation routes; it has no independent reference table.
See [the measured comparison](method-comparison.md) for raw results, build hashes
and limitations. Native calculations have not yet been run on INFN; validate the
target installation before its production campaign.

Successive-cutoff changes are not certified tail estimates. Reference agreement
also does not establish refinement convergence, correctness of an unimplemented
general gluing map, or a continuum limit.

The experimental edgewise benchmark additionally validates its local basis
conventions against direct magnetic-index BF contractions and checks that
internal basis rotations leave its real EPRL block invariant. Nine exact
evaluation routes agree across five finite-cutoff cases. These are internal
checks; there is no independent published reference value for this block.
The edgewise topology checks bring the project test count to eleven, all
passing locally with the native backend. See [the refinement report](refinement-benchmark.md).

Three optional compression checks additionally passed with the real backend:
exact reduction of a hyper-index sum, completeness of the binary spin-parity
generators at 256 simplices, and agreement of a fixed-spin sector and high-rank
compression with the exact full small-block calculation. Run them with
`python -m unittest benchmarks.test_compression -v` after installing the optional
benchmark dependencies. Large-sector compression is explicitly not converged;
see [the experiment report](compression-experiments.md).

Three more optional refinement checks passed: independent geometric
containment/volume partition for the 20,25,30-cell subdivisions; rejection of
an integer-only builder on a complex with half-integer sectors; and explicit
20-cell spin-sector enumeration versus joint contraction, followed by
fixed-order compression agreement and input-immutability checks. The native
test uses a fresh process to avoid repeated native-library initialization.
The full six-check optional suite passed. The new
[cutoff and refinement report](cutoff-and-refinement-checks.md) distinguishes
complete spin sums, finite shell cutoffs and unconverged larger calculations.

The restartable 16-simplex driver is checked by
`python -m unittest benchmarks.test_block_scan -v`: full required vertex sets,
real native thread/process agreement, reuse after changing execution layout,
and SIGTERM interruption followed by recovery of completed atomic tensors.
The additional [preparation and shell study](preparation-and-shell-tests.md)
records fixed-input CPU comparisons and a boundary-area audit.

This campaign also reran the 11 default checks (including the published native
Delta4 reference) and six existing optional refinement/compression checks.
All passed, in addition to the three new block-scan checks.

The matched-boundary campaign adds two optional checks, run with
`python -m unittest benchmarks.test_matched_boundary -v`. They reject an
unproved finite spin bound on 1->5 and validate the exact holonomy pullbacks,
inherited areas, full BF functionals and half-integer spin domains on both
the local 2-cell split and the actual 16-cell block. The BF comparisons retain
all 32 or 243 coarse intertwiner coefficients, including normalization and
signs. All eight fine BF functionals agree with their coarse counterparts.
The 11 default checks were also rerun and passed. See
[the matched-boundary report](matched-boundary-tests.md) for EPRL measurements
and the limitations of the j=0 kinematical vacuum.

A separate two-build convention control completes three native vertex cases
and records one unsupported case: rho=gamma*j with zero spins is refused by
the pinned upstream dsmall implementation. This is a backend limitation,
not a successful amplitude or a physical divergence.
