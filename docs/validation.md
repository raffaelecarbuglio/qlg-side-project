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
