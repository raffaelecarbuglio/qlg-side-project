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

The native EPRL benchmark has **not** been executed in the initial development
environment, which lacks its dependencies. Consequently the initial release
does not claim a reproduced EPRL value. Execute the native test on the target
installation before a production campaign. This status should be updated with
the backend build manifest and run results after that execution.

Successive-cutoff changes are not certified tail estimates. Reference agreement
also does not establish refinement convergence, correctness of an unimplemented
general gluing map, or a continuum limit.
