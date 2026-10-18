# Numerical method comparison — 18 October 2026

The evidence supports keeping sl2cfoam-next as the vertex backend and comparing
different ways of requesting and combining its results. It does not establish
an optimal algorithm for refinement or the continuum limit.

## Methods checked

| Method | Evidence and availability | Consequence for this project |
|---|---|---|
| sl2cfoam-next, complete vertex tensors | Runnable Lorentzian EPRL implementation; published comparisons with the original sl2cfoam. | Reference calculation for small complexes and reused boundary data. |
| sl2cfoam-next, selected boundary slices | Existing upstream range API; locally benchmarked against complete tensors below. | Smaller returned arrays do not necessarily mean a faster amplitude. |
| Low-rank matrix contractions, Asante–Steinhaus (2024) | Their paper constructs a Lorentzian EPRL algorithm, but explicitly leaves its numerical implementation to future work. The released implementation and measured speedups concern SU(2) BF coherent amplitudes. | Promising development route; no demonstrated EPRL speedup can be assigned from those BF benchmarks. |
| Uniform Monte Carlo, Donà–Frisoni (2023) | Published EPRL calculations, including the five-vertex renormalization / 5–1 diagram, with released Julia code using sl2cfoam-next. | The most directly relevant existing approach for summing many internal spins in our first refinement calculation. |
| Integration with deformed contours and Monte Carlo, Han et al. (2021) | Published Lorentzian EPRL propagator calculations on one simplex, targeting relatively large spins. | Different observables and numerical formulation; a separate implementation and matched comparison would be needed. |

These methods address different parts of the calculation. Monte Carlo over
bulk spins can retain the same vertex library, as can a new contraction order.
Changing that order exactly, or estimating a sum without bias, need not impose
a geometric restriction on the underlying EPRL model. Sampling and truncation
errors must still be measured separately.

## Local comparison

The executable [benchmark](../benchmarks/README.md) compares complete tensors
with vertex slices at identical boundary data, native accuracy and shell cutoff.
The slices fix only external intertwiner labels; both internal intertwiner axes
and the entire admissible bulk-spin sum are retained. This tests the same finite
sum by two evaluation routes, not a restricted internal model.

Machine: Intel Xeon Platinum 8573C, Ubuntu 24.04, GCC 13.3, OpenBLAS 0.3.26,
WIGXJPF 1.13, FASTWIGXJ 1.4.1. One OpenMP thread and one BLAS thread; gamma=1.2,
very-high native accuracy. Each entry is the median of three repetitions.
The published j=1 values enter only the checks after the timed calculation.

With previously computed booster data, but with every vertex recomputed:

| Boundary states | j | Delta l | Full table, ms | Selected slices, ms |
|---|---:|---:|---:|---:|
| i=0,1,2 | 1 | 0 | 0.795 | 1.100 |
| i=0,1,2 | 1 | 2 | 9.056 | 18.057 |
| i=0,1,2 | 2 | 0 | 1.126 | 2.390 |
| i=0,1,2 | 2 | 2 | 120.471 | 89.742 |
| i=0 | 1 | 0 | 0.606 | 0.562 |
| i=0 | 1 | 2 | 8.225 | 5.237 |

There is no uniform winner. For three states at j=1, Delta l=2, the full table
is about twice as fast. For a single state at that spin and cutoff, slicing is
about 1.57 times as fast. For three states at j=2, Delta l=2, slicing is about
1.34 times as fast. Sub-millisecond differences should be treated cautiously.
These observations are limited to the listed small-spin, low-cutoff cases.

Starting without booster files largely removes the timing advantage. At
Delta l=2, three states take 5.837 / 5.976 seconds at j=1 and 18.982 / 18.808
seconds at j=2 (full / slice). These differences are small compared with the
first-calculation cost. Native vertex work accounts for over 99.7% of the timed
cold calculation in every three-state case; optimizing the final Delta4 ring
contraction would barely affect those runs. Native initialization is excluded
from both cold and warm timings, as detailed in the benchmark README.

Returned array sizes are reduced by factors of 9 at j=1 and about 41.7 at j=2
for three states, but process peak memory stays around 124–129 MiB: lookup
tables and other work dominate at these sizes. Array-size reductions must not
be presented as comparable reductions of total memory.

The j=1 amplitudes at Delta l=0,2 agree with the published finite-cutoff table
to a maximum relative difference of 1.61e-9. The two methods and their repetitions
agree within 9e-16 over the tested cases. This checks the finite-cutoff evaluation,
not the error from the omitted auxiliary-spin tail. All ten existing tests pass,
including the optional real-native Delta4 test. The native build required an
additional WIGXJPF quadmath archive, now handled by our build script without
editing the pinned EPRL sources.

Raw results, including individual timings and build/table hashes:
[three states](../benchmarks/results/2026-10-18-three-states.json),
[one state](../benchmarks/results/2026-10-18-one-state.json).

For the current production Delta4 runner, retain the full-table route: it is
validated, convenient for reuse, and cold timings show no compelling reason
to change it. The benchmark makes the slice route available for further
comparisons without introducing an automatic choice into production.

## Relevance to refinement

Delta4 at j=1 has just four bulk-spin sectors. This is useful for validation,
but cannot reveal the sampling difficulty of a refined interior with ten
internal faces. A large speedup in this benchmark would not prove that a method
scales to refinement; a small speedup does not rule it out there either.

The Donà–Frisoni paper is particularly relevant because its five-vertex graph is
the first interior subdivision we want to study. The authors report about 0.9%
sampling uncertainty for their EPRL vertex-renormalization example, under their
specific boundary state and finite cutoffs. This is neither a total error bound
including truncation nor a promised precision for our future boundary states.
Fewer sampled terms also do not translate automatically into the same factor
of wall-time savings: vertex preparation and reuse matter.

The recommended next comparison is an exact small-cutoff 1–5 calculation
against a sampler using that same amplitude convention, followed by measurements
of variance, sign cancellations, time and memory as the internal-spin cutoff
increases. Our present repository has no evaluator for that graph; implementing
and validating its gluing is necessary before that comparison can be run.
This remains a finite-complex calculation and does not itself establish a
continuum limit.

## Primary sources and inspected revisions

- Gozzini, [A high-performance code for EPRL spin foam amplitudes](https://arxiv.org/abs/2107.13952),
  especially section 3.3. Its comparisons are specific benchmarks against the
  predecessor, not a universal optimality result. The GPU examples accelerate
  selected contractions, not every stage of EPRL evaluation.
- Donà–Frisoni, [Summing bulk quantum numbers with Monte Carlo in spin foam theories](https://arxiv.org/abs/2302.00072),
  especially sections 3, 8–9;
  [released code](https://github.com/PietropaoloFrisoni/Monte_Carlo_spinfoams/tree/84b375f2a2e0d29b44dd8820953553624bb007a3).
  The five-vertex script calls the native vertex routine and performs the
  additional basis changes needed for gluing.
- Asante–Steinhaus, [Efficient Tensor Network Algorithms for Spin Foam Models](https://arxiv.org/abs/2406.19676),
  section IV.B / Algorithm 2 and the BF performance section;
  [released code](https://github.com/sethkasante/su2bf-TNAlgo/tree/2460cda77b8fe27a4106e98bf39a94fa9059bc92).
- Han et al., [Spinfoam on a Lefschetz thimble](https://arxiv.org/abs/2012.11515).
- Native source: [sl2cfoam-next, pinned revision](https://github.com/qg-cpt-marseille/sl2cfoam-next/tree/052e4346028870bd76f69a3034e6cae8defb8f7f).

No INFN jobs or GPU benchmarks were run for this comparison.
