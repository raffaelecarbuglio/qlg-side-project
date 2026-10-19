# Refinement protocol v1 — 19 October 2026

**Decision:** begin with an exact, regulated 1-to-5 bulk refinement at fixed
boundary, retaining the complete boundary functional in each selected spin
sector. Its first deliverable is an effective boundary amplitude and a study
of its internal-spin dependence. It is not required to equal the bare vertex.
The first [regulated half-spin calculation](fivecell-effective-amplitude.md)
now implements this step at K=1/2,1. The broader probes, regulator removal
and later refinement levels remain research tasks.

The specification below replaces an unrestricted campaign of larger meshes.
It makes a definite next experiment possible. It does not supply a proven
renormalization prescription or guarantee a continuum limit.

## 1. Target and fixed physical input

Let B be the boundary of a four-simplex, with dual graph Gamma=K5. Keep this
boundary, its orientation, its canonical normalized spin-network basis, and
each tested state psi exactly unchanged as the bulk is subdivided. This fixes
the same **kinematical boundary data**. It does not assert that we have already
constructed physical states or identified a semiclassical spacetime.

Use the existing Lorentzian EPRL convention throughout: gamma=1.2,
rho=gamma(j+1), unit internal edge weight, internal face weight 2j+1,
and sqrt(2j+1) per boundary triangle. Change of convention requires a separate
experiment. Keep all admissible physical spins and intertwiners within the
explicit regulator; use no symmetry-reduced internal model.

For a triangulation Delta with boundary B, define

```math
W_{\Delta,K,L}(s)=
\left(\prod_{f\subset B}\sqrt{2j_f+1}\right)
\sum_{\substack{j_f\le K\;(f\not\subset B)\\ i_e\;\mathrm{admissible}}}
\left(\prod_{f\not\subset B}(2j_f+1)\right)
\operatorname{Contract}_{s}\left[\bigotimes_{v\in\Delta} A_v^{(L)}\right].
```

The contraction includes the canonical recoupling and orientation maps; s
specifies the boundary labels. K bounds physical internal spins, including
half-integers. L=Delta l truncates the separate auxiliary sums within vertices.
Neither is a refinement level. Numerical integration accuracy is another
control and is kept at VERYHIGH initially.

The first bare-model question is whether, for each fixed boundary spin network,

```math
W_B(s)=\lim_{\Delta\,\mathrm{refines}\,B}
       \left[\lim_{K\to\infty}\lim_{L\to\infty}W_{\Delta,K,L}(s)\right]
```

exists and is independent of the refinement path and admissible regulator
removal procedure. The outer symbol denotes the directed refinement problem,
not a claim that our sampled sequence is cofinal. Divergence of an inner limit
blocks this particular bare prescription; it does not disprove a renormalized
theory or all possible joint limits.

The test domain is the finite linear span of spin networks on Gamma. A limit
on every basis state defines a linear functional there. Extending this to
all boundary graphs, manifolds and gluing is additional work. Finite-sector
norms are diagnostic tools; convergence of the full Hilbert-space vector is
not imposed. This follows the pointwise target of [1], whose EPRL convergence
is an assumption, not an established construction.

## 2. Exact boundary probes

Use every intertwiner coefficient in each sector, in this order:

| Sector on the ten K5 links | Coefficients |
|---|---:|
| All j=1/2 | 32 |
| j=1 on links 01,02,12; j=1/2 on the other links | 32 |
| All j=1 | 243 |

The mixed sector breaks the homogeneous spin choice. These 307 coefficients
are an initial probe set, not a dense set by themselves. Subsequent coverage
must include all admissible assignments below an increasing boundary-spin
bound, rather than just increasing the common homogeneous spin.

Use absolute signed coefficients as the primary data. Norms, normalized
directions and ratios are secondary diagnostics, not new model amplitudes.
Boundary operator insertions W(O psi) can be reconstructed where O stays in
the measured space. Do not label these as Dirac observables without a physical
inner product and a constraint-compatibility argument. No artificial boundary
embedding or prescribed zero-spin refinement vacuum is needed at this stage.

## 3. Refinement complexes and histories

Start with 01234. Insert interior vertex 5 and replace it by its five cones:
01235,01245,01345,02345,12345. The boundary is exactly unchanged. The five-cell
ball has ten internal triangles and ten internal tetrahedra.

The first branching comparison is fixed in advance:

- A: split cell 01235 by interior vertex 6, giving nine cells.
- B: split cell 12345 by interior vertex 7, giving another nine-cell ball.
- C: perform both splits, giving thirteen cells. Both orders give the exact
  same triangulation and hence the same state and amplitude to numerical
  accuracy. A and B themselves need not have equal amplitudes at finite size.

Thus compare 1,5,9A,9B,13 cells as a **finite refinement window**. A reference
longer family is repeated subdivision of every nonboundary face in decreasing
dimension, holding the entire boundary subcomplex fixed. Each round snapshots
the current faces, then stellarly subdivides their stars, ordered first by
decreasing face size and then lexicographically. Vertices need no subdivision.
This also subdivides old internal edges and triangles, unlike repeated
1-to-5 moves alone. Its first complete rounds have 1,5,325 four-simplices.

Do not schedule the 325-cell sum now. Individual stellar moves provide
intermediate complexes. Seed the longer construction separately with A and B
to investigate history dependence. A proof of geometric mesh control and of
cofinality in the intended triangulation order is not supplied by these
combinatorial checks. Cell count and auxiliary barycentric coordinates are
not a physical lattice spacing in a sum over geometries. The fixed coarse
boundary also limits which continuum modes this programme can test.

The reproducible audit [4] verifies the identical boundaries, the common
13-cell refinement, the complete-round counts, boundary-sector dimensions,
and initial spin-assignment counts. It computes no physical amplitudes.

## 4. What coarse graining means here

Define the effective tensor T5(K,L) to be exactly the entire functional
W5(K,L) on B after summing the interior. Compare it with T1(L) coefficient by
coefficient. Retain T5 as a general boundary tensor: do not force it back into
the EPRL vertex form by adjusting only gamma or a scalar. Its change in
boundary dependence is the information to preserve, as our 3-to-3 data show.
The microscopic vertex remains fixed in this first experiment; effective
weights change because degrees of freedom were actually summed out [2].

In the orthonormal boundary basis, exact effective coefficients simply are
g_s(K,L)=T5(K,L)[s]. No fitted parameter space is needed for the initial finite
probe sectors. However, these sectors alone are **insufficient to reuse T5
as an internal building block**: all interface-spin sectors allowed by K must
be supplied. We must not iterate the 32-component half-spin tensor as if it
were a full effective vertex.

Gluing tensors that already contain sqrt(d_j) boundary factors needs explicit
interface weights. If a shared triangle occurs on m block boundaries, multiply
by d_j^(1-m/2) when it becomes internal, or d_j^((1-m)/2) if it remains on the
external boundary, with d_j=2j+1. These factors turn the product of block
boundary weights into the one desired face weight. Shared intertwiners still
require the normalized orientation maps. Count each regulator once per final
internal face. This formula concerns un-subdivided identified interface
triangles; it is not an embedding for a changing boundary graph.

At a fixed finite regulator, direct and staged contraction of the same
fully specified complex must agree. Equality between T1 and T5 is not required.
Compression, if eventually needed, must be checked against this exact result
and retain its scale and error separately.

## 5. Regulators, divergence and normalization

The 1-to-5 spin sum is genuinely unbounded. With boundary j=1/2, setting all
ten internal spins to any half-integer x>=1/2 satisfies tetrahedral
admissibility: each internal tetrahedron has spins (1/2,x,x,x). This is an
unbounded admissible family, not a claim that every corresponding amplitude
is nonzero. No finite endpoint analogous to the 3-to-3 sum exists.

Start with K=1/2 and 1. They have respectively 16 and 721 admissible internal
spin assignments and require 8 and 81 distinct ordered native vertices for
the homogeneous half-spin boundary and ascending cell frames. These are
counts, not runtime estimates. Enumerate before computing; contract exactly
at these small regulators and preserve all 32 boundary coefficients together.

For each K, scan L=0,2,4 and extend L as needed, using L=8,12 as measured
starting references rather than guaranteed sufficient values. Check different
local frame orderings after canonical boundary recoupling. Increase K to
3/2,2,... only after measuring the cost and resolving the shell dependence
at the preceding values. Never infer physical-spin convergence from a scan
performed at an uncontrolled fixed L.

Compare regulator increments as vectors: Delta_K T=T(K')-T(K), retaining
both their size and direction in the same boundary space. If they look
unbounded, distinguish a common multiplicative growth from changes of shape.
Power-counting literature on this ball [3] makes divergence a serious
possibility; its estimates and conventions are not a proven exponent for our
normalization. An apparently constant shape does not establish a finite
unnormalized functional.

No division by a separately fitted per-level or per-state factor is allowed.
An actual subtraction or change in microscopic weights would define a
renormalized prescription and needs specified counterterms, renormalization
conditions, and gluing tests on other complexes. Those are **not yet derived**.
If needed, stop the bare refinement claim here and make their derivation the
research problem. A finite-spin cutoff is not a removal of gauge volume;
neither a BF gauge fixing nor a vertex-displacement division can be imported
into EPRL without a derivation.

If the K scan appears stable, confirm with another exhaustion of the same
spin sums, for example a bound on the sum of internal spins rather than each
spin separately. Agreement of their finite cutoff parameters is not expected;
agreement of a claimed regulator-independent limit is.

## 6. Numerical decisions and stopping conditions

For each finite boundary sector fix S0=max_s |T1(s)| at controlled shell
accuracy; if this vanishes, specify a different fixed reference before any
relative comparison. Keep absolute values in every report. For vectors A,B
use max_s |A_s-B_s|, with the provisional tolerance

```math
10^{-6}S_0+0.01\max(\|A\|_\infty,\|B\|_\infty).
```

Require the numerical changes to be smaller than the refinement effect being
interpreted; if they overlap, label that comparison unresolved. A provisional
plateau requires agreement across three successive K values, two further
shell increments, and the ordering check. This is an operational criterion,
not a theorem or a bound on an omitted tail. For a limit near zero, the fixed
absolute tolerance matters: a zero candidate must be reported as such rather
than rescaled into a nonzero answer.

The [first five-cell results](fivecell-effective-amplitude.md) are strongly
suppressed relative to S0. The coarse-reference absolute floor by itself
therefore cannot resolve their K or L dependence. Being below that floor
does not establish a zero result or regulator removal. For a resolved
nonzero effective amplitude, separately report relative regulator and
ordering changes at the amplitude's own scale; the initial 1% relative
targets remain unmet. Retain both kinds of diagnostic without rescaling W.

Only after five-cell regulator behavior is understood proceed to 9A,9B,13
with the same convention and probe states. Look for stability on a tail of
levels, not equality of the first two levels. One stable finite window does
not prove convergence of a sequence or of the full directed net. Divergence,
nonmonotonic drift or history dependence must remain visible in the report.

For the 1-to-5 BF control, do **not** require a raw finite-K amplitude to equal
the single BF vertex: the internal-vertex configuration has bubble/gauge
redundancies, and the regulator changes the topological identities. First
check an independent magnetic-index/recoupling evaluation of the same finite
sum at the smallest K. Use a Pachner identity only with an explicitly derived
gauge/normalization treatment. The successful 3-to-3 BF check does not remove
this new issue.

## 7. First deliverable and INFN decision

The first implementation milestone is the full **regulated five-cell effective
tensor** in the 32-component half-spin sector, with exact internal sums
at K=1/2,1, separate shell control, and an independent smallest-K contraction
check. The [implementation and report](fivecell-effective-amplitude.md) retain
T5, T1, their difference, internal-spin increments and measured cost.
Then add the mixed and j=1 boundary probes before claiming robustness
across boundary data. This is one coarse-graining step, not a full RG flow.

Go to INFN when this specifies a validated calculation whose obstacle is
measured time or memory, with checkpoints and explicit K,L jobs. Do not use a
larger machine to bypass a missing divergence prescription or to run the
325-cell level with uncontrolled sums.

A general continuum construction will additionally require compatible
boundary refinements, broader manifolds/cylinders and their gluing, and a
test domain covering those boundaries. Fixed B deliberately postpones that
ambiguity; it does not solve it. This is the strongest concrete starting
protocol justified here, not a claim of an optimal or completed construction.

## References and audit

1. Bruno, Colafranceschi, Mele, Rovelli, [The Structure of the Continuum Limit
   of Spin Foams](https://arxiv.org/html/2603.16999v2), especially §5.3.
2. Asante, Dittrich, Steinhaus, [Spin foams, Refinement limit and
   Renormalization](https://arxiv.org/abs/2211.09578), especially §4–5.
3. Donà, [Infrared divergences in the EPRL-FK Spin Foam model](https://arxiv.org/abs/1803.00835),
   especially §6.4; estimates are not an exact convergence theorem.
4. [Audit code](../benchmarks/refinement_protocol.py) and
   [verified combinatorial data](../benchmarks/results/2026-10-19-protocol.json).
   Reproduce with `python -m benchmarks.refinement_protocol --output PATH`.
