# Mathematical and numerical specification

## Vertex

We evaluate the gauge-fixed Lorentzian EPRL four-simplex vertex of sl2cfoam-next,
with the default compiled Y-map convention `rho = gamma*(j+1)`. The build script
does not set `RHO_GJ`. Different conventions require separate reference data and
a new run directory.

Zero-spin refinement probes need particular care with this convention. The
[matched-boundary tests](matched-boundary-tests.md) include a separate
rho=gamma*j build control: positive-spin vertices work, but its rho=0 routine
is unimplemented in the pinned upstream source. No alternative-convention
matched-refinement amplitude has therefore been obtained.

Input spin order:

```text
j12 j13 j14 j15 j23 j24 j25 j34 j35 j45
```

Intertwiner leg order (the first two spins are coupled against the last two):

| Label | Ordered spins |
|---|---|
| i1 | j12 j13 j14 j15 |
| i2 | j23 j24 j25 j12 |
| i3 | j34 j35 j13 j23 |
| i4 | j45 j14 j24 j34 |
| i5 | j15 j25 j35 j45 |

The returned array uses axes `(i5,i4,i3,i2,i1)` and Fortran/column-major storage.
An axis starts at that intertwiner's **minimum admissible value**, not necessarily
zero. `spins.py` computes the parity-correct ranges. The Python wrapper copies
the tensor before releasing native memory and checks dimensions and strides.

The parameter Delta l truncates the auxiliary sums to `j <= l <= j + Delta l`.
Every scan point is a finite-cutoff numerical approximation to the same EPRL
vertex, not a separate simplified-EPRL theory taken as the final answer.

## Delta4

The implemented contraction follows Donà & Frisoni's accompanying
`VertexComputation.ipynb` and `VertexContraction.ipynb`, at repository revision
`4fa7e31ffc6553e56da94a443a53a4059a5d2035`.

All boundary face spins are fixed to j. There is one bulk face, with spin x.
For each x, compute the vertex tensor with spins

```text
(j,j,j,j,j,j,j,x,j,j).
```

Allowed bulk spins are `0,1,...,3j` for integer j and
`1/2,3/2,...,3j` for half-integer j. These finite ranges follow from boundary
admissibility; there is no additional physical-spin cutoff K in this complex.
All admissible internal intertwiners are retained.

For each of the four vertices r, fix its three external intertwiner indices
`(a_r,b_r,c_r)` and form

```text
M_r = T_x[a_r, :, :, b_r, c_r].
```

The amplitude is

```math
W_{\Delta_4}(\psi;\Delta l)
=\sum_x(2x+1)\operatorname{tr}(M_0M_1M_2M_3).
```

For the published uniform boundary intertwiner i, all twelve external labels
equal i. No extra factors `(2i+1)` are inserted into the contraction: the
normalization is the one used by the native tensors and published notebook.
Boundary factors are treated as in that benchmark, rather than added by a new
general triangulation convention.

The contraction uses extended precision where NumPy's `longdouble` supports it;
native tensor entries remain double precision. Bulk-spin contributions are
combined using compensated summation (`math.fsum`). Signed entries are retained.
The reported cancellation ratio measures only cancellations **between bulk-spin
sectors**, not inside each vertex or intertwiner contraction.

## Interpretation

The homogeneous boundary spins and selected boundary states specify the
amplitude being asked for. They do not freeze the internal EPRL variables or
replace them with cuboidal geometries. Nevertheless, these are only particular
boundary probes, not the full boundary amplitude functional.

Different Delta l values refine a numerical sum on the same complex. Changing
j changes the boundary area scale. Neither operation increases the number of
four-simplices. In particular, agreement with the published table validates this
finite-complex calculation; it is not evidence of a continuum limit.
