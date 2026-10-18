"""Experimental finite-cutoff edgewise block, with explicit SU(2) basis maps.

Integer boundary spins, integer internal spins 0..K, unit edge weight and
internal face weight 2j+1. Boundary states use a common canonical intertwiner.
The first block has five internal faces; parity excludes half-integer spins
when the boundary spin is integer. This is not a boundary embedding or RG flow.
"""

import ctypes as ct
from functools import lru_cache
from itertools import combinations, product
import math
import time

import numpy as np
import opt_einsum as oe

from qlg.spins import LEGS, intertwiner_range, vertex_ranges
from qlg.topology import boundary, edgewise_simplex, facets

PAIRS = tuple(combinations(range(5), 2))


def layout(divisions):
    vertices, cells = edgewise_simplex(divisions)
    tetrahedra = facets(cells, 4)
    outside = boundary(cells)
    boundary_faces = {f for t in outside for f in combinations(t, 3)}
    inside_faces = tuple(sorted(set(facets(cells, 3)) - boundary_faces))
    inside_tetrahedra = tuple(sorted(t for t, n in tetrahedra.items() if n == 2))
    fi = {f: i for i, f in enumerate(inside_faces)}
    ti = {t: i + len(fi) for i, t in enumerate(inside_tetrahedra)}
    nodes = []
    for cell in cells:
        ts = tuple(tuple(v for k, v in enumerate(cell) if k != a) for a in range(5))
        fs = tuple(tuple(v for k, v in enumerate(cell) if k not in (a, b)) for a, b in PAIRS)
        nodes.append(dict(tetrahedra=ts, faces=fs,
                          spin_indices=tuple(sorted(fi[f] for f in fs if f in fi)),
                          intertwiner_indices=tuple(ti[t] for t in ts if t in ti)))
    return dict(vertices=vertices, cells=cells, nodes=nodes, face_indices=fi,
                tetrahedron_indices=ti, boundary_tetrahedra=outside)


class BasisMaps:
    """Normalized upstream 4jm tensors; overlaps perform leg permutations.

    A native vertex call frees its WIGXJPF thread workspace. Initialize and
    free a separate workspace around each uncached magnetic-basis calculation.
    """
    def __init__(self, backend):
        self.backend = backend
        self.lib = backend.lib
        self.lib.wig3jj.argtypes = [ct.c_int] * 6
        self.lib.wig3jj.restype = ct.c_double
        self.lib.wig_thread_temp_init.argtypes = [ct.c_int]
        self.lib.wig_temp_free.argtypes = []

    @lru_cache(maxsize=None)
    def basis(self, js):
        self.lib.wig_thread_temp_init(self.backend.conf.max_two_spin)
        try:
            out = np.zeros((len(intertwiner_range(*js)),) + tuple(j + 1 for j in js))
            for k, i in enumerate(intertwiner_range(*js)):
                for indices in product(*(range(j + 1) for j in js)):
                    m = tuple(2 * x - j for x, j in zip(indices, js))
                    if sum(m):
                        continue
                    out[(k,) + indices] = ((-1) ** ((i + m[0] + m[1]) // 2)
                        * self.lib.wig3jj(js[0], js[1], i, m[0], m[1], -m[0] - m[1])
                        * self.lib.wig3jj(i, js[2], js[3], m[0] + m[1], m[2], m[3])
                        * math.sqrt(i + 1))
            return out
        finally:
            self.lib.wig_temp_free()

    def overlap(self, faces, spins):
        canonical = tuple(sorted(faces))
        js = tuple(spins[f] for f in faces)
        reference = self.basis(tuple(spins[f] for f in canonical))
        local = self.basis(js).transpose((0,) + tuple(faces.index(f) + 1 for f in canonical))
        matrix = reference.reshape(len(reference), -1) @ local.reshape(len(local), -1).T
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(len(matrix)), atol=2e-14, rtol=0)
        return matrix


def native_bf(backend, spins):
    function = backend.lib.sl2cfoam_vertex_BF_fullrange
    from qlg.backend import Tensor
    function.argtypes = [ct.POINTER(ct.c_int)]
    function.restype = ct.POINTER(Tensor)
    ptr = function((ct.c_int * 10)(*spins))
    if not ptr:
        raise RuntimeError("Null BF vertex")
    try:
        t = ptr.contents
        return np.ctypeslib.as_array(t.d, shape=(t.dim,)).copy().reshape(tuple(t.dims), order="F")
    finally:
        backend.lib.sl2cfoam_vertex_free(ptr)


def verify_native_basis(backend, maps):
    """Independent magnetic-index K5 contraction vs upstream normalized 15j.

    BF is only used to validate conventions; benchmark amplitudes use EPRL.
    """
    errors = []
    for spins in ((1,) * 10, (2,) * 10, (0,) + (2,) * 9):
        operands = []
        for node, legs in enumerate(LEGS):
            tensor = maps.basis(tuple(spins[e] for e in legs)).copy()
            for k, edge in enumerate(legs):
                if PAIRS[edge][1] == node:
                    phase = np.array([(-1) ** ((spins[edge] - m) // 2)
                                      for m in range(-spins[edge], spins[edge] + 1, 2)])
                    shape = [1] * 5
                    shape[k + 1] = len(phase)
                    tensor = np.flip(tensor, axis=k + 1) * phase.reshape(shape)
            operands.extend((tensor, [10 + node, *legs]))
        direct = oe.contract(*operands, list(range(10, 15)), optimize="greedy")
        native = native_bf(backend, spins).transpose(4, 3, 2, 1, 0)
        errors.append(float(np.max(np.abs(native - direct))))
        np.testing.assert_allclose(native, direct, rtol=2e-13, atol=2e-15)
    return errors


def build_block(backend, cutoff=1, dl=0, boundary_i=1):
    """All internal spins through K, all intertwiners, fixed boundary j=1.

    Use common sorted-face orthonormal bases on each tetrahedron. The integer
    spins make (-1)^(2j) face-orientation factors unity. Native calls are cached
    by their full ordered spin tuple; no symmetry of gauge-fixed shell cutoffs
    is assumed. Returned factors have independent spin and intertwiner indices.
    """
    graph = layout(2)
    maps = BasisMaps(backend)
    basis_errors = verify_native_basis(backend, maps)
    nf = len(graph["face_indices"])
    dim_i = 2 * max(1, cutoff) + 1
    dim_j = cutoff + 1
    arrays, inputs, cache = [], [], {}
    native_seconds = 0.0
    start = time.perf_counter()
    for node in graph["nodes"]:
        spin_ids = node["spin_indices"]
        internal_axes = [a for a, t in enumerate(node["tetrahedra"])
                         if t in graph["tetrahedron_indices"]]
        tensor = np.zeros((dim_j,) * len(spin_ids) + (dim_i,) * len(internal_axes))
        for labels in product(range(dim_j), repeat=len(spin_ids)):
            assignment = dict(zip(spin_ids, labels))
            spin_map = {f: 2 * assignment[graph["face_indices"][f]]
                        if f in graph["face_indices"] else 2 for f in node["faces"]}
            spins = tuple(spin_map[f] for f in node["faces"])
            if any(not r for r in vertex_ranges(spins)):
                continue
            if spins not in cache:
                before = time.perf_counter()
                cache[spins] = backend.vertex(spins, dl)
                native_seconds += time.perf_counter() - before
            vertex = cache[spins].transpose(4, 3, 2, 1, 0)
            common_ranges = []
            for axis, legs in enumerate(LEGS):
                faces = tuple(node["faces"][e] for e in legs)
                matrix = maps.overlap(faces, spin_map)
                vertex = np.moveaxis(np.tensordot(matrix, vertex, axes=(1, axis)), 0, axis)
                common_ranges.append(intertwiner_range(*(spin_map[f] for f in sorted(faces))))
            vertex = vertex[tuple(slice(None) if a in internal_axes else boundary_i for a in range(5))]
            # All boundary spins are one; boundary_i is also the common-basis index.
            padded = np.zeros((dim_i,) * len(internal_axes))
            indices = np.ix_(*[[i // 2 for i in common_ranges[a]] for a in internal_axes])
            padded[indices] = vertex
            tensor[labels] = padded
        arrays.append(tensor)
        inputs.append(tuple(spin_ids) + node["intertwiner_indices"])
    # Each internal face weight appears once, independent of face valence.
    for index in range(nf):
        arrays.append(np.arange(dim_j, dtype=float) * 2 + 1)
        inputs.append((index,))
    scales = [float(np.max(np.abs(a))) for a in arrays]
    arrays = [a / s if s else a for a, s in zip(arrays, scales)]
    return graph, arrays, inputs, dict(
        build_seconds=time.perf_counter() - start, native_seconds=native_seconds,
        unique_native_vertices=len(cache), factor_bytes=sum(a.nbytes for a in arrays),
        log_amplitude_scale=math.fsum(math.log(s) for s in scales if s),
        native_basis_max_absolute_errors=basis_errors)


def expression(inputs):
    return ",".join("".join(oe.get_symbol(i) for i in inds) for inds in inputs) + "->"


def sector_factors(arrays, inputs, spins, nf):
    selected, remaining = [], []
    for array, indices in zip(arrays, inputs):
        selected.append(array[tuple(spins[i] if i < nf else slice(None) for i in indices)])
        remaining.append(tuple(i for i in indices if i >= nf))
    return selected, remaining


def condition_factors(arrays, inputs, assignment):
    """Fix a chosen subset of indices and leave every other sum exact."""
    selected, remaining = [], []
    for array, indices in zip(arrays, inputs):
        selected.append(array[tuple(assignment[i] if i in assignment else slice(None) for i in indices)])
        remaining.append(tuple(i for i in indices if i not in assignment))
    return selected, remaining
