"""Exact support reduction and experimental compression of refinement networks.

Compression is a numerical approximation, not a renormalization prescription.
Fixed-spin sectors must not be confused with a sum over internal face spins.
"""

from itertools import combinations
import math
import time

import numpy as np
import opt_einsum as oe

from benchmarks.refinement import BasisMaps, layout
from qlg.spins import LEGS


def fixed_sector(backend, divisions=2, dl=0, boundary_i=1):
    """j=1 on every triangle, with complete internal intertwiner sums."""
    graph = layout(divisions)
    maps = BasisMaps(backend)
    vertex = backend.vertex((2,) * 10, dl).transpose(4, 3, 2, 1, 0)
    # With globally ordered simplex vertices, these five permutations are
    # identical in every simplex of the complex.
    node = graph["nodes"][0]
    spin_map = dict.fromkeys(node["faces"], 2)
    for axis, legs in enumerate(LEGS):
        matrix = maps.overlap(tuple(node["faces"][e] for e in legs), spin_map)
        vertex = np.moveaxis(np.tensordot(matrix, vertex, axes=(1, axis)), 0, axis)
    arrays, inputs, logscale = [], [], 0.0
    for node in graph["nodes"]:
        selected = tuple(slice(None) if t in graph["tetrahedron_indices"] else boundary_i
                         for t in node["tetrahedra"])
        array = vertex[selected].copy()
        scale = float(np.max(np.abs(array)))
        if scale:
            array /= scale
            logscale += math.log(scale)
        arrays.append(array)
        inputs.append(node["intertwiner_indices"])
    logscale += len(graph["face_indices"]) * math.log(3)
    return graph, arrays, inputs, logscale


def prune_support(arrays, inputs):
    """Remove only exactly zero-supported index values, iterating to closure.

    If any factor forbids a value, that value cannot contribute to the product.
    No floating-point tolerance is used. Works for ordinary and hyper indices.
    """
    arrays = [a.copy() for a in arrays]
    removed = 0
    while True:
        domains = {}
        for array, indices in zip(arrays, inputs):
            for axis, index in enumerate(indices):
                mask = np.any(array != 0, axis=tuple(i for i in range(array.ndim) if i != axis))
                domains[index] = domains.get(index, np.ones_like(mask)) & mask
        changed = {i: np.flatnonzero(mask) for i, mask in domains.items() if not np.all(mask)}
        if not changed:
            break
        if any(len(v) == 0 for v in changed.values()):
            return [np.asarray(0.0)], [()], dict(removed_values=removed, zero=True)
        removed += sum(len(domains[i]) - len(values) for i, values in changed.items())
        for k, indices in enumerate(inputs):
            for axis, index in enumerate(indices):
                if index in changed:
                    arrays[k] = np.take(arrays[k], changed[index], axis=axis)
    return arrays, list(inputs), dict(removed_values=removed, zero=False)


def local_contract(arrays, inputs, max_entries=10000):
    """Exact connected contractions, bounded by the output size.

    Every summed index must occur on just the two contracted factors. Hyper
    indices are retained until all of their incident factors are present.
    """
    arrays, inputs = list(arrays), list(map(tuple, inputs))
    steps = 0
    largest_generated = 0
    while len(arrays) > 1:
        locations = {}
        for k, indices in enumerate(inputs):
            for index in indices:
                locations.setdefault(index, set()).add(k)
        candidates = set()
        for incident in locations.values():
            candidates.update(combinations(sorted(incident), 2))
        choices = []
        for a, b in sorted(candidates):
            involved = set(inputs[a]) | set(inputs[b])
            output = tuple(sorted(i for i in involved if not locations[i] <= {a, b}))
            sizes = dict(zip(inputs[a], arrays[a].shape)) | dict(zip(inputs[b], arrays[b].shape))
            size = math.prod(sizes[i] for i in output)
            if size <= max_entries:
                choices.append((size, a, b, output))
        if not choices:
            break
        _, a, b, output = min(choices)
        result = oe.contract(arrays[a], inputs[a], arrays[b], inputs[b], output, optimize="greedy")
        largest_generated = max(largest_generated, result.size)
        for k in sorted((a, b), reverse=True):
            del arrays[k], inputs[k]
        arrays.append(result)
        inputs.append(output)
        steps += 1
    return arrays, inputs, dict(contractions=steps, remaining_tensors=len(arrays),
                               largest_factor_entries=max(a.size for a in arrays),
                               largest_generated_entries=largest_generated)


def compressed_contract(arrays, inputs, chi, seed=37, gauges=1,
                        max_entries=32_000_000, max_seconds=90, mode="basic"):
    """SVD-truncated contraction with a measured cap on each intermediate.

    Returned logarithms refer to the supplied factors. No physical input scale
    is included. Increasing chi is a diagnostic, not an error certificate.
    """
    import cotengra as ctg
    import quimb.tensor as qtn

    start = time.perf_counter()
    network = qtn.TensorNetwork([qtn.Tensor(a.copy(), inds=tuple(f"i{i}" for i in inds))
                                 for a, inds in zip(arrays, inputs)])
    if any(len(tids) > 2 for tids in network.ind_map.values()):
        network = network.hyperinds_resolve(mode="tree", sorter="clustering")
    optimizer = ctg.HyperCompressedOptimizer(
        chi=chi, methods=["greedy-compressed", "greedy-span"], optlib="random",
        optlib_opts={"seed": seed}, max_repeats=16, max_time=5, parallel=False)
    tree = network.contraction_tree(optimizer, output_inds=())
    planning = time.perf_counter() - start
    stats = dict(peak_factor_entries=max(t.size for t in network), compressions=0)

    def pre_contract(tn, tids):
        if time.perf_counter() - before > max_seconds:
            raise TimeoutError(f"Contraction exceeded {max_seconds}s")
        a, b = (tn.tensor_map[k] for k in tids)
        shared = a.bonds(b)
        size = a.size * b.size // math.prod(a.ind_size(i)**2 for i in shared)
        if size > max_entries:
            raise MemoryError(f"Contraction would create {size} entries (cap {max_entries})")

    def post_contract(tn, tid):
        stats["peak_factor_entries"] = max(stats["peak_factor_entries"], tn.tensor_map[tid].size)

    def post_compress(tn, tids):
        stats["compressions"] += 1
        stats["peak_factor_entries"] = max(stats["peak_factor_entries"],
                                            *(tn.tensor_map[t].size for t in tids))

    before = time.perf_counter()
    mantissa, exponent = network.contract_compressed(
        tree, max_bond=chi, cutoff=0.0, tree_gauge_distance=gauges,
        compress_mode=mode,
        strip_exponent=True, equalize_norms=True,
        callback_pre_contract=pre_contract, callback_post_contract=post_contract,
        callback_post_compress=post_compress)
    mantissa = float(mantissa)
    if not math.isfinite(mantissa) or not math.isfinite(exponent):
        raise FloatingPointError("Non-finite compressed contraction")
    return dict(chi=chi, seed=seed, gauge_distance=gauges, search_seconds=planning,
                compression_mode=mode,
                contract_seconds=time.perf_counter() - before,
                sign=float(np.sign(mantissa)),
                log_absolute_value=(float(math.log(abs(mantissa)) + exponent * math.log(10))
                                    if mantissa else None),
                path=[[int(i) for i in p] for p in tree.get_path()],
                **{k: int(v) for k, v in stats.items()})


def parity_structure(divisions):
    """Exact GF(2) constraints on integer versus half-integer internal spins.

    Four spins at every tetrahedron must have integer sum. The boundary spins
    here are integers, so the right-hand side of each binary equation is zero.
    This is necessary admissibility, not the full triangle-inequality test.
    """
    graph = layout(divisions)
    fi = graph["face_indices"]
    rows = [sum(1 << fi[f] for f in combinations(t, 3) if f in fi)
            for t in graph["tetrahedron_indices"]]

    def rank(vectors):
        pivots = {}
        for value in vectors:
            while value:
                p = value.bit_length() - 1
                if p not in pivots:
                    pivots[p] = value
                    break
                value ^= pivots[p]
        return len(pivots)

    boundary_edges = {e for t in graph["boundary_tetrahedra"] for e in combinations(t, 2)}
    edges = sorted({e for cell in graph["cells"] for e in combinations(cell, 2)} - boundary_edges)
    columns = [sum(1 << index for f, index in fi.items() if set(edge) <= set(f)) for edge in edges]
    assert all((row & column).bit_count() % 2 == 0 for row in rows for column in columns)
    independent = rank(rows)
    return dict(divisions=divisions, simplices=len(graph["cells"]),
                internal_faces=len(fi), parity_constraint_rank=independent,
                independent_parity_labels=len(fi) - independent,
                internal_edge_generators=len(edges), generator_rank=rank(columns))
