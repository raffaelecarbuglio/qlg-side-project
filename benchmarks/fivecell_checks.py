"""Independent small-sum checks for the regulated five-cell experiment."""

from collections import defaultdict
from itertools import combinations, product

import numpy as np
import opt_einsum as oe

from benchmarks.boundary_embedding import epsilon
from benchmarks.refinement import PAIRS


def magnetic_vertex(maps, node, spins):
    """Canonical BF tensor from magnetic indices, without native 15j or rotations."""
    operands = []
    for axis, tetrahedron in enumerate(node["tetrahedra"]):
        faces = tuple(combinations(tetrahedron, 3))
        tensor = maps.basis(tuple(spins[f] for f in faces)).copy()
        for leg, face in enumerate(faces, 1):
            edge = node["faces"].index(face)
            if PAIRS[edge][1] == axis:
                tensor = np.moveaxis(np.tensordot(epsilon(spins[face]), tensor,
                                                axes=(1, leg)), 0, leg)
        operands.extend((tensor, [10 + axis] + [node["faces"].index(f) for f in faces]))
    return oe.contract(*operands, list(range(10, 15)), optimize="greedy")


def explicit_intertwiners(arrays, inputs, outputs):
    """Enumerate every internal intertwiner; accumulate outer products of five vectors."""
    internal = sorted(set(i for ids in inputs for i in ids) - set(outputs))
    sizes = {i: a.shape[k] for a, ids in zip(arrays, inputs) for k, i in enumerate(ids)}
    total = np.zeros(tuple(sizes[i] for i in outputs), dtype=np.longdouble)
    outside = [next(i for i in ids if i in outputs) for ids in inputs]
    permutation = [outside.index(i) for i in outputs]
    for labels in product(*(range(sizes[i]) for i in internal)):
        selected = dict(zip(internal, labels))
        value = np.array(1., dtype=np.longdouble)
        for a, ids in zip(arrays, inputs):
            vector = a[tuple(selected[i] if i in selected else slice(None) for i in ids)]
            value = np.multiply.outer(value, vector)
        total += value.transpose(permutation)
    return total.astype(float)


def epsilon_walks(graph):
    """Build actual epsilon-matrix paths and cycles, independent of parity signs."""
    paths = defaultdict(lambda: defaultdict(list))
    shared = defaultdict(list)
    for v, node in enumerate(graph["nodes"]):
        for t in node["tetrahedra"]:
            shared[t].append(v)
        for a, b in combinations(range(5), 2):
            ta, tb = node["tetrahedra"][a], node["tetrahedra"][b]
            face = tuple(sorted(set(ta) & set(tb)))
            x, y = (v, ta), (v, tb)
            paths[face][x].append((y, False))
            paths[face][y].append((x, True))
    for t, vertices in shared.items():
        if len(vertices) == 2:
            x, y = (min(vertices), t), (max(vertices), t)
            for face in combinations(t, 3):
                paths[face][x].append((y, False))
                paths[face][y].append((x, True))
    result = {}
    for face, adjacency in paths.items():
        ends = sorted((x for x in adjacency if len(adjacency[x]) == 1), key=lambda x: x[1])
        start = ends[0] if ends else min(adjacency)
        here, previous, backwards = start, None, []
        while True:
            following, reverse = next((x, r) for x, r in adjacency[here] if x != previous)
            backwards.append(reverse)
            previous, here = here, following
            if here == start or (ends and here == ends[1]):
                break
        result[face] = (tuple(backwards), bool(ends))
    return result


def matrix_phase(walks, spins):
    phase = 1.
    for face, (steps, open_path) in walks.items():
        e = epsilon(spins[face])
        value = np.eye(len(e))
        for backwards in steps:
            value = value @ (e.T if backwards else e)
        target = e if open_path else np.eye(len(e))
        sign = float(np.sum(value * target) / len(e))
        np.testing.assert_allclose(value, sign * target, atol=1e-14, rtol=0)
        assert abs(abs(sign) - 1.) < 1e-14
        phase *= sign
    return phase
