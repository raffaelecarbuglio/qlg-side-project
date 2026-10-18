"""Combinatorial refinement checks, independent of amplitude conventions.

This module does not assign EPRL amplitudes to arbitrary triangulations.
"""

from collections import Counter
from itertools import combinations, permutations, product


def facets(simplices, size):
    return Counter(face for simplex in simplices for face in combinations(sorted(simplex), size))


def boundary(simplices):
    counts = facets(simplices, 4)
    if any(n > 2 for n in counts.values()):
        raise ValueError("A tetrahedron belongs to more than two four-simplices")
    return tuple(sorted(face for face, n in counts.items() if n == 1))


def edgewise_simplex(divisions):
    """Ordered edgewise subdivision of one abstract four-simplex.

    Return integer barycentric coordinates and cells using vertex indices.
    ``divisions=2,4,8`` gives 16,256,4096 cells. Coordinates specify the
    subdivision, not a physical metric or fixed areas in the spin-foam sum.
    Construction: Freudenthal triangulation of 0<=y3<=y2<=y1<=y0<=divisions.
    """
    if type(divisions) is not int or divisions < 1:
        raise ValueError("divisions must be a positive integer")
    cells = []
    for base in product(range(divisions), repeat=4):
        if any(base[i] < base[i + 1] for i in range(3)):
            continue
        for order in permutations(range(4)):
            point = list(base)
            chain = [tuple(point)]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                chain.append(tuple(point))
            if all(all(p[i] >= p[i + 1] for i in range(3)) for p in chain):
                cells.append(tuple((divisions - p[0], p[0] - p[1],
                                    p[1] - p[2], p[2] - p[3], p[3]) for p in chain))
    vertices = tuple(sorted({v for cell in cells for v in cell}))
    lookup = {v: k for k, v in enumerate(vertices)}
    return vertices, tuple(tuple(sorted(lookup[v] for v in cell)) for cell in cells)


def refine(simplices, index):
    """Insert an interior vertex into one four-simplex (a 1->5 subdivision)."""
    simplices = [tuple(sorted(s)) for s in simplices]
    if any(len(s) != 5 or len(set(s)) != 5 for s in simplices):
        raise ValueError("Need four-simplices with five distinct vertex labels")
    old = simplices[index]
    new_vertex = max(v for s in simplices for v in s) + 1
    result = simplices[:index] + simplices[index + 1:]
    result.extend(tuple(sorted(face + (new_vertex,))) for face in combinations(old, 4))
    if boundary(simplices) != boundary(result):
        raise RuntimeError("Subdivision changed the external boundary")
    return result


def refinement_plan(steps):
    simplices = [tuple(range(5))]
    levels = []
    for level in range(steps + 1):
        tetrahedra = facets(simplices, 4)
        triangles = facets(simplices, 3)
        outside = boundary(simplices)
        boundary_triangles = {f for t in outside for f in combinations(t, 3)}
        internal_faces = len(set(triangles) - boundary_triangles)
        levels.append({"level": level, "simplices": [list(s) for s in simplices],
                       "boundary_tetrahedra": [list(t) for t in outside],
                       "internal_tetrahedra": sum(n == 2 for n in tetrahedra.values()),
                       "internal_triangles": internal_faces})
        if level < steps:
            simplices = refine(simplices, 0)
    return {"kind": "combinatorial_1_to_5_plan", "levels": levels,
            "amplitudes_implemented": False,
            "remaining": "Specify and validate recoupling/orientation maps and normalization before evaluating these complexes."}
