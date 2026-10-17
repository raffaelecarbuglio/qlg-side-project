"""Combinatorial refinement checks, independent of amplitude conventions.

This module does not assign EPRL amplitudes to arbitrary triangulations.
"""

from collections import Counter
from itertools import combinations


def facets(simplices, size):
    return Counter(face for simplex in simplices for face in combinations(sorted(simplex), size))


def boundary(simplices):
    counts = facets(simplices, 4)
    if any(n > 2 for n in counts.values()):
        raise ValueError("A tetrahedron belongs to more than two four-simplices")
    return tuple(sorted(face for face, n in counts.items() if n == 1))


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
