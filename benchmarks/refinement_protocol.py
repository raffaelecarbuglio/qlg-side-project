"""Combinatorial audit of the proposed protocol; no new EPRL amplitudes."""

import argparse
from itertools import combinations, product
from math import prod

from qlg.jobs import atomic_json
from qlg.spins import intertwiner_range
from qlg.topology import boundary, facets


def stellar(cells, face, vertex):
    """Subdivide the entire star of a face, preserving conformity."""
    face = set(face)
    assert face and any(face <= set(c) for c in cells)
    assert all(vertex not in c for c in cells)
    return tuple(sorted([c for c in cells if not face <= set(c)] +
        [tuple(sorted(set(c) - {a} | {vertex}))
         for c in cells if face <= set(c) for a in sorted(face)]))


def relative_round(cells):
    """Stellarly subdivide all nonboundary faces, in decreasing dimension."""
    outside = boundary(cells)
    faces = {s for c in cells for k in range(2, 6) for s in combinations(c, k)}
    fixed = {s for t in outside for k in range(2, 5) for s in combinations(t, k)}
    vertex = 1 + max(v for c in cells for v in c)
    for face in sorted(faces - fixed, key=lambda s: (-len(s), s)):
        cells = stellar(cells, face, vertex)
        vertex += 1
    assert boundary(cells) == outside
    return cells


def counts(cells):
    outside = boundary(cells)
    boundary_faces = {f for t in outside for f in combinations(t, 3)}
    return dict(cells=len(cells), vertices=len({v for c in cells for v in c}),
        internal_triangles=len(set(facets(cells, 3)) - boundary_faces),
        internal_tetrahedra=sum(n == 2 for n in facets(cells, 4).values()),
        boundary_tetrahedra=[list(t) for t in outside])


def assignments(cells, two_k):
    boundary_faces = {f for t in boundary(cells) for f in combinations(t, 3)}
    internal = sorted(set(facets(cells, 3)) - boundary_faces)
    tetrahedra = [tuple(combinations(t, 3)) for t in facets(cells, 4)]
    vertex_faces = [tuple(tuple(v for k, v in enumerate(c) if k not in (a, b))
                         for a, b in combinations(range(5), 2)) for c in cells]
    total, unique = 0, set()
    for values in product(range(two_k + 1), repeat=len(internal)):
        spins = {f: 1 for f in boundary_faces}
        spins.update(zip(internal, values))
        if not all(intertwiner_range(*(spins[f] for f in t)) for t in tetrahedra):
            continue
        total += 1
        unique.update(tuple(spins[f] for f in fs) for fs in vertex_faces)
    return dict(two_k=two_k, admissible_spin_assignments=total,
                unique_ordered_native_vertices=len(unique))


def audit():
    single = (tuple(range(5)),)
    five = relative_round(single)
    a, b = five[0], five[-1]
    nine_a, nine_b = stellar(five, a, 6), stellar(five, b, 7)
    common = stellar(nine_a, b, 7)
    assert common == stellar(nine_b, a, 6)
    cases = dict(single=single, five=five, nine_a=nine_a, nine_b=nine_b, common13=common)
    assert all(boundary(c) == boundary(single) for c in cases.values())
    second = relative_round(five)
    assert len(second) == 325
    pairs = tuple(combinations(range(5), 2))
    cycle = {(0, 1), (0, 2), (1, 2)}
    sectors = {}
    for name, spins in (
        ("half", {e: 1 for e in pairs}),
        ("mixed_triangle", {e: 2 if e in cycle else 1 for e in pairs}),
        ("one", {e: 2 for e in pairs}),
    ):
        dims = [len(intertwiner_range(*(spins[e] for e in pairs if v in e))) for v in range(5)]
        sectors[name] = dict(two_spins=[spins[e] for e in pairs],
                             node_dimensions=dims, dimension=prod(dims))
    low_cutoffs = [assignments(five, k) for k in (1, 2)]
    assert [(r["admissible_spin_assignments"], r["unique_ordered_native_vertices"])
            for r in low_cutoffs] == [(16, 8), (721, 81)]
    return dict(status="combinatorics_verified_amplitudes_unimplemented",
        cases={name: dict(**counts(c), simplices=[list(s) for s in c]) for name, c in cases.items()},
        split_cells=dict(a=list(a), b=list(b)), two_orders_same_common_refinement=True,
        second_complete_relative_round=counts(second), boundary_sectors=sectors,
        half_spin_initial_cutoffs=low_cutoffs,
        scope="Finite combinatorial audit; no mesh convergence, cofinality, or amplitude result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    atomic_json(args.output, result)
    print(result["status"])
