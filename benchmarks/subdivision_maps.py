"""Boundary pullbacks induced by a specified history of edge bisections.

These maps transport a coarse K5 spin network, with trivial representations
on unused fine links. They do not define a history-independent map on the
whole refinement poset, or a nondegenerate semiclassical geometry.
"""

from itertools import combinations

import numpy as np

from benchmarks.boundary_embedding import boundary_links, coarse_boundary


def initial():
    spec = coarse_boundary()
    spec["vertices"] = tuple(tuple(int(a == b) for a in range(5)) for b in range(5))
    spec["history"] = ()
    return spec


def reduced_walk(nodes):
    """Cancel immediate retracings, including their inverse holonomies."""
    result = []
    for node in nodes:
        if result and node == result[-1]:
            continue
        if len(result) > 1 and node == result[-2]:
            result.pop()
        else:
            result.append(node)
    return tuple(result)


def bisect(spec, edge):
    """Bisect every incident cell; retain the first endpoint at each root.

    Each old boundary triangle has one carrier child, retaining edge[0].
    The elementary map is constructed for ALL old boundary links, independently
    of the K5 routes that are subsequently transported through it.
    """
    a, b = edge
    if a == b or not any(a in c and b in c for c in spec["cells"]):
        raise ValueError("Need an existing edge with distinct endpoints")
    midpoint = len(spec["vertices"])
    vertices = spec["vertices"] + (tuple((x + y) / 2 for x, y in
                                        zip(spec["vertices"][a], spec["vertices"][b])),)

    def children(simplex):
        if a in simplex and b in simplex:
            return tuple(tuple(sorted(set(simplex) - {v} | {midpoint})) for v in (b, a))
        return (simplex,)

    cells = tuple(sorted(child for cell in spec["cells"] for child in children(cell)))
    links = boundary_links(cells)
    old_nodes = {t for pair in spec["links"].values() for t in pair}
    roots = {t: children(t)[0] for t in old_nodes}
    elementary = {}
    for face, (left, right) in spec["links"].items():
        carrier = children(face)[0]
        attach_left = next(t for t in links[carrier] if t in children(left))
        attach_right = next(t for t in links[carrier] if t in children(right))
        nodes = reduced_walk((roots[left], attach_left, attach_right, roots[right]))
        faces = tuple(tuple(sorted(set(x) & set(y))) for x, y in zip(nodes, nodes[1:]))
        assert all(links[f] == tuple(sorted((x, y))) for f, x, y in zip(faces, nodes, nodes[1:]))
        elementary[face] = dict(nodes=nodes, faces=faces)
    routes = {}
    for pair, route in spec["routes"].items():
        walk = []
        for k, face in enumerate(route["faces"]):
            segment = elementary[face]["nodes"]
            if route["nodes"][k] != spec["links"][face][0]:
                segment = segment[::-1]
            walk.extend(segment)
        nodes = reduced_walk(walk)
        routes[pair] = dict(nodes=nodes, faces=tuple(tuple(sorted(set(x) & set(y)))
                                                    for x, y in zip(nodes, nodes[1:])))
    fine = dict(vertices=vertices, cells=cells, links=links,
                roots={p: roots[t] for p, t in spec["roots"].items()}, routes=routes,
                history=spec["history"] + (tuple(edge),))
    used = [t for route in routes.values() for t in route["nodes"][1:-1]]
    if len(used) != len(set(used)) or set(used) & set(fine["roots"].values()):
        raise ValueError("Transported K5 paths are not internally disjoint")
    return fine, elementary


def transport(routes, holonomies):
    values = {}
    for key, route in routes.items():
        value = np.eye(2, dtype=complex)
        for k, face in enumerate(route["faces"]):
            g = holonomies[face]
            value = value @ (g if route["nodes"][k] < route["nodes"][k + 1] else g.conj().T)
        values[key] = value
    return values


def coordinate_signature(spec):
    """Compare histories without depending on the order of new vertex IDs."""
    def node(t):
        return tuple(sorted(spec["vertices"][v] for v in t))
    return dict(cells=tuple(sorted(node(c) for c in spec["cells"])),
                roots={p: node(t) for p, t in spec["roots"].items()},
                routes={p: tuple(node(t) for t in r["nodes"]) for p, r in spec["routes"].items()})


def family():
    """Two paths reach the same four-cell triangulation, then one extends it."""
    coarse = initial()
    first, map01 = bisect(coarse, (0, 1))
    second, map12 = bisect(first, (2, 3))
    other_first, _ = bisect(coarse, (2, 3))
    other_second, _ = bisect(other_first, (0, 1))
    third, map23 = bisect(second, (0, 5))
    return [coarse, first, second, third], (map01, map12, map23), other_second


def check_composition(samples=12):
    levels, elementary_maps, alternative = family()
    rng = np.random.default_rng(1977)
    errors = []
    for steps in (2, 3):
        fine = levels[steps]
        for _ in range(samples):
            holonomies = {}
            for face in fine["links"]:
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                a, b = complex(q[0], q[1]), complex(q[2], q[3])
                holonomies[face] = np.array([[a, b], [-b.conjugate(), a.conjugate()]])
            via = holonomies
            for mapping in reversed(elementary_maps[:steps]):
                via = transport(mapping, via)
            direct = transport(fine["routes"], holonomies)
            for (a, b), value in direct.items():
                face = tuple(v for v in range(5) if v not in (a, b))
                # K5 routes run in patch order, opposite lexical tetrahedron order.
                errors.append(float(np.max(np.abs(value.conj().T - via[face]))))
    maximum = max(errors)
    if maximum > 1e-12:
        raise AssertionError("Elementary holonomy composition disagrees with transported routes")
    left, right = coordinate_signature(levels[2]), coordinate_signature(alternative)
    return dict(samples_per_depth=samples, composition_depths=[2, 3], max_matrix_absolute_error=maximum,
                same_final_triangulation=left["cells"] == right["cells"],
                commuting_disjoint_bisections_same_roots=left["roots"] == right["roots"],
                commuting_disjoint_bisections_same_routes=left["routes"] == right["routes"],
                cells=[len(s["cells"]) for s in levels],
                scope="Specified histories only; not a proof over all subdivisions")
