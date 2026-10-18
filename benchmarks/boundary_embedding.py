"""Exact SU(2) graph-subdivision embeddings, not a dynamical vacuum.

Keep a routed copy of K5 on the fine boundary, put j=0 on unused links,
and insert normalized bivalent intertwiners along each routed link.
All spin labels are doubled integers, including half-integer tests.
"""

from collections import defaultdict
from itertools import combinations
import math

import numpy as np
import opt_einsum as oe

from qlg.topology import boundary, edgewise_simplex


def epsilon(two_j):
    return np.diag([(-1.) ** k for k in range(two_j, -1, -1)])[:, ::-1]


def representation(g, two_j):
    """Symmetric powers needed here: j=0,1/2,1, in increasing-m order."""
    if two_j == 0:
        return np.ones((1, 1), dtype=complex)
    if two_j == 1:
        return g
    if two_j == 2:
        symmetric = np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0],
                              [0, 0, 0, 1]])
        return symmetric @ np.kron(g, g) @ symmetric.T
    raise ValueError("Boundary representation check supports j<=1")


def area_and_flux_check():
    """Matching an additive area does not by itself match a coarse flux.

    Four spin-1/2 pieces have the same summed area as spin 3, but their
    combined SU(2) representation contains only J=0,1,2. Areas omit the
    common 8*pi*gamma*l_P^2 factor; flux magnitudes are sqrt(J*(J+1)).
    """
    allowed = {0}
    for _ in range(4):
        allowed = {coupled for j in allowed for coupled in range(abs(j - 1), j + 2, 2)}
    coarse_area = math.sqrt(3 * 4)
    fine_area = 4 * math.sqrt(.5 * 1.5)
    np.testing.assert_allclose(coarse_area, fine_area, atol=1e-15, rtol=0)
    assert 6 not in allowed
    return dict(coarse_j=3, fine_j=.5, pieces=4, coarse_area=coarse_area,
                summed_fine_area=fine_area, allowed_total_J=[j/2 for j in sorted(allowed)],
                coarse_flux_representation_available=False,
                interpretation="Equal additive areas alone do not specify a matched quantum geometry")


def boundary_links(cells):
    links = defaultdict(list)
    for tetrahedron in boundary(cells):
        for face in combinations(tetrahedron, 3):
            links[face].append(tetrahedron)
    if any(len(pair) != 2 for pair in links.values()):
        raise ValueError("Expected a closed simplicial boundary")
    return {f: tuple(sorted(pair)) for f, pair in sorted(links.items())}


def coarse_boundary():
    roots = {a: tuple(v for v in range(5) if v != a) for a in range(5)}
    cells = (tuple(range(5)),)
    return dict(vertices=tuple(range(5)), cells=cells, links=boundary_links(cells), roots=roots,
                routes={(a, b): dict(faces=(tuple(v for v in range(5) if v not in (a, b)),),
                                     nodes=(roots[a], roots[b])) for a, b in combinations(range(5), 2)})


def routed_boundary(kind="edgewise", reverse=False):
    """Find disjoint K5 paths within the five inherited tetrahedron patches.

    'split' bisects edge (0,1) once. 'edgewise' uses the actual 16-cell block.
    A reversed deterministic search supplies a second, independently routed map.
    """
    if kind == "edgewise":
        vertices, cells = edgewise_simplex(2)
        divisions = 2
    elif kind == "split":
        vertices = tuple(tuple(int(a == b) for a in range(5)) for b in range(5))
        vertices += ((.5, .5, 0, 0, 0),)
        cells = ((0, 2, 3, 4, 5), (1, 2, 3, 4, 5))
        divisions = 1
    else:
        raise ValueError(kind)
    links = boundary_links(cells)
    owner = {t: next(a for a in range(5) if all(vertices[v][a] == 0 for v in t))
             for t in boundary(cells)}
    adjacency, ports = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for face, (a, b) in links.items():
        if owner[a] == owner[b]:
            adjacency[a].append((b, face))
            adjacency[b].append((a, face))
        else:
            ports[a][owner[b]].append(face)
            ports[b][owner[a]].append(face)
    options = {}
    for patch in range(5):
        choices = []
        others = [b for b in range(5) if b != patch]
        for root in sorted(t for t in owner if owner[t] == patch):
            paths = defaultdict(list)

            def walk(t, nodes, edges):
                for other, faces in ports[t].items():
                    for face in faces:
                        paths[other].append((frozenset(nodes[1:]), edges + (face,)))
                for following, face in adjacency[t]:
                    if following not in nodes:
                        walk(following, nodes + (following,), edges + (face,))

            walk(root, (root,), ())

            def choose(k, used, selected):
                if k == 4:
                    choices.append(dict(root=root, paths=dict(zip(others, selected)), used=len(used)))
                    return
                for nodes, edges in paths[others[k]]:
                    if used & nodes or any(edges[0] == p[0] for p in selected):
                        continue
                    choose(k + 1, used | nodes, selected + [edges])

            choose(0, frozenset(), [])
        choices.sort(key=lambda choice: (choice["used"], choice["root"], repr(choice["paths"])),
                     reverse=reverse)
        options[patch] = choices
    solution = {}

    def solve(patch):
        if patch == 5:
            return True
        for choice in options[patch]:
            if any(choice["paths"][b][-1] != solution[b]["paths"][patch][-1]
                   for b in range(patch)):
                continue
            solution[patch] = choice
            if solve(patch + 1):
                return True
        solution.pop(patch, None)
        return False

    if not solve(0):
        raise ValueError("No routed K5 found within inherited boundary patches")
    routes = {}
    for a, b in combinations(range(5), 2):
        first, second = solution[a]["paths"][b], solution[b]["paths"][a]
        faces = first + second[-2::-1]
        nodes = [solution[a]["root"]]
        for face in faces:
            nodes.append(next(t for t in links[face] if t != nodes[-1]))
        assert nodes[-1] == solution[b]["root"]
        routes[a, b] = dict(faces=faces, nodes=tuple(nodes), carrier=first[-1])
    used_nodes = [t for route in routes.values() for t in route["nodes"][1:-1]]
    assert len(used_nodes) == len(set(used_nodes))
    assert not set(used_nodes) & {choice["root"] for choice in solution.values()}
    return dict(kind=kind, vertices=vertices, cells=cells, divisions=divisions, links=links,
                roots={a: choice["root"] for a, choice in solution.items()}, routes=routes,
                reversed_search=reverse)


def embedding(spec, maps, two_j):
    """Local coefficient matrices plus the path signs in the all-outgoing basis."""
    spins = {face: 0 for face in spec["links"]}
    for route in spec["routes"].values():
        for face in route["faces"]:
            spins[face] = two_j
    projections, open_nodes = {}, {}
    original = maps.basis((two_j,) * 4)
    for patch, root in spec["roots"].items():
        faces = tuple(combinations(root, 3))
        coarse_faces = sorted(tuple(a for a in range(5) if a not in (patch, other))
                              for other in range(5) if other != patch)
        correspondence = {spec["routes"][tuple(sorted((patch, other)))]["faces"][0 if patch < other else -1]:
                          tuple(a for a in range(5) if a not in (patch, other))
                          for other in range(5) if other != patch}
        reordered = original.transpose((0,) + tuple(coarse_faces.index(correspondence[f]) + 1 for f in faces))
        local = maps.basis(tuple(spins[f] for f in faces))
        matrix = local.reshape(len(local), -1) @ reordered.reshape(len(original), -1).T
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(len(original)), atol=3e-14, rtol=0)
        projections[root], open_nodes[root] = matrix, patch
    for t in boundary(spec["cells"]):
        if t not in projections:
            local = maps.basis(tuple(spins[f] for f in combinations(t, 3)))
            assert len(local) == 1
            projections[t] = np.ones(1)
    phase = 1.
    e = epsilon(two_j)
    for route in spec["routes"].values():
        effective = np.eye(two_j + 1)
        for k, face in enumerate(route["faces"]):
            a, b = route["nodes"][k:k + 2]
            effective = effective @ (e if a < b else e.T)
            if k + 1 < len(route["faces"]):
                faces = tuple(combinations(b, 3))
                local = maps.basis(tuple(spins[f] for f in faces))[0]
                order = [faces.index(face), faces.index(route["faces"][k + 1])]
                order += [i for i in range(4) if i not in order]
                bivalent = local.transpose(order).reshape(two_j + 1, two_j + 1)
                effective = effective @ (math.sqrt(two_j + 1) * bivalent)
        sign = float(np.sum(effective * e) / (two_j + 1))
        np.testing.assert_allclose(effective, sign * e, atol=3e-14, rtol=0)
        assert abs(abs(sign) - 1) < 3e-14
        phase *= round(sign)
    return dict(spins=spins, projections=projections, open_nodes=open_nodes, phase=phase)


def spin_network(links, spins, node_tensors, holonomies):
    """Independent magnetic-index evaluation with normalized spin-network edges."""
    operands, leg_labels = [], {}
    for t in sorted(node_tensors):
        faces = sorted(f for f, pair in links.items() if t in pair)
        labels = []
        for face in faces:
            leg_labels[t, face] = len(leg_labels)
            labels.append(leg_labels[t, face])
        operands.extend((node_tensors[t], labels))
    for face, (a, b) in links.items():
        matrix = representation(holonomies[face], spins[face]) @ epsilon(spins[face])
        operands.extend((math.sqrt(spins[face] + 1) * matrix,
                         [leg_labels[a, face], leg_labels[b, face]]))
    return complex(oe.contract(*operands, [], optimize="greedy"))


def check_embedding(spec, maps, two_j, samples=6):
    """Pointwise pullback check at random SU(2) holonomies and boundary states.

    Haar invariance makes this pullback isometric: every coarse edge contains a
    unique fine carrier. Node maps are orthogonal and inserted bivalent states
    have unit norm. No numerical Monte Carlo norm estimate is used.
    """
    data = embedding(spec, maps, two_j)
    coarse_links = boundary_links((tuple(range(5)),))
    coarse_spins = {f: two_j for f in coarse_links}
    rng = np.random.default_rng(1729 + two_j)
    errors = []
    area = math.sqrt(two_j * (two_j + 2)) / 2
    area_errors = []
    for axes in combinations(range(5), 3):
        pieces = [f for f in spec["links"] if all(spec["vertices"][v][a] == 0
                  for v in f for a in range(5) if a not in axes)]
        actual_area = sum(math.sqrt(data["spins"][f] * (data["spins"][f] + 2)) / 2 for f in pieces)
        area_errors.append(abs(actual_area - area))
        assert sum(data["spins"][f] > 0 for f in pieces) == 1
        np.testing.assert_allclose(actual_area, area, atol=2e-15, rtol=0)
    for sample in range(samples):
        holonomies = {}
        for face in spec["links"]:
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            a, b = complex(q[0], q[1]), complex(q[2], q[3])
            holonomies[face] = np.array([[a, b], [-b.conjugate(), a.conjugate()]])
        coarse_holonomies = {}
        for (a, b), route in spec["routes"].items():
            h = np.eye(2, dtype=complex)
            for k, face in enumerate(route["faces"]):
                g = holonomies[face]
                if route["nodes"][k] > route["nodes"][k + 1]:
                    g = g.conj().T
                h = h @ g
            # Coarse tetrahedra are lexically ordered in reverse patch order.
            coarse_holonomies[tuple(v for v in range(5) if v not in (a, b))] = h.conj().T
        states = rng.integers(two_j + 1, size=5)
        original = maps.basis((two_j,) * 4)
        coarse_nodes = {tuple(v for v in range(5) if v != a): original[states[a]] for a in range(5)}
        fine_nodes = {}
        for t, projection in data["projections"].items():
            coefficients = projection[:, states[data["open_nodes"][t]]] if t in data["open_nodes"] else projection
            basis = maps.basis(tuple(data["spins"][f] for f in combinations(t, 3)))
            fine_nodes[t] = np.tensordot(coefficients, basis, axes=1)
        actual = data["phase"] * spin_network(spec["links"], data["spins"], fine_nodes, holonomies)
        expected = spin_network(coarse_links, coarse_spins, coarse_nodes, coarse_holonomies)
        errors.append(abs(actual - expected))
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-11)
    return dict(random_holonomy_checks=samples, max_absolute_error=max(errors),
                fine_nonzero_links=sum(j > 0 for j in data["spins"].values()),
                fine_zero_links=sum(j == 0 for j in data["spins"].values()),
                coarse_basis_states=(two_j + 1)**5, phase=data["phase"],
                preserved_inherited_triangle_area=area,
                max_inherited_triangle_area_error=max(area_errors))
