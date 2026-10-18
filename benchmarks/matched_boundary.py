"""Full EPRL amplitudes pulled back to a common coarse boundary basis.

This is a diagnostic of the j=0 graph-subdivision vacuum. It does not supply
a nondegenerate semiclassical boundary or a dynamically selected embedding.
"""

import argparse
from collections import defaultdict
from itertools import combinations, product
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import opt_einsum as oe

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.boundary_embedding import (area_and_flux_check, check_embedding, coarse_boundary, embedding,
                                           routed_boundary)
from benchmarks.check_cutoffs import native_data
from benchmarks.refinement import BasisMaps, LEGS, complex_layout, native_bf, verify_native_basis
from qlg.backend import NativeBackend
from qlg.jobs import atomic_json, exclusive_lock, source_version
from qlg.spins import intertwiner_range, vertex_ranges
from qlg.topology import facets


def spin_domains(graph, boundary_spins):
    """Exact finite bounds for these balls; retain every allowed half-integer.

    Every internal triangle must have a tetrahedron whose other three faces
    are boundary faces. Angular momentum addition bounds it by their sum.
    Arc consistency removes only values with no admissible tetrahedral support.
    """
    internal = graph["face_indices"]
    tetrahedra = sorted(facets(graph["cells"], 4))
    domains = {f: (j,) for f, j in boundary_spins.items()}
    for face in internal:
        bounds = []
        for t in tetrahedra:
            faces = tuple(combinations(t, 3))
            if face in faces and all(f == face or f in boundary_spins for f in faces):
                bounds.append(sum(boundary_spins[f] for f in faces if f != face))
        if not bounds:
            raise ValueError("No proved finite spin bound; this evaluator cannot sum this complex")
        domains[face] = tuple(range(min(bounds) + 1))
    initial = {f: domains[f] for f in internal}
    changed = True
    while changed:
        changed = False
        for t in tetrahedra:
            faces = tuple(combinations(t, 3))
            supported = [set() for _ in faces]
            for labels in product(*(domains[f] for f in faces)):
                if intertwiner_range(*labels):
                    for support, j in zip(supported, labels):
                        support.add(j)
            for f, support in zip(faces, supported):
                reduced = tuple(j for j in domains[f] if j in support)
                if not reduced:
                    raise ValueError(f"Boundary has no admissible extension at {t}")
                if reduced != domains[f]:
                    domains[f], changed = reduced, True
    channels = {}
    for t in graph["tetrahedron_indices"]:
        channels[t] = tuple(sorted({i for labels in product(*(domains[f] for f in combinations(t, 3)))
                                     for i in intertwiner_range(*labels)}))
    return domains, channels, initial


def strand_signs(graph):
    """Compute epsilon-metric signs, including half-integer closed strands.

    Native wedges run from tetrahedron a to b for a<b. Insert one epsilon on
    each leg of a glued invariant (this leaves it invariant), oriented by cell
    index. Each triangle then forms a path or cycle. Compare its epsilon product
    with the canonical boundary epsilon or internal identity. E_j^2=(-1)^(2j)I.
    """
    strands = defaultdict(lambda: defaultdict(list))
    incidences = defaultdict(list)
    for cell_index, node in enumerate(graph["nodes"]):
        for t in node["tetrahedra"]:
            incidences[t].append(cell_index)
        for a, b in combinations(range(5), 2):
            face = tuple(sorted(set(node["tetrahedra"][a]) & set(node["tetrahedra"][b])))
            x, y = (cell_index, node["tetrahedra"][a]), (cell_index, node["tetrahedra"][b])
            strands[face][x].append((y, False))
            strands[face][y].append((x, True))
    for t, cells in incidences.items():
        if len(cells) != 2:
            continue
        x, y = (min(cells), t), (max(cells), t)
        for face in combinations(t, 3):
            strands[face][x].append((y, False))
            strands[face][y].append((x, True))
    exponents = {}
    for face, adjacency in strands.items():
        ends = sorted((x for x in adjacency if len(adjacency[x]) == 1), key=lambda x: x[1])
        assert len(ends) in (0, 2)
        start = ends[0] if ends else min(adjacency)
        current, previous, edges, reverse = start, None, 0, 0
        while True:
            following, backwards = next((x, r) for x, r in adjacency[current] if x != previous)
            edges += 1
            reverse += backwards
            previous, current = current, following
            if current == start or (ends and current == ends[1]):
                break
        assert edges == len(adjacency) - bool(ends)
        exponents[face] = (edges // 2 + reverse) % 2
    return exponents


def contract_open(arrays, inputs, outputs):
    """Two independent exact contraction orders, retaining all coarse states."""
    operands = [item for pair in zip(arrays, inputs) for item in pair]
    values, records = [], []
    for method in ("greedy", "random-greedy-128"):
        path, info = oe.contract_path(*operands, outputs, optimize=method)
        if info.largest_intermediate > 16_000_000:
            raise MemoryError("Exact intermediate-entry cap exceeded")
        values.append(oe.contract(*operands, outputs, optimize=path))
        records.append(dict(method=method, largest_intermediate_entries=int(info.largest_intermediate)))
    scale = max(float(np.max(np.abs(values[0]))), np.finfo(float).tiny)
    error = float(np.max(np.abs(values[1] - values[0])) / scale)
    np.testing.assert_allclose(values[1], values[0], atol=2e-12 * scale, rtol=2e-11)
    return values[0], dict(paths=records, maxnorm_relative_disagreement=error)


def amplitude(backend, maps, spec, two_j, dl, run, identity, bf=False):
    """All bounded internal spins and all internal intertwiners; open coarse i."""
    data = embedding(spec, maps, two_j)
    graph = complex_layout(spec["vertices"], spec["cells"])
    domains, channels, initial = spin_domains(graph, data["spins"])
    signs = strand_signs(graph)
    fi, ti = graph["face_indices"], graph["tetrahedron_indices"]
    output_ids = list(range(len(fi) + len(ti), len(fi) + len(ti) + 5))
    arrays, inputs, cache = [], [], {}
    computed = reused = 0
    before = time.perf_counter()
    for node in graph["nodes"]:
        spin_faces = sorted((f for f in node["faces"] if f in fi), key=fi.get)
        kept = [(a, t) for a, t in enumerate(node["tetrahedra"])
                if t in ti or t in data["open_nodes"]]
        shape = [len(domains[f]) for f in spin_faces]
        shape += [len(channels[t]) if t in ti else two_j + 1 for _, t in kept]
        tensor = np.zeros(shape)
        for positions in product(*(range(len(domains[f])) for f in spin_faces)):
            spin_map = dict(data["spins"])
            spin_map.update({f: domains[f][p] for f, p in zip(spin_faces, positions)})
            spins = tuple(spin_map[f] for f in node["faces"])
            if not all(vertex_ranges(spins)):
                continue
            if spins not in cache:
                if bf:
                    cache[spins] = native_bf(backend, spins)
                    computed += 1
                else:
                    value, _, hit = checkpoint(run, identity, spins, dl,
                                               lambda: backend.vertex(spins, dl))
                    cache[spins] = value
                    computed += not hit
                    reused += hit
            vertex = cache[spins].transpose(4, 3, 2, 1, 0)
            common_ranges = []
            for axis, legs in enumerate(LEGS):
                faces = tuple(node["faces"][e] for e in legs)
                matrix = maps.overlap(faces, spin_map)
                vertex = np.moveaxis(np.tensordot(matrix, vertex, axes=(1, axis)), 0, axis)
                common_ranges.append(intertwiner_range(*(spin_map[f] for f in sorted(faces))))
            # Remove fixed fine boundary states, map roots to the common coarse basis.
            for axis in reversed(range(5)):
                t = node["tetrahedra"][axis]
                if t in ti:
                    continue
                projection = data["projections"][t]
                if t in data["open_nodes"]:
                    vertex = np.moveaxis(np.tensordot(vertex, projection, axes=(axis, 0)), -1, axis)
                else:
                    vertex = np.tensordot(vertex, projection, axes=(axis, 0))
            indices = np.ix_(*[[channels[t].index(i) for i in common_ranges[a]] if t in ti
                              else list(range(two_j + 1)) for a, t in kept])
            tensor[positions][indices] = vertex
        arrays.append(tensor)
        inputs.append([fi[f] for f in spin_faces] + [ti[t] if t in ti else
                      output_ids[data["open_nodes"][t]] for _, t in kept])
    for face, index in fi.items():
        js = np.array(domains[face])
        arrays.append((js + 1.) * (-1.) ** (signs[face] * js))
        inputs.append([index])
    boundary_phase = data["phase"] * (-1.) ** sum(signs[f] * j for f, j in data["spins"].items())
    # Explicit convention: sqrt(2j+1) per boundary link; unit edge measure.
    log_scale = .5 * math.fsum(math.log(j + 1) for j in data["spins"].values())
    for k, array in enumerate(arrays):
        scale = float(np.max(np.abs(array)))
        if scale:
            arrays[k] = array / scale
            log_scale += math.log(scale)
    result, check = contract_open(arrays, inputs, output_ids)
    result *= boundary_phase
    return result, log_scale, dict(
        cells=len(spec["cells"]), internal_faces=len(fi), internal_tetrahedra=len(ti),
        doubled_spin_domains=[dict(face=list(f), initial=list(initial[f]), retained=list(domains[f])) for f in fi],
        half_integer_internal_values=sum(j % 2 for f in fi for j in domains[f]),
        unique_native_vertices=len(cache), computed=computed, reused=reused,
        elapsed_seconds=time.perf_counter() - before, contraction=check)


def serialized(tensor, scale):
    maximum = float(np.max(np.abs(tensor)))
    if not maximum:
        raise FloatingPointError("Boundary functional vanished")
    return dict(shape=list(tensor.shape), entries=(tensor / maximum).ravel().tolist(),
                log_absolute_scale=scale + math.log(maximum))


def comparison(fine, fine_scale, coarse, coarse_scale):
    f, c = fine.ravel(), coarse.ravel()
    overlap = float(np.dot(f, c) / (np.linalg.norm(f) * np.linalg.norm(c)))
    residual = float(np.linalg.norm(f - np.dot(f, c) / np.dot(c, c) * c) / np.linalg.norm(f))
    return dict(normalized_overlap=overlap, relative_residual_after_one_global_scale=residual,
                log_norm_ratio=fine_scale - coarse_scale + math.log(np.linalg.norm(f) / np.linalg.norm(c)))


def compare_saved(left, right):
    a, b = left["fine"], right["fine"]
    return comparison(np.array(a["entries"]), a["log_absolute_scale"],
                      np.array(b["entries"]), b["log_absolute_scale"])


def summarize(report):
    report["cutoff_changes"], report["routing_comparisons"] = [], []
    groups = defaultdict(list)
    for row in report["results"]:
        groups[row["two_j"], row["kind"], row["reversed_search"]].append(row)
    for (two_j, kind, reverse), rows in groups.items():
        rows.sort(key=lambda row: row["dl"])
        for earlier, later in zip(rows, rows[1:]):
            report["cutoff_changes"].append(dict(two_j=two_j, kind=kind, reversed_search=reverse,
                earlier_dl=earlier["dl"], later_dl=later["dl"], **compare_saved(later, earlier)))
        if reverse:
            other = {row["dl"]: row for row in groups[two_j, kind, False]}
            for row in rows:
                report["routing_comparisons"].append(dict(two_j=two_j, kind=kind, dl=row["dl"],
                                                         **compare_saved(row, other[row["dl"]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dl", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--two-j", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--kinds", nargs="+", choices=("split", "edgewise"), default=["split", "edgewise"])
    parser.add_argument("--skip-bf", action="store_true")
    args = parser.parse_args()
    if any(dl < 0 or dl > 12 for dl in args.dl):
        parser.error("Shell cutoffs must be in 0..12")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("Set OPENBLAS_NUM_THREADS=1")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    report = dict(status="running", source=source_version(), gamma=1.2, accuracy="veryhigh",
                  rho_convention=identity["backend"]["build"].get("rho_convention"),
                  backend_source_revision=identity["backend"]["build"].get("commit"),
                  omp_threads=int(os.environ.get("OMP_NUM_THREADS", "1")), blas_threads=1,
                  boundary_measure="sqrt(2j+1) per boundary link; unit edge measure",
                  interpretation="Kinematical j=0 embedding diagnostic, not a continuum result",
                  area_and_flux_diagnostic=area_and_flux_check(),
                  embeddings=[], embedding_checks=[], bf_checks=[], results=[])
    with exclusive_lock(run / ".matched.lock"):
        # Individual atomic vertices contain the strict local cache identity.
        with tempfile.TemporaryDirectory(prefix="qlg-matched-native-") as temporary:
            native = Path(temporary)
            native_data(native)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], native, 1.2, 2, 128, 128) as backend:
                maps = BasisMaps(backend)
                report["native_basis_max_absolute_errors"] = verify_native_basis(backend, maps)
                for two_j in args.two_j:
                    coarse = coarse_boundary()
                    cases = [(kind, reverse, routed_boundary(kind, reverse)) for kind in args.kinds
                             for reverse in (False, True)]
                    if two_j == args.two_j[0]:
                        report["embeddings"] = [dict(kind=kind, reversed_search=reverse,
                            roots=[dict(coarse_node=a, fine_tetrahedron=list(t)) for a, t in spec["roots"].items()],
                            routes=[dict(coarse_nodes=list(pair), faces=[list(f) for f in route["faces"]],
                                         nodes=[list(t) for t in route["nodes"]])
                                    for pair, route in spec["routes"].items()]) for kind, reverse, spec in cases]
                    for kind, reverse, spec in cases:
                        checks = check_embedding(spec, maps, two_j)
                        report["embedding_checks"].append(dict(kind=kind, reversed_search=reverse,
                                                               two_j=two_j, **checks))
                    for dl in ([None] if not args.skip_bf else []) + args.dl:
                        bf = dl is None
                        reference, reference_scale, _ = amplitude(
                            backend, maps, coarse, two_j, dl, run, identity, bf)
                        for kind, reverse, spec in cases:
                            value, scale, info = amplitude(backend, maps, spec, two_j, dl, run, identity, bf)
                            measures = comparison(value, scale, reference, reference_scale)
                            row = dict(kind=kind, reversed_search=reverse, two_j=two_j, dl=dl,
                                       comparison=measures, preparation=info,
                                       coarse=serialized(reference, reference_scale),
                                       fine=serialized(value, scale))
                            if bf:
                                # BF is topological here; failure blocks all EPRL interpretation.
                                np.testing.assert_allclose(value * math.exp(scale - reference_scale),
                                                           reference, rtol=2e-10,
                                                           atol=2e-12 * np.max(np.abs(reference)))
                                report["bf_checks"].append(row)
                            else:
                                report["results"].append(row)
                            atomic_json(args.output, report)
                            print(f"j={two_j/2:g} {'BF' if bf else 'DL='+str(dl)} {kind} "
                                  f"route={int(reverse)} residual={measures['relative_residual_after_one_global_scale']:.6g}",
                                  flush=True)
    summarize(report)
    report["status"] = "completed"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
