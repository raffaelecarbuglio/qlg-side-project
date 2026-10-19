"""Full regulated 1->5 EPRL boundary amplitude at physical spin cutoffs K=1/2,1.

All 32 half-spin boundary coefficients and all admissible internal labels are
kept. The physical cutoff K and auxiliary shell cutoff L are independent.
"""

import argparse
from itertools import combinations, product
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import opt_einsum as oe

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.check_cutoffs import native_data
from benchmarks.check_pachner import difference
from benchmarks.fivecell_checks import (epsilon_walks, explicit_intertwiners,
                                         magnetic_vertex, matrix_phase)
from benchmarks.matched_boundary import strand_signs
from benchmarks.refinement import BasisMaps, LEGS, complex_layout, native_bf, verify_native_basis
from benchmarks.refinement_protocol import relative_round
from qlg.backend import NativeBackend, digest
from qlg.jobs import atomic_json, exclusive_lock, fingerprint, source_version
from qlg.spins import intertwiner_range
from qlg.topology import facets


def layout(five=True, ordering="ascending"):
    cells = relative_round((tuple(range(5)),)) if five else (tuple(range(5)),)
    if ordering == "descending":
        cells = tuple(tuple(reversed(c)) for c in cells)
    graph = complex_layout(tuple(range(6 if five else 5)), cells)
    for node in graph["nodes"]:
        for key in ("faces", "tetrahedra"):
            node[key] = tuple(tuple(sorted(s)) for s in node[key])
    return graph


def spin_assignments(graph, two_k):
    outside = {f: 1 for t in graph["boundary_tetrahedra"] for f in combinations(t, 3)}
    inside = sorted(graph["face_indices"])
    tetrahedra = [tuple(combinations(t, 3)) for t in facets(graph["cells"], 4)]
    rows = []
    for values in product(range(two_k + 1), repeat=len(inside)):
        spins = dict(outside)
        spins.update(zip(inside, values))
        if all(intertwiner_range(*(spins[f] for f in t)) for t in tetrahedra):
            rows.append(spins)
    if len(inside) == 10:
        assert len(rows) == {1: 16, 2: 721}[two_k]
    return rows


def canonical_vertex(maps, node, spins, raw, cache):
    labels = tuple(spins[f] for f in node["faces"])
    key = (labels, node["tetrahedra"])
    if key not in cache:
        vertex = raw[labels].transpose(4, 3, 2, 1, 0)
        for axis, legs in enumerate(LEGS):
            faces = tuple(node["faces"][e] for e in legs)
            rotation = maps.overlap(faces, spins)
            vertex = np.moveaxis(np.tensordot(rotation, vertex, axes=(1, axis)), 0, axis)
        cache[key] = vertex
    return cache[key]


def exact_orders(arrays, inputs, outputs, paths):
    operands = [x for a, ids in zip(arrays, inputs) for x in (a, ids)]
    key = (tuple(a.shape for a in arrays), tuple(tuple(i) for i in inputs), tuple(outputs))
    if key not in paths:
        choices = []
        for method in ("greedy", "optimal"):
            path, info = oe.contract_path(*operands, outputs, optimize=method)
            if info.largest_intermediate > 16_000_000:
                raise MemoryError("Exact contraction exceeds intermediate-entry cap")
            choices.append(path)
        paths[key] = choices
    return [oe.contract(*operands, outputs, optimize=p) for p in paths[key]]


def evaluate(maps, graph, rows, raw, paths, independent=False, bf=False):
    """Sum physical-spin assignments; contract all intertwiners and keep boundary open."""
    before = time.perf_counter()
    internal = sorted(graph["tetrahedron_indices"])
    outside = graph["boundary_tetrahedra"]
    ids = {t: i for i, t in enumerate(internal + list(outside))}
    inputs = [[ids[t] for t in n["tetrahedra"]] for n in graph["nodes"]]
    outputs = [ids[t] for t in outside]
    signs = strand_signs(graph)
    walks = epsilon_walks(graph) if independent else None
    totals = [np.zeros((2,) * 5, dtype=np.longdouble) for _ in range(2)]
    reference = np.zeros_like(totals[0])
    no_face_weight = np.zeros_like(reference)
    absolute = np.zeros_like(reference)
    shells, cache, magnetic_cache = {}, {}, {}
    worst_sector = worst_vertex = 0.
    for spins in rows:
        arrays = [canonical_vertex(maps, n, spins, raw, cache) for n in graph["nodes"]]
        values = exact_orders(arrays, inputs, outputs, paths)
        bound = math.prod(float(np.max(np.abs(a))) for a in arrays)
        if bound:
            worst_sector = max(worst_sector, float(np.max(np.abs(values[0] - values[1]))) / bound)
        phase = (-1.) ** sum(signs[f] * j for f, j in spins.items())
        weight = math.prod(spins[f] + 1 for f in graph["face_indices"])
        # Ten boundary triangles with j=1/2: product sqrt(2j+1) = 32.
        multiplier = 32. * phase * weight
        for total, value in zip(totals, values):
            total += value.astype(np.longdouble) * multiplier
        contribution = values[0].astype(np.longdouble) * multiplier
        absolute += np.abs(contribution)
        maximum = max((spins[f] for f in graph["face_indices"]), default=0)
        shells.setdefault(maximum, np.zeros_like(reference))
        shells[maximum] += contribution
        if independent:
            explicit_phase = matrix_phase(walks, spins)
            assert explicit_phase == phase
            alternate = arrays
            if bf:
                alternate = []
                for n, native in zip(graph["nodes"], arrays):
                    key = (tuple(spins[f] for f in n["faces"]), n["tetrahedra"])
                    if key not in magnetic_cache:
                        magnetic_cache[key] = magnetic_vertex(maps, n, spins)
                    vertex = magnetic_cache[key]
                    scale = max(float(np.max(np.abs(native))), np.finfo(float).tiny)
                    error = float(np.max(np.abs(vertex - native))) / scale
                    worst_vertex = max(worst_vertex, error)
                    np.testing.assert_allclose(vertex, native, atol=2e-14 * scale, rtol=2e-12)
                    alternate.append(vertex)
            explicit = explicit_intertwiners(alternate, inputs, outputs)
            reference += explicit.astype(np.longdouble) * (32. * explicit_phase * weight)
            no_face_weight += explicit.astype(np.longdouble) * (32. * explicit_phase)
    value = totals[0].astype(float)
    scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
    order_error = float(np.max(np.abs(totals[0] - totals[1]))) / scale
    if order_error > 1e-10:
        raise AssertionError(f"Contraction-order disagreement: {order_error}")
    record = dict(shape=list(value.shape), entries=value.ravel().tolist(),
        spin_assignments=len(rows), assignment_sha256=fingerprint([
            [s[f] for f in sorted(graph["face_indices"])] for s in rows]),
        max_absolute_coefficient=scale, elapsed_seconds=time.perf_counter() - before,
        aggregate_sector_cancellation=float(np.linalg.norm(value.ravel()) /
                                           np.linalg.norm(absolute.ravel())),
        spin_shell_contributions=[dict(max_two_spin=k, entries=v.astype(float).ravel().tolist())
                                  for k, v in sorted(shells.items())],
        contraction_order_relative_error=order_error,
        max_sector_order_difference_over_vertex_product=worst_sector)
    if independent:
        error = float(np.max(np.abs(totals[0] - reference))) / scale
        if error > 1e-10:
            raise AssertionError(f"Independent finite-sum disagreement: {error}")
        record["independent_check"] = dict(maxnorm_relative_error=error,
            magnetic_vertex_relative_error=worst_vertex if bf else None,
            epsilon_matrix_phases_checked=True,
            method="Explicit internal-intertwiner loops" + (" and magnetic BF vertices" if bf else ""),
            omitted_face_weights_negative_control=float(np.max(np.abs(totals[0] - no_face_weight))) / scale)
    return record


def summarize(report):
    get = {(r["dl"], r["two_k"], r["ordering"]): r for r in report["results"]}
    report["cutoff_changes"], report["ordering_changes"], report["spin_cutoff_changes"] = [], [], []
    for dl in sorted({k[0] for k in get}):
        for two_k in (1, 2):
            keys = [(dl, two_k, o) for o in ("ascending", "descending")]
            if all(k in get for k in keys):
                report["ordering_changes"].append(dict(dl=dl, two_k=two_k,
                    **difference(*(get[k] for k in keys))))
        for ordering in ("ascending", "descending"):
            keys = [(dl, k, ordering) for k in (2, 1)]
            if all(k in get for k in keys):
                report["spin_cutoff_changes"].append(dict(dl=dl, ordering=ordering,
                    **difference(*(get[k] for k in keys))))
    for two_k in (1, 2):
        for ordering in ("ascending", "descending"):
            selected = sorted((r for r in report["results"] if r["two_k"] == two_k and
                               r["ordering"] == ordering), key=lambda r: r["dl"])
            for a, b in zip(selected, selected[1:]):
                report["cutoff_changes"].append(dict(two_k=two_k, ordering=ordering,
                    earlier_dl=a["dl"], later_dl=b["dl"], **difference(b, a)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dl", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--two-k", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--orders", nargs="+", choices=("ascending", "descending"),
                        default=["ascending", "descending"])
    args = parser.parse_args()
    if any(dl < 0 or dl > 12 for dl in args.dl) or len(set(args.dl)) != len(args.dl):
        parser.error("Specify distinct auxiliary cutoffs in 0..12")
    threads = os.environ.get("OMP_NUM_THREADS", "")
    if (not threads.isdecimal() or int(threads) < 1 or
            os.environ.get("OPENBLAS_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS", "1") != "1"):
        parser.error("Require positive OMP_NUM_THREADS and single-threaded BLAS")
    os.environ["OMP_DYNAMIC"] = "FALSE"
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    graphs = {o: layout(ordering=o) for o in args.orders}
    coarse = layout(five=False)
    outside = coarse["boundary_tetrahedra"]
    assert all(g["boundary_tetrahedra"] == outside for g in graphs.values())
    all_rows = {k: spin_assignments(next(iter(graphs.values())), k) for k in args.two_k}
    root = Path(__file__).parents[1]
    report = dict(status="running", source=source_version(), gamma=1.2, accuracy="veryhigh",
        rho_convention=identity["backend"]["build"].get("rho_convention"),
        backend_source_revision=identity["backend"]["build"].get("commit"),
        source_files_sha256={n: digest(root / n) for n in ("benchmarks/check_fivecell.py",
            "benchmarks/fivecell_checks.py", "benchmarks/refinement.py", "benchmarks/matched_boundary.py",
            "benchmarks/block_scan.py", "qlg/backend.py", "qlg/spins.py")},
        omp_threads=int(threads), blas_threads=1, boundary_j=.5,
        boundary_tetrahedra=[list(t) for t in outside],
        internal_faces=[list(f) for f in sorted(next(iter(graphs.values()))["face_indices"])],
        measure="sqrt(2j+1) boundary, internal face 2j+1, unit edge weight",
        normalization="Absolute signed amplitudes, no fitted rescaling",
        scope="Full finite regulated sums only; no removal of K, no renormalization or continuum limit",
        bf_checks=[], native_preparation=[], coarse=[], results=[])
    with exclusive_lock(run / ".fivecell.lock"), tempfile.TemporaryDirectory(prefix="qlg-fivecell-") as tmp:
        atomic_json(args.output, report)
        data = Path(tmp)
        native_data(data)
        max_two_spin = 4 * (max(args.two_k) + 2 * max(args.dl)) + 8
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, max(32, max_two_spin), 128) as backend:
            maps, paths = BasisMaps(backend), {}
            report["native_basis_max_absolute_errors"] = verify_native_basis(backend, maps)
            for ordering, graph in graphs.items():
                rows = spin_assignments(graph, 1)
                labels = {tuple(s[f] for f in n["faces"]) for s in rows for n in graph["nodes"]}
                raw = {s: native_bf(backend, s) for s in labels}
                value = evaluate(maps, graph, rows, raw, paths, independent=True, bf=True)
                assert value["independent_check"]["omitted_face_weights_negative_control"] > .01
                report["bf_checks"].append(dict(two_k=1, ordering=ordering, **value))
                atomic_json(args.output, report)
                print(f"BF {ordering}: independent finite sum passed", flush=True)
            for dl in sorted(args.dl):
                raw = {}
                for two_k in sorted(set(args.two_k)):
                    rows = all_rows[two_k]
                    for ordering, graph in graphs.items():
                        needed = sorted({tuple(s[f] for f in n["faces"]) for s in rows for n in graph["nodes"]})
                        assert len(needed) == {1: 8, 2: 81}[two_k]
                        computed = reused = 0
                        seconds = 0.
                        for i, labels in enumerate(needed, 1):
                            if labels in raw:
                                continue
                            raw[labels], elapsed, hit = checkpoint(run, identity, labels, dl,
                                lambda: backend.vertex(labels, dl))
                            computed += not hit
                            reused += hit
                            seconds += elapsed if not hit else 0.
                            atomic_json(run / "progress.json", dict(dl=dl, two_k=two_k,
                                ordering=ordering, position=i, needed=len(needed),
                                computed=computed, reused=reused, native_seconds=seconds))
                            print(f"DL={dl} K={two_k/2:g} {ordering}: vertex {i}/{len(needed)}, "
                                  f"{'cached' if hit else 'new'} {elapsed:.3f}s", flush=True)
                        prep = dict(dl=dl, two_k=two_k, ordering=ordering, needed=len(needed),
                                    computed=computed, reused=reused, native_seconds=seconds)
                        report["native_preparation"].append(prep)
                        value = evaluate(maps, graph, rows, raw, paths, independent=two_k == 1)
                        report["results"].append(dict(dl=dl, two_k=two_k, ordering=ordering, **value))
                        summarize(report)
                        atomic_json(args.output, report)
                        print(f"Finished DL={dl} K={two_k/2:g} {ordering}: "
                              f"max|W5|={value['max_absolute_coefficient']:.9e}", flush=True)
                crows = spin_assignments(coarse, 0)
                report["coarse"].append(dict(dl=dl, **evaluate(maps, coarse, crows, raw, paths)))
                atomic_json(args.output, report)
    report["status"] = "completed"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
