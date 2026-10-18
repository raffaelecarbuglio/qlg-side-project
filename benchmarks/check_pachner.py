"""Full common-boundary 3<->3 diagnostic, not a continuum-limit test.

All physical internal labels are summed. Only the infinite auxiliary-spin
sum is truncated. Two local vertex orderings diagnose finite-shell effects.
"""

import argparse
from itertools import combinations
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.check_cutoffs import native_data
from benchmarks.matched_boundary import contract_open, spin_domains, strand_signs
from benchmarks.refinement import BasisMaps, LEGS, complex_layout, native_bf, verify_native_basis
from qlg.backend import NativeBackend, digest
from qlg.jobs import atomic_json, exclusive_lock, source_version
from qlg.spins import intertwiner_range


CELLS = {
    "left": ((0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (0, 1, 2, 4, 5)),
    "right": ((0, 1, 3, 4, 5), (0, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
}


def layout(side, ordering):
    cells = CELLS[side]
    if ordering == "descending":
        cells = tuple(tuple(reversed(c)) for c in cells)
    graph = complex_layout(tuple(range(6)), cells)
    # Local order changes, but shared simplex names remain canonical.
    for node in graph["nodes"]:
        for key in ("faces", "tetrahedra"):
            node[key] = tuple(tuple(sorted(s)) for s in node[key])
    assert len(graph["face_indices"]) == 1
    assert len(graph["tetrahedron_indices"]) == 3
    assert len(graph["boundary_tetrahedra"]) == 9
    return graph


def evaluate(backend, maps, graph, two_j, dl, run, identity, cache):
    """Return every coefficient in the same nine-tetrahedron boundary basis."""
    before = time.perf_counter()
    outside = graph["boundary_tetrahedra"]
    spins = {f: two_j for t in outside for f in combinations(t, 3)}
    assert len(spins) == 18
    domains, _, _ = spin_domains(graph, spins)
    face = next(iter(graph["face_indices"]))
    expected = tuple(range(two_j % 2, 3 * two_j + 1, 2))
    assert domains[face] == expected  # Exact angular-momentum bound, not a chosen cutoff.
    internal = sorted(graph["tetrahedron_indices"])
    ids = {t: k for k, t in enumerate(internal + list(outside))}
    signs = strand_signs(graph)
    boundary_weight = math.prod(math.sqrt(j + 1) for j in spins.values())
    parts, records = [], []
    computed = reused = 0
    for two_x in domains[face]:
        assigned = dict(spins)
        assigned[face] = two_x
        arrays, inputs = [], []
        for node in graph["nodes"]:
            labels = tuple(assigned[f] for f in node["faces"])
            key = (dl, labels)
            if key not in cache:
                if dl is None:
                    cache[key] = native_bf(backend, labels)
                    computed += 1
                else:
                    cache[key], _, hit = checkpoint(
                        run, identity, labels, dl, lambda: backend.vertex(labels, dl))
                    computed += not hit
                    reused += hit
            vertex = cache[key].transpose(4, 3, 2, 1, 0)
            for axis, legs in enumerate(LEGS):
                faces = tuple(node["faces"][e] for e in legs)
                rotation = maps.overlap(faces, assigned)
                vertex = np.moveaxis(np.tensordot(rotation, vertex, axes=(1, axis)), 0, axis)
            arrays.append(vertex)
            inputs.append([ids[t] for t in node["tetrahedra"]])
        value, check = contract_open(arrays, inputs, list(range(3, 12)))
        value *= boundary_weight * (two_x + 1) * (-1.) ** sum(
            signs[f] * assigned[f] for f in assigned)
        parts.append(value)
        records.append(dict(two_x=two_x, entries=value.ravel().tolist(),
                            internal_intertwiner_dimensions=[len(intertwiner_range(
                                *(assigned[f] for f in combinations(t, 3)))) for t in internal],
                            contraction=check))
    total = np.sum(np.array(parts, dtype=np.longdouble), axis=0).astype(float)
    absolute = np.sum(np.abs(parts), axis=0)
    return dict(shape=list(total.shape), entries=total.ravel().tolist(), sectors=records,
                internal_face=list(face), doubled_spin_domain=list(domains[face]),
                computed_vertices=computed, reused_vertices=reused,
                elapsed_seconds=time.perf_counter() - before,
                max_absolute_coefficient=float(np.max(np.abs(total))),
                aggregate_sector_cancellation=float(np.linalg.norm(total.ravel()) /
                                                    np.linalg.norm(absolute.ravel())))


def difference(left, right):
    """Keep absolute differences; the shape statistic does not redefine W."""
    a, b = np.array(left["entries"]), np.array(right["entries"])
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return dict(max_absolute_difference=float(np.max(np.abs(a - b))),
                symmetric_maxnorm_relative_difference=float(np.max(np.abs(a - b)) / scale),
                l2_relative_difference=float(np.linalg.norm(a - b) / max(norm_a, norm_b)),
                normalized_vector_distance=float(np.linalg.norm(a / norm_a - b / norm_b)),
                l2_norm_ratio=float(norm_a / norm_b))


def summarize(report):
    rows = report["results"]
    report["triangulation_comparisons"] = []
    report["ordering_comparisons"] = []
    report["cutoff_changes"] = []
    for two_j in sorted({r["two_j"] for r in rows}):
        selected = [r for r in rows if r["two_j"] == two_j]
        get = {(r["dl"], r["side"], r["ordering"]): r for r in selected}
        cutoffs = sorted({r["dl"] for r in selected})
        for dl in cutoffs:
            for ordering in ("ascending", "descending"):
                keys = [(dl, side, ordering) for side in CELLS]
                if all(k in get for k in keys):
                    report["triangulation_comparisons"].append(dict(two_j=two_j, dl=dl,
                        ordering=ordering, **difference(*(get[k] for k in keys))))
            for side in CELLS:
                keys = [(dl, side, order) for order in ("ascending", "descending")]
                if all(k in get for k in keys):
                    report["ordering_comparisons"].append(dict(two_j=two_j, dl=dl,
                        side=side, **difference(*(get[k] for k in keys))))
        for earlier, later in zip(cutoffs, cutoffs[1:]):
            for side in CELLS:
                for ordering in ("ascending", "descending"):
                    keys = [(later, side, ordering), (earlier, side, ordering)]
                    if all(k in get for k in keys):
                        report["cutoff_changes"].append(dict(two_j=two_j, side=side,
                            ordering=ordering, earlier_dl=earlier, later_dl=later,
                            **difference(*(get[k] for k in keys))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dl", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--two-j", type=int, nargs="+", choices=(1, 2), default=[1])
    parser.add_argument("--bf-only", action="store_true")
    args = parser.parse_args()
    if any(dl < 0 or dl > 12 for dl in args.dl) or len(set(args.dl)) != len(args.dl):
        parser.error("Specify distinct shell cutoffs in 0..12")
    threads = os.environ.get("OMP_NUM_THREADS", "")
    if (os.environ.get("OPENBLAS_NUM_THREADS") != "1" or
            os.environ.get("MKL_NUM_THREADS", "1") != "1" or
            not threads.isdecimal() or int(threads) < 1):
        parser.error("Require positive OMP_NUM_THREADS and single-threaded BLAS")
    os.environ["OMP_DYNAMIC"] = "FALSE"
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    graphs = {(side, order): layout(side, order) for side in CELLS
              for order in ("ascending", "descending")}
    outside = graphs["left", "ascending"]["boundary_tetrahedra"]
    assert all(g["boundary_tetrahedra"] == outside for g in graphs.values())
    root = Path(__file__).parents[1]
    report = dict(status="running", source=source_version(),
        source_files_sha256={name: digest(root / name) for name in (
            "benchmarks/check_pachner.py", "benchmarks/refinement.py",
            "benchmarks/matched_boundary.py", "qlg/backend.py", "qlg/spins.py")},
        gamma=1.2, accuracy="veryhigh", omp_threads=int(threads), blas_threads=1,
        rho_convention=identity["backend"]["build"].get("rho_convention"),
        backend_source_revision=identity["backend"]["build"].get("commit"),
        cells={side: [list(c) for c in cells] for side, cells in CELLS.items()},
        boundary_tetrahedra=[list(t) for t in outside],
        boundary_basis="Sorted triangle legs; normalized canonical SU(2) 4jm at each tetrahedron",
        measure="sqrt(2j+1) per boundary triangle, internal face 2j+1, unit edge weight",
        normalization="Absolute signed amplitudes; no fitted scalar or per-state rescaling",
        scope="Finite 3<->3 diagnostic, neither a refinement sequence nor a continuum limit",
        numerical_target="1% last-cutoff and ordering changes, diagnostic only; no tail bound",
        bf_checks=[], results=[])
    with exclusive_lock(run / ".pachner.lock"), tempfile.TemporaryDirectory(prefix="qlg-pachner-") as tmp:
        atomic_json(args.output, report)
        data = Path(tmp)
        native_data(data)
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 128, 128) as backend:
            maps, cache = BasisMaps(backend), {}
            report["native_basis_max_absolute_errors"] = verify_native_basis(backend, maps)
            for two_j in sorted(set(args.two_j)):
                bf_rows = []
                for (side, ordering), graph in graphs.items():
                    value = evaluate(backend, maps, graph, two_j, None, run, identity, cache)
                    bf_rows.append(dict(side=side, ordering=ordering, **value))
                reference = bf_rows[0]
                errors = [difference(r, reference) for r in bf_rows]
                assert max(r["symmetric_maxnorm_relative_difference"] for r in errors) < 1e-11
                # Removing a real internal-spin sector must be detected by the BF identity.
                omitted = dict(reference, entries=(np.array(reference["entries"]) -
                    np.array(reference["sectors"][-1]["entries"])).tolist())
                negative = difference(omitted, bf_rows[2])
                assert negative["symmetric_maxnorm_relative_difference"] > .01
                report["bf_checks"].append(dict(two_j=two_j, comparisons=errors,
                    omitted_highest_spin_negative_control=negative, amplitudes=bf_rows))
                atomic_json(args.output, report)
                print(f"j={two_j/2:g}: BF identity and missing-sector control passed", flush=True)
                if args.bf_only:
                    continue
                for dl in sorted(args.dl):
                    for (side, ordering), graph in graphs.items():
                        value = evaluate(backend, maps, graph, two_j, dl, run, identity, cache)
                        report["results"].append(dict(two_j=two_j, dl=dl, side=side,
                                                       ordering=ordering, **value))
                        summarize(report)
                        atomic_json(args.output, report)
                        print(f"j={two_j/2:g} DL={dl} {side} {ordering}: "
                              f"max|W|={value['max_absolute_coefficient']:.9e}, "
                              f"{value['elapsed_seconds']:.3f}s", flush=True)
    report["status"] = "completed"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
