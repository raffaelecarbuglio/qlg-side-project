"""Small, unrescaled EPRL boundary-pairing experiment on 1,2,4,6 cells.

Validate a specified family of boundary maps before comparing amplitudes.
No state-dependent or level-dependent normalization is fitted to the data.
"""

import argparse
from itertools import combinations
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import opt_einsum as oe

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.boundary_embedding import check_embedding, embedding, epsilon
from benchmarks.check_cutoffs import native_data
from benchmarks.matched_boundary import amplitude, serialized, strand_signs
from benchmarks.refinement import BasisMaps, LEGS, complex_layout, native_bf, verify_native_basis
from benchmarks.subdivision_maps import bisect, check_composition, coordinate_signature, family, initial
from qlg.backend import NativeBackend
from qlg.jobs import atomic_json, exclusive_lock, source_version


def gram_check(spec, maps, two_j):
    data = embedding(spec, maps, two_j)
    gram = np.ones((1, 1))
    for patch in range(5):
        matrix = data["projections"][spec["roots"][patch]]
        gram = np.kron(gram, matrix.T @ matrix)
    error = float(np.max(np.abs(gram - np.eye((two_j + 1)**5))))
    np.testing.assert_allclose(gram, np.eye(len(gram)), atol=1e-13, rtol=0)
    return dict(coarse_sector_dimension=len(gram), max_gram_error=error,
                method="All root-basis coefficients; normalized bivalent states and SU(2) orthogonality")


def check_disjoint_histories():
    checks = []
    for first, second in combinations(combinations(range(5), 2), 2):
        if set(first) & set(second):
            continue
        left, _ = bisect(initial(), first)
        left, _ = bisect(left, second)
        right, _ = bisect(initial(), second)
        right, _ = bisect(right, first)
        same = coordinate_signature(left) == coordinate_signature(right)
        if not same:
            raise AssertionError(f"Disjoint histories differ: {first}, {second}")
        checks.append(dict(first=list(first), second=list(second), same_state=True))
    return checks


def glue_two_cells(backend, maps, spec, two_j, dl, run, identity, bf=False):
    """Independent normalized two-vertex gluing, including a missing-factor control.

    Each single-cell amplitude carries sqrt(d_j) on its boundary links.
    Joining across a tetrahedron therefore needs d_j**(-1/2) on each of
    its four triangles. Epsilon matrices determine the orientation signs.
    This is a finite tensor gluing check, not a continuum cylinder test.
    """
    data = embedding(spec, maps, two_j)
    graph = complex_layout(spec["vertices"], spec["cells"])
    if len(graph["nodes"]) != 2 or len(graph["tetrahedron_indices"]) != 1 or graph["face_indices"]:
        raise ValueError("Need two cells meeting in one tetrahedron, without internal triangles")
    shared = next(iter(graph["tetrahedron_indices"]))
    faces = tuple(combinations(shared, 3))
    basis = maps.basis(tuple(data["spins"][f] for f in faces))
    transformed = basis.copy()
    for axis, face in enumerate(faces, 1):
        transformed = np.moveaxis(np.tensordot(epsilon(data["spins"][face]), transformed,
                                              axes=(1, axis)), 0, axis)
    metric = basis.reshape(len(basis), -1) @ transformed.reshape(len(basis), -1).T
    np.testing.assert_allclose(metric, np.eye(len(basis)), atol=3e-14, rtol=0)
    operands = [metric, [0, 1]]
    normalization, orientation = 1., 1.
    for face in faces:
        external = [next(t for t in node["tetrahedra"] if t != shared and set(face) <= set(t))
                    for node in graph["nodes"]]
        e = epsilon(data["spins"][face])
        chain = (e if external[0] < shared else e.T) @ e @ (e if shared < external[1] else e.T)
        target = e if external[0] < external[1] else e.T
        sign = float(np.sum(chain * target) / len(e))
        np.testing.assert_allclose(chain, sign * target, atol=3e-14, rtol=0)
        orientation *= round(sign)
        normalization /= math.sqrt(data["spins"][face] + 1)
    for cell_index, node in enumerate(graph["nodes"]):
        spins = tuple(data["spins"][f] for f in node["faces"])
        vertex = native_bf(backend, spins) if bf else checkpoint(
            run, identity, spins, dl, lambda: backend.vertex(spins, dl))[0]
        vertex = vertex.transpose(4, 3, 2, 1, 0)
        for axis, legs in enumerate(LEGS):
            ordered_faces = tuple(node["faces"][k] for k in legs)
            rotation = maps.overlap(ordered_faces, data["spins"])
            vertex = np.moveaxis(np.tensordot(rotation, vertex, axes=(1, axis)), 0, axis)
        single = complex_layout(spec["vertices"], (spec["cells"][cell_index],))
        signs = strand_signs(single)
        vertex *= (-1.)**sum(signs[f] * data["spins"][f] for f in signs)
        vertex *= math.prod(math.sqrt(data["spins"][f] + 1) for f in node["faces"])
        kept = []
        for axis in reversed(range(5)):
            t = node["tetrahedra"][axis]
            if t == shared:
                continue
            projection = data["projections"][t]
            if t in data["open_nodes"]:
                vertex = np.moveaxis(np.tensordot(vertex, projection, axes=(axis, 0)), -1, axis)
            else:
                vertex = np.tensordot(vertex, projection, axes=(axis, 0))
        for t in node["tetrahedra"]:
            if t == shared:
                kept.append(cell_index)
            elif t in data["open_nodes"]:
                kept.append(2 + data["open_nodes"][t])
        operands.extend((vertex, kept))
    raw = oe.contract(*operands, list(range(2, 7)), optimize="greedy") * data["phase"] * orientation
    glued = raw * normalization
    direct, scale, _ = amplitude(backend, maps, spec, two_j, dl, run, identity, bf)
    direct = direct * math.exp(scale)
    magnitude = float(np.max(np.abs(direct)))
    error = float(np.max(np.abs(glued - direct)) / magnitude)
    np.testing.assert_allclose(glued, direct, atol=2e-12 * magnitude, rtol=2e-11)
    missing = float(np.max(np.abs(raw - direct)) / magnitude)
    assert missing > .1  # The check must detect a physically relevant normalization error.
    return dict(two_j=two_j, dl=dl, model="BF" if bf else "EPRL",
                shared_intertwiners=len(basis), interface_factor=normalization,
                orientation_sign=orientation, maxnorm_relative_error=error,
                missing_interface_factor_relative_error=missing,
                metric_max_absolute_error=float(np.max(np.abs(metric - np.eye(len(basis))))))


def measures(left, right):
    """Absolute scale and shape are both retained; no best-fit scalar."""
    a, b = np.array(left["entries"]), np.array(right["entries"])
    difference = left["log_absolute_scale"] - right["log_absolute_scale"]
    common = max(left["log_absolute_scale"], right["log_absolute_scale"])
    delta = a * math.exp(left["log_absolute_scale"] - common) - b * math.exp(right["log_absolute_scale"] - common)
    maximum = float(np.max(np.abs(delta)))
    return dict(log_max_amplitude_ratio=difference,
                maxnorm_relative_change=float(np.max(np.abs(a * math.exp(difference) - b))),
                log_max_absolute_change=common + math.log(maximum) if maximum else None,
                normalized_overlap=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def summarize(report):
    report["cutoff_changes"], report["refinement_changes"] = [], []
    rows = report["results"]
    for two_j in sorted({r["two_j"] for r in rows}):
        for level in sorted({r["level"] for r in rows if r["two_j"] == two_j}):
            selected = sorted((r for r in rows if r["two_j"] == two_j and r["level"] == level), key=lambda r: r["dl"])
            for first, last in zip(selected, selected[1:]):
                report["cutoff_changes"].append(dict(two_j=two_j, level=level, cells=last["cells"],
                    earlier_dl=first["dl"], later_dl=last["dl"], **measures(last["amplitude"], first["amplitude"])))
        for dl in sorted({r["dl"] for r in rows if r["two_j"] == two_j}):
            selected = sorted((r for r in rows if r["two_j"] == two_j and r["dl"] == dl), key=lambda r: r["level"])
            for first, last in zip(selected, selected[1:]):
                report["refinement_changes"].append(dict(two_j=two_j, dl=dl,
                    earlier_cells=first["cells"], later_cells=last["cells"],
                    **measures(last["amplitude"], first["amplitude"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dl", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--two-j", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    args = parser.parse_args()
    if any(dl < 0 or dl > 12 for dl in args.dl):
        parser.error("Shell cutoffs must be in 0..12")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1" or not os.environ.get("OMP_NUM_THREADS", "").isdecimal():
        parser.error("Set OPENBLAS_NUM_THREADS=1 and a positive OMP_NUM_THREADS")
    if int(os.environ["OMP_NUM_THREADS"]) < 1 or os.environ.get("MKL_NUM_THREADS", "1") != "1":
        parser.error("Require positive OMP_NUM_THREADS and MKL_NUM_THREADS=1")
    os.environ["OMP_DYNAMIC"] = "FALSE"
    levels, _, _ = family()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    report = dict(status="running", source=source_version(), gamma=1.2, accuracy="veryhigh",
                  rho_convention=identity["backend"]["build"].get("rho_convention"),
                  backend_source_revision=identity["backend"]["build"].get("commit"),
                  omp_threads=int(os.environ["OMP_NUM_THREADS"]), blas_threads=1,
                  boundary_measure="sqrt(2j+1) per boundary link; unit edge weight; internal face 2j+1",
                  normalization="No fitted rescaling, per-state or per-level",
                  composition=check_composition(), disjoint_histories=check_disjoint_histories(),
                  boundary_checks=[], gluing_checks=[], bf_checks=[], results=[],
                  scope="Finite tests on specified histories, not a continuum limit or a full directed family",
                  levels=[dict(level=k, cells=[list(c) for c in s["cells"]],
                               vertices=[list(v) for v in s["vertices"]],
                               history=[list(e) for e in s["history"]],
                               routes=[dict(coarse_nodes=list(p), nodes=[list(t) for t in r["nodes"]],
                                            faces=[list(f) for f in r["faces"]]) for p, r in s["routes"].items()])
                          for k, s in enumerate(levels)])
    with exclusive_lock(run / ".subdivision.lock"), tempfile.TemporaryDirectory(prefix="qlg-subdivision-") as temporary:
        native = Path(temporary)
        native_data(native)
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], native, 1.2, 2, 128, 128) as backend:
            maps = BasisMaps(backend)
            report["native_basis_max_absolute_errors"] = verify_native_basis(backend, maps)
            for two_j in args.two_j:
                for level, spec in enumerate(levels):
                    report["boundary_checks"].append(dict(two_j=two_j, level=level,
                        **check_embedding(spec, maps, two_j), **gram_check(spec, maps, two_j)))
                coarse, coarse_scale, _ = amplitude(backend, maps, levels[0], two_j, None, run, identity, bf=True)
                for level, spec in enumerate(levels[1:], 1):
                    value, scale, info = amplitude(backend, maps, spec, two_j, None, run, identity, bf=True)
                    error = float(np.max(np.abs(value * math.exp(scale - coarse_scale) - coarse)) / np.max(np.abs(coarse)))
                    np.testing.assert_allclose(value * math.exp(scale - coarse_scale), coarse,
                                               atol=2e-12 * np.max(np.abs(coarse)), rtol=2e-10)
                    report["bf_checks"].append(dict(two_j=two_j, level=level, maxnorm_relative_error=error, preparation=info))
                report["gluing_checks"].append(glue_two_cells(backend, maps, levels[1], two_j, None, run, identity, bf=True))
                atomic_json(args.output, report)
                for dl in args.dl:
                    report["gluing_checks"].append(glue_two_cells(backend, maps, levels[1], two_j, dl, run, identity))
                    for level, spec in enumerate(levels):
                        value, scale, info = amplitude(backend, maps, spec, two_j, dl, run, identity)
                        report["results"].append(dict(two_j=two_j, dl=dl, level=level,
                                                       cells=len(spec["cells"]), amplitude=serialized(value, scale), preparation=info))
                        summarize(report)
                        atomic_json(args.output, report)
                        print(f"j={two_j/2:g} DL={dl} cells={len(spec['cells'])}: "
                              f"log(max|W|)={report['results'][-1]['amplitude']['log_absolute_scale']:.9f}", flush=True)
    report["status"] = "completed"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
