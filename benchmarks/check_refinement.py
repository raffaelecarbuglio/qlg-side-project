"""Fixed-order compression against exact full spin sums on 16--30 simplices.

The extra complexes are conforming boundary-edge subdivisions for algorithm
calibration, not matched-boundary continuum measurements. Integer internal
spins are forced by parity on every selected complex, not imposed by hand.
"""

import argparse
from itertools import combinations
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmarks.check_cutoffs import exact_value, native_data, provenance
from benchmarks.compression import (compressed_contract, compression_network,
                                     compression_path, prune_support)
from benchmarks.refinement import build_block, complex_layout
from qlg.backend import NativeBackend
from qlg.topology import boundary, edgewise_simplex, facets


def intermediate_complex(steps):
    """Nested 16,20,25,30-cell balls, splitting all cells incident on an edge."""
    vertices, cells = edgewise_simplex(2)
    for edge in ((1, 3), (4, 7), (8, 12))[:steps]:
        boundary_edges = {e for t in boundary(cells) for e in combinations(t, 2)}
        assert edge in boundary_edges
        midpoint = tuple((a + b) / 2 for a, b in zip(vertices[edge[0]], vertices[edge[1]]))
        new_vertex = len(vertices)
        children = []
        for cell in cells:
            if set(edge) <= set(cell):
                children.extend(tuple(sorted((set(cell) - {v}) | {new_vertex})) for v in edge)
            else:
                children.append(cell)
        vertices, cells = vertices + (midpoint,), tuple(children)
    return complex_layout(vertices, cells)


def geometry_checks(graph):
    """No degenerate cells, same auxiliary four-volume, valid incidence."""
    volumes = []
    for cell in graph["cells"]:
        points = np.asarray([graph["vertices"][i][1:] for i in cell], dtype=float)
        volumes.append(float(abs(np.linalg.det(points[1:] - points[0])) / 24))
    assert min(volumes) > 0
    # Original barycentric coordinates sum to two: coordinate four-volume 2^4/4!.
    np.testing.assert_allclose(math.fsum(volumes), 2**4 / 24, atol=1e-14, rtol=0)
    boundary_edges = {e for t in graph["boundary_tetrahedra"] for e in combinations(t, 2)}
    internal_edges = set(facets(graph["cells"], 2)) - boundary_edges
    assert not internal_edges
    for tetrahedron in graph["boundary_tetrahedra"]:
        assert any(all(graph["vertices"][i][a] == 0 for i in tetrahedron) for a in range(5))
    euler = sum((-1)**(size - 1) * len(facets(graph["cells"], size)) for size in range(1, 6))
    assert euler == 1
    return dict(auxiliary_volume=math.fsum(volumes), euler_characteristic=euler,
                internal_edges=0, cells=len(volumes), internal_faces=len(graph["face_indices"]),
                internal_tetrahedra=len(graph["tetrahedron_indices"]),
                boundary_tetrahedra=len(graph["boundary_tetrahedra"]))


def worker(args):
    graph = intermediate_complex(args.steps)
    geometry = geometry_checks(graph)
    with tempfile.TemporaryDirectory(prefix="qlg-fixed-order-") as temporary:
        data = Path(temporary)
        native_data(data)
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 60, 256) as backend:
            _, arrays, inputs, preparation = build_block(backend, args.cutoff, args.dl,
                                                        args.boundary_i, graph=graph)
            arrays, inputs, support = prune_support(arrays, inputs)
            reference, exact_methods = exact_value(arrays, inputs)
            row = dict(steps=args.steps, K=args.cutoff, dl=args.dl, boundary_i=args.boundary_i,
                       geometry=geometry, preparation=preparation, exact_support=support,
                       exact_normalized_reference=reference, exact_methods=exact_methods,
                       log_absolute_amplitude=math.log(abs(reference)) + preparation["log_amplitude_scale"],
                       sign=float(np.sign(reference)), fixed_paths=[], results=[])
            before = time.perf_counter()
            network = compression_network(arrays, inputs)
            row["equality_tree_seconds"] = time.perf_counter() - before
            # Save the actual resolved network: random equality-index names are
            # replaced with ordinal labels so a recorded path can be replayed.
            labels = {label: i for i, label in enumerate(network.ind_map)}
            row["resolved_inputs"] = [[labels[i] for i in t.inds] for t in network]
            row["resolved_shapes"] = [list(t.shape) for t in network]
            for seed in args.seed:
                before = time.perf_counter()
                path = compression_path(network, chi=8, seed=seed)
                row["fixed_paths"].append(dict(seed=seed, search_chi=8,
                                               search_seconds=time.perf_counter() - before,
                                               path=[list(p) for p in path]))
                for chi in args.chi:
                    before = time.perf_counter()
                    try:
                        result = compressed_contract(arrays, inputs, chi, seed=seed, network=network,
                                                     path=path, max_entries=args.max_entries)
                        estimate = result["sign"] * math.exp(result["log_absolute_value"]) if result["sign"] else 0
                        result.update(relative_error=estimate / reference - 1, status="completed")
                    except (MemoryError, TimeoutError) as error:
                        result = dict(chi=chi, seed=seed, status="resource_limit", reason=str(error),
                                      elapsed_seconds=time.perf_counter() - before)
                    except (np.linalg.LinAlgError, FloatingPointError) as error:
                        result = dict(chi=chi, seed=seed, status="numerical_failure", reason=str(error),
                                      elapsed_seconds=time.perf_counter() - before)
                    row["results"].append(result)
                    Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
                    print(f"{geometry['cells']} cells K={args.cutoff} DL={args.dl} i={args.boundary_i} "
                          f"path={seed} chi={chi}: {result['status']} error={result.get('relative_error')}", flush=True)
            row["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            Path(args.output).write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/fixed-order-local.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--steps", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--cutoff", type=int, default=1, choices=(1, 2))
    parser.add_argument("--dl", type=int, default=0)
    parser.add_argument("--boundary-i", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--chi", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--seed", type=int, nargs="+", default=[37, 71])
    parser.add_argument("--max-entries", type=int, default=16_000_000)
    parser.add_argument("--cases", nargs="+", default=["0:1:0:1", "0:1:2:1", "0:2:0:1", "0:1:0:2",
                                                     "1:1:0:1", "2:1:0:1", "3:1:0:1"],
                        help="local_refinement_steps:K:Delta_l:boundary_i")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT"):
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before starting Python")
    if args.dl < 0 or min(args.chi) < 1 or args.max_entries < 1:
        parser.error("Invalid shell cutoff, retained rank or entry cap")
    if args.worker:
        worker(args)
        return
    report = dict(schema=1, **provenance(("benchmarks/check_refinement.py", "benchmarks/check_cutoffs.py",
                                        "benchmarks/refinement.py", "benchmarks/compression.py",
                                        "benchmarks/compare_refinement.py", "qlg/topology.py")),
                  numba_jit=False, max_entries=args.max_entries, worker_timeout_seconds=600, results=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        steps, cutoff, dl, i = map(int, case.split(":"))
        with tempfile.TemporaryDirectory(prefix="qlg-fixed-order-result-") as temporary:
            child_output = Path(temporary) / "result.json"
            command = [sys.executable, "-m", "benchmarks.check_refinement", "--worker",
                       "--steps", str(steps), "--cutoff", str(cutoff), "--dl", str(dl),
                       "--boundary-i", str(i), "--chi", *map(str, args.chi),
                       "--seed", *map(str, args.seed), "--max-entries", str(args.max_entries),
                       "--output", str(child_output)]
            try:
                subprocess.run(command, check=True, timeout=600)
                row = json.loads(child_output.read_text())
            except subprocess.TimeoutExpired:
                row = json.loads(child_output.read_text()) if child_output.exists() else {}
                row.update(status="resource_limit", reason="worker time limit", steps=steps, K=cutoff, dl=dl)
            report["results"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
