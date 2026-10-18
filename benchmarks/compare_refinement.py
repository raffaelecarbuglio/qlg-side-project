"""Compare exact contractions and uniform sampling on a real 1->16 EPRL block.

Run with one OpenMP/BLAS thread, after installing benchmarks/requirements.txt.
Each cutoff uses an isolated native cache. Results include preparation and path
search separately; larger-level costs are symbolic estimates, not timings.
"""

import argparse
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import cotengra as ctg
import numpy as np
import opt_einsum as oe

from benchmarks.refinement import build_block, condition_factors, expression, layout, sector_factors
from qlg.backend import NativeBackend, backend_identity, digest
from qlg.spins import intertwiner_range


def search(inputs, shapes, method):
    equation = expression(inputs)
    before = time.perf_counter()
    if method in ("random-greedy-64", "kahypar-32"):
        sizes = {i: d for inds, shape in zip(inputs, shapes) for i, d in zip(inds, shape)}
        if method == "random-greedy-64":
            optimizer = ctg.RandomGreedyOptimizer(max_repeats=64, parallel=False, seed=37)
        else:
            optimizer = ctg.HyperOptimizer(methods=["kahypar"], optlib="random", max_repeats=32,
                                          max_time=30, parallel=False, optlib_opts={"seed": 37})
        tree = optimizer.search(inputs, (), sizes)
        path = tree.get_path()
    else:
        path, _ = oe.contract_path(equation, *shapes, shapes=True, optimize=method)
    seconds = time.perf_counter() - before
    _, info = oe.contract_path(equation, *shapes, shapes=True, optimize=path)
    return path, dict(search_seconds=seconds, estimated_flops=str(info.opt_cost),
                      largest_intermediate_bytes=int(info.largest_intermediate) * 8,
                      path=[list(p) for p in path])


def measure(function, repeats):
    times, values = [], []
    for _ in range(repeats):
        before = time.perf_counter()
        values.append(float(function()))
        times.append(time.perf_counter() - before)
    return dict(seconds=times, median_seconds=statistics.median(times), value=values[-1])


def configurations(graph, assignment):
    """Number of admissible intertwiner assignments at these internal spins."""
    count = 1
    fi = graph["face_indices"]
    for tetrahedron in graph["tetrahedron_indices"]:
        js = [2 * assignment[fi[f]] if f in fi else 2 for f in combinations(tetrahedron, 3)]
        count *= len(intertwiner_range(*js))
    return count


def conditioned_methods(arrays, inputs, nf, spin_dim, repeats):
    """Try exact cutset conditioning: enumerate 0..5 chosen spin indices.

    Choose a subset by predicted arithmetic cost; evaluate every assignment
    of that subset exactly. These are independent tasks suitable for HPC.
    """
    results = {}
    for number in range(nf + 1):
        before = time.perf_counter()
        candidates = []
        for subset in combinations(range(nf), number):
            selected, remaining = condition_factors(arrays, inputs, dict.fromkeys(subset, 0))
            path, info = search(remaining, [a.shape for a in selected], "greedy")
            candidates.append((int(info["estimated_flops"]), subset, path, info))
        _, subset, greedy_path, greedy_info = min(candidates, key=lambda x: x[0])
        selected, remaining = condition_factors(arrays, inputs, dict.fromkeys(subset, 0))
        random_path, random_info = search(remaining, [a.shape for a in selected], "random-greedy-64")
        path, info = ((random_path, random_info) if int(random_info["estimated_flops"]) <
                      int(greedy_info["estimated_flops"]) else (greedy_path, greedy_info))
        planning = time.perf_counter() - before
        function = oe.contract_expression(expression(remaining), *[a.shape for a in selected], optimize=path)
        assignments = [dict(zip(subset, values)) for values in product(range(spin_dim), repeat=number)]

        def evaluate():
            return math.fsum(float(function(*condition_factors(arrays, inputs, s)[0])) for s in assignments)

        results[f"condition-{number}-spins"] = dict(
            measure(evaluate, repeats), chosen_spin_indices=subset,
            search_seconds=planning, slices=len(assignments),
            estimated_total_flops=str(int(info["estimated_flops"]) * len(assignments)),
            largest_intermediate_bytes=info["largest_intermediate_bytes"], path=info["path"])
    return results


def compare(backend, cutoff, dl, repeats, boundary_i):
    graph, arrays, inputs, preparation = build_block(backend, cutoff, dl, boundary_i)
    shapes = [a.shape for a in arrays]
    equation = expression(inputs)
    results = {}
    for method in ("greedy", "random-greedy-64"):
        path, info = search(inputs, shapes, method)
        if info["largest_intermediate_bytes"] > 256 * 1024**2:
            results[method] = dict(info, executed=False)
            continue
        before = time.perf_counter()
        function = oe.contract_expression(equation, *shapes, optimize=path)
        info["compile_seconds"] = time.perf_counter() - before
        info.update(measure(lambda: function(*arrays), repeats))
        results[method] = dict(info, executed=True)
    reference = results["random-greedy-64"]["value"]
    nf = len(graph["face_indices"])
    assignments = list(product(range(cutoff + 1), repeat=nf))
    sample_arrays, sample_inputs = sector_factors(arrays, inputs, assignments[0], nf)
    sector_shapes = [a.shape for a in sample_arrays]
    before = time.perf_counter()
    sector_function = oe.contract_expression(expression(sample_inputs), *sector_shapes, optimize="greedy")
    sector_planning = time.perf_counter() - before

    def evaluate(spins, squared=False):
        selected, _ = sector_factors(arrays, inputs, spins, nf)
        return float(sector_function(*[a * a if squared else a for a in selected]))

    def exhaustive():
        return math.fsum(evaluate(spins) for spins in assignments)

    results["sector-enumeration"] = dict(measure(exhaustive, repeats),
                                         search_compile_seconds=sector_planning,
                                         spin_sectors=len(assignments))
    results.update(conditioned_methods(arrays, inputs, nf, cutoff + 1, repeats))
    for result in results.values():
        if "value" in result:
            np.testing.assert_allclose(result["value"], reference, rtol=5e-12, atol=0)
    sectors = [evaluate(spins) for spins in assignments]
    counts = [configurations(graph, spins) for spins in assignments]
    squared = [evaluate(spins, squared=True) for spins in assignments]
    n = len(assignments)
    spin_variance = max(0.0, n * math.fsum(x * x for x in sectors) / reference**2 - 1)
    joint_variance = max(0.0, n * math.fsum(c * s for c, s in zip(counts, squared)) / reference**2 - 1)
    rng = np.random.default_rng(20261018)
    trials = []
    # Actual repeated contractions, without memoizing sector answers.
    for samples in (128, 1024):
        for trial in range(4):
            draws = rng.integers(0, n, size=samples)
            before = time.perf_counter()
            estimate = n * math.fsum(evaluate(assignments[k]) for k in draws) / samples
            trials.append(dict(samples=samples, trial=trial, seconds=time.perf_counter() - before,
                               value=estimate, relative_error=estimate / reference - 1,
                               predicted_relative_standard_error=math.sqrt(spin_variance / samples)))
    # All transformations below are exact and leave the edge contractions invariant.
    rotated = [a.copy() for a in arrays]
    for index in graph["tetrahedron_indices"].values():
        dim = next(a.shape[inds.index(index)] for a, inds in zip(arrays, inputs) if index in inds)
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        for k, inds in enumerate(inputs):
            if index in inds:
                axis = inds.index(index)
                rotated[k] = np.moveaxis(np.tensordot(q, rotated[k], axes=(1, axis)), 0, axis)
    path = [tuple(p) for p in results["random-greedy-64"]["path"]]
    rotated_value = float(oe.contract(equation, *rotated, optimize=path))
    np.testing.assert_allclose(rotated_value, reference, rtol=5e-12, atol=0)
    absolute_sum = float(oe.contract(equation, *[np.abs(a) for a in arrays], optimize=path))
    return dict(cutoff=cutoff, dl=dl, boundary_j=1, boundary_i=boundary_i, preparation=preparation,
                normalized_value=reference,
                log_absolute_amplitude=math.log(abs(reference)) + preparation["log_amplitude_scale"],
                exact_methods=results, uniform_spin_monte_carlo=trials,
                uniform_spin_relative_variance=spin_variance,
                uniform_spin_samples_for_one_percent_rms=math.ceil(spin_variance / 0.01**2),
                uniform_spin_and_admissible_intertwiner_relative_variance=joint_variance,
                uniform_joint_samples_for_one_percent_rms=math.ceil(joint_variance / 0.01**2),
                explicit_admissible_configurations=sum(counts),
                absolute_term_cancellation_ratio=abs(reference) / absolute_sum,
                basis_rotation_relative_error=abs(rotated_value / reference - 1),
                sectors=[dict(spins=s, value=v, configurations=c) for s, v, c in zip(assignments, sectors, counts)],
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def larger_plans():
    """Dense symbolic upper-cost comparisons; no amplitude arrays allocated.

    Fixed spins j=1 have three intertwiner states. With all j<=1, a generic
    later level needs three spin states (0,1/2,1) and five padded intertwiner
    states (0,1/2,1,3/2,2). Actual admissibility produces substantial sparsity.
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    plans = []
    graph = layout(4)
    for kind, spin_dim, intertwiner_dim in (("fixed-j=1-sector", 0, 3),
                                            ("all-j<=1-dense-padded", 3, 5)):
        inputs = [(node["spin_indices"] if spin_dim else ()) + node["intertwiner_indices"]
                  for node in graph["nodes"]]
        if spin_dim:
            inputs += [(i,) for i in range(len(graph["face_indices"]))]
        sizes = {i: intertwiner_dim for i in graph["tetrahedron_indices"].values()}
        if spin_dim:
            sizes.update({i: spin_dim for i in graph["face_indices"].values()})
        shapes = [tuple(sizes[i] for i in inds) for inds in inputs]
        for method in ("greedy", "random-greedy-64", "kahypar-32"):
            path, info = search(inputs, shapes, method)
            tree = ctg.ContractionTree.from_path(inputs, (), sizes, path=path)
            before = time.perf_counter()
            sliced = tree.slice(target_size=8 * 1024**2, max_repeats=8, seed=37)
            plans.append(dict(kind=kind, method=method, simplices=len(graph["cells"]),
                              internal_faces=len(graph["face_indices"]),
                              internal_tetrahedra=len(graph["tetrahedron_indices"]),
                              **info, slicing_search_seconds=time.perf_counter() - before,
                              slices=sliced.nslices,
                              sliced_largest_intermediate_bytes=int(sliced.max_size()) * 8,
                              sliced_estimated_flops=str(sliced.contraction_cost())))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/refinement-local.json")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--case", nargs=2, type=int, action="append", metavar=("K", "DL"))
    parser.add_argument("--skip-larger-plans", action="store_true")
    parser.add_argument("--boundary-i", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            parser.error(f"Set {variable}=1 before starting Python")
    library, tables = os.environ["SL2CFOAM_LIBRARY"], Path(os.environ["SL2CFOAM_TABLES"])
    if args.worker:
        cutoff, dl = args.case[0]
        with tempfile.TemporaryDirectory(prefix="qlg-edgewise-") as temporary:
            data = Path(temporary)
            for table in tables.iterdir():
                if table.suffix in (".3j", ".6j"):
                    (data / table.name).symlink_to(table.resolve())
            with NativeBackend(library, data, 1.2, 2, 60, 256) as backend:
                row = compare(backend, cutoff, dl, args.repeats, args.boundary_i)
        Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(schema=1, gamma=1.2, accuracy="veryhigh", threads=1,
                  python=sys.version, numpy=np.__version__, opt_einsum=oe.__version__,
                  cotengra=ctg.__version__, platform=platform.platform(),
                  cpu=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")), "unknown"),
                  tables_sha256={p.name: digest(p) for p in tables.iterdir() if p.suffix in (".3j", ".6j")},
                  backend=backend_identity(library),
                  source_sha256={p: digest(Path(__file__).parents[1] / p) for p in
                                 ("benchmarks/compare_refinement.py", "benchmarks/refinement.py", "qlg/topology.py")},
                  results=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for cutoff, dl in args.case or [(1, 0), (1, 2), (2, 0)]:
        if cutoff not in (1, 2) or dl < 0:
            parser.error("This small-block benchmark supports K=1,2 and nonnegative DL")
        with tempfile.TemporaryDirectory(prefix="qlg-edgewise-result-") as temporary:
            child_output = Path(temporary) / "result.json"
            command = [sys.executable, "-m", "benchmarks.compare_refinement", "--worker",
                       "--case", str(cutoff), str(dl), "--repeats", str(args.repeats),
                       "--boundary-i", str(args.boundary_i), "--output", str(child_output)]
            subprocess.run(command, check=True, timeout=600)
            row = json.loads(child_output.read_text())
        report["results"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"K={cutoff} DL={dl}: build {row['preparation']['build_seconds']:.3f}s; "
              + "; ".join(f"{k} {v['median_seconds']:.6f}s" for k, v in row["exact_methods"].items()
                          if "median_seconds" in v), flush=True)
    if not args.skip_larger_plans:
        report["larger_level_symbolic_plans"] = larger_plans()
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
