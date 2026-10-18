"""Full 16-simplex shell/spin-cutoff scan, with exact contractions.

Each case owns one native process and cache. Changes between cutoffs are
diagnostics, not certified errors or evidence of a continuum limit.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import opt_einsum as oe

from benchmarks.compression import prune_support
from benchmarks.refinement import build_block, expression
from qlg.backend import NativeBackend, backend_identity, digest


def native_data(directory):
    for table in Path(os.environ["SL2CFOAM_TABLES"]).iterdir():
        if table.suffix in (".3j", ".6j"):
            (directory / table.name).symlink_to(table.resolve())


def exact_value(arrays, inputs, max_entries=16_000_000):
    """Compare two complete contractions; refuse oversized planned outputs."""
    equation = expression(inputs)
    values, records = [], []
    for method in ("greedy", "random-greedy-64"):
        from benchmarks.compare_refinement import search
        path, info = search(inputs, [a.shape for a in arrays], method)
        if info["largest_intermediate_bytes"] > 8 * max_entries:
            records.append(dict(method=method, **info, status="resource_limit"))
            continue
        before = time.perf_counter()
        value = float(oe.contract(equation, *arrays, optimize=path))
        values.append(value)
        records.append(dict(method=method, **info, status="completed", value=value,
                            contract_seconds=time.perf_counter() - before))
    if not values:
        raise MemoryError("No exact path fits the intermediate-entry cap")
    np.testing.assert_allclose(values, values[0], rtol=5e-11, atol=0)
    return values[-1], records


def provenance(paths):
    return dict(gamma=1.2, boundary_j=1, python=sys.version, numpy=np.__version__,
                platform=platform.platform(), threads=1,
                backend=backend_identity(os.environ["SL2CFOAM_LIBRARY"]),
                source_sha256={p: digest(Path(__file__).parents[1] / p) for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/cutoffs-local.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cutoff", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--dl", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--cases", nargs="+", default=["1:0", "1:1", "1:2", "1:3", "1:4",
                                                     "2:0", "2:1", "2:2", "3:0", "3:1", "3:2"],
                        help="K:Delta_l pairs")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before starting Python")
    if args.dl < 0 or args.timeout < 1:
        parser.error("Nonnegative shell cutoff and positive timeout required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        with tempfile.TemporaryDirectory(prefix="qlg-cutoff-") as temporary:
            data = Path(temporary)
            native_data(data)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 60, 256) as backend:
                graph, arrays, inputs, preparation = build_block(backend, args.cutoff, args.dl, 1)
                # Each internal face is constrained by a tetrahedron with
                # three j=1 boundary faces: j_internal <= 3, exactly.
                from itertools import combinations
                bounded = set()
                for tetrahedron in graph["tetrahedron_indices"]:
                    faces = [f for f in combinations(tetrahedron, 3) if f in graph["face_indices"]]
                    if len(faces) == 1:
                        bounded.add(faces[0])
                assert len(bounded) == len(graph["face_indices"])
                reduced, indices, support = prune_support(arrays, inputs)
                value, methods = exact_value(reduced, indices)
                result = dict(K=args.cutoff, dl=args.dl, boundary_i=1, status="completed",
                              preparation=preparation, support=support,
                              trimmed_factor_bytes=sum(a.nbytes for a in reduced),
                              normalized_value=value, sign=float(np.sign(value)),
                              log_absolute_amplitude=math.log(abs(value)) + preparation["log_amplitude_scale"],
                              exact_methods=methods, exact_internal_spin_upper_bound=3,
                              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(result, indent=2) + "\n")
        return
    report = dict(schema=1, **provenance(("benchmarks/check_cutoffs.py", "benchmarks/refinement.py",
                                        "benchmarks/compression.py", "benchmarks/compare_refinement.py")),
                  case_timeout_seconds=args.timeout, results=[])
    for case in args.cases:
        cutoff, dl = map(int, case.split(":"))
        with tempfile.TemporaryDirectory(prefix="qlg-cutoff-result-") as temporary:
            child_output = Path(temporary) / "result.json"
            command = [sys.executable, "-m", "benchmarks.check_cutoffs", "--worker",
                       "--cutoff", str(cutoff), "--dl", str(dl), "--output", str(child_output)]
            before = time.perf_counter()
            try:
                subprocess.run(command, check=True, timeout=args.timeout)
                row = json.loads(child_output.read_text())
            except subprocess.TimeoutExpired:
                row = dict(K=cutoff, dl=dl, status="resource_limit", reason="worker time limit")
            row["elapsed_seconds"] = time.perf_counter() - before
            report["results"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"K={cutoff} DL={dl}: {row['status']} "
                  f"log|A|={row.get('log_absolute_amplitude')} ({row['elapsed_seconds']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
