"""Calibrate SVD compression on exact blocks, then explore a larger sector.

The 16-simplex full cases include all internal spins through K. Fixed-sector
cases hold every face at j=1 and sum only intertwiners. No RG map is assumed.
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
import quimb

from benchmarks.compression import (compressed_contract, fixed_sector,
                                     local_contract, parity_structure, prune_support)
from benchmarks.refinement import build_block, expression
from qlg.backend import NativeBackend, backend_identity, digest


def worker(args):
    with tempfile.TemporaryDirectory(prefix="qlg-compression-") as temporary:
        data = Path(temporary)
        for table in Path(os.environ["SL2CFOAM_TABLES"]).iterdir():
            if table.suffix in (".3j", ".6j"):
                (data / table.name).symlink_to(table.resolve())
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 60, 256) as backend:
            before = time.perf_counter()
            if args.kind == "full":
                graph, arrays, inputs, preparation = build_block(backend, args.cutoff, args.dl, args.boundary_i)
                scale = preparation["log_amplitude_scale"]
            else:
                graph, arrays, inputs, scale = fixed_sector(backend, args.divisions, args.dl, args.boundary_i)
                preparation = dict(build_seconds=time.perf_counter() - before)
            row = dict(kind=args.kind, divisions=args.divisions, dl=args.dl,
                       cutoff=args.cutoff if args.kind == "full" else None,
                       boundary_i=args.boundary_i, simplices=len(graph["cells"]),
                       preparation=preparation, log_amplitude_scale=scale,
                       input_entries=sum(a.size for a in arrays),
                       nonzero_entries=int(sum(np.count_nonzero(a) for a in arrays)), results=[])
            reference = (float(oe.contract(expression(inputs), *arrays, optimize="greedy"))
                         if args.divisions == 2 else None)
            row["exact_normalized_reference"] = reference
            before = time.perf_counter()
            reduced, indices, support = prune_support(arrays, inputs)
            local, local_inputs, exact = local_contract(reduced, indices, max_entries=10000 if args.kind == "full" else 243)
            row["exact_reduction"] = dict(support, **exact, seconds=time.perf_counter() - before)
            if len(local) == 1 and not local_inputs[0]:
                value = float(local[0])
                row["exact_reduction"]["value"] = value
                if reference:
                    np.testing.assert_allclose(value, reference, rtol=5e-12, atol=0)
            # Record first-use linear-algebra overhead separately.
            if args.divisions == 2:
                warmup = compressed_contract(arrays, inputs, 2)
                row["warmup_seconds"] = warmup["search_seconds"] + warmup["contract_seconds"]
            for chi in args.chi:
                for seed in args.seed:
                    before = time.perf_counter()
                    try:
                        result = compressed_contract(reduced, inputs, chi, seed=seed,
                                                     gauges=args.gauges, max_entries=args.max_entries)
                        if reference and result["log_absolute_value"] is not None:
                            estimate = result["sign"] * math.exp(result["log_absolute_value"])
                            result["relative_error"] = estimate / reference - 1
                        result["status"] = "completed"
                    except (MemoryError, TimeoutError) as error:
                        result = dict(chi=chi, seed=seed, gauge_distance=args.gauges,
                                      status="resource_limit", reason=str(error),
                                      elapsed_seconds=time.perf_counter() - before)
                    except (np.linalg.LinAlgError, FloatingPointError) as error:
                        result = dict(chi=chi, seed=seed, gauge_distance=args.gauges,
                                      status="numerical_failure", reason=str(error),
                                      elapsed_seconds=time.perf_counter() - before)
                    row["results"].append(result)
                    Path(args.output).write_text(json.dumps(row, indent=2) + "\n")
                    print(f"{args.kind} {args.divisions**4} cells i={args.boundary_i} "
                          f"DL={args.dl} chi={chi} seed={seed}: {result['status']}", flush=True)
            row["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            Path(args.output).write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/compression-local.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kind", choices=("full", "fixed-sector"), default="full")
    parser.add_argument("--divisions", type=int, choices=(2, 4), default=2)
    parser.add_argument("--cutoff", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dl", type=int, default=0)
    parser.add_argument("--boundary-i", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--chi", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, nargs="+", default=[37, 71])
    parser.add_argument("--gauges", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=32_000_000)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if args.dl < 0 or min(args.chi) < 1 or args.gauges < 0 or args.max_entries < 1:
        parser.error("Nonnegative shell/gauge settings and positive sizes required")
    if args.kind == "full" and args.divisions != 2:
        parser.error("A full spin-summed division-4 block is not implemented")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_DISABLE_JIT"):
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before starting Python")
    if args.worker:
        worker(args)
        return
    report = dict(schema=1, gamma=1.2, boundary_j=1, python=sys.version,
                  numpy=np.__version__, quimb=quimb.__version__, platform=platform.platform(),
                  numba_jit=False, threads=1,
                  backend=backend_identity(os.environ["SL2CFOAM_LIBRARY"]),
                  source_sha256={p: digest(Path(__file__).parents[1] / p) for p in
                                 ("benchmarks/compression.py", "benchmarks/compare_compression.py",
                                  "benchmarks/refinement.py", "qlg/topology.py")},
                  parity_structure=[parity_structure(d) for d in (2, 4)], results=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [("full", 2, 1, 0, i, args.chi, args.gauges) for i in (0, 1, 2)]
    cases += [("full", 2, 1, 2, 1, args.chi, args.gauges),
              ("full", 2, 2, 0, 1, args.chi, args.gauges),
              ("fixed-sector", 2, 1, 0, 1, args.chi, args.gauges)]
    if not args.small_only:
        cases += [("fixed-sector", 4, 1, 0, 1, [2, 3, 4, 6, 8], args.gauges),
                  ("fixed-sector", 4, 1, 0, 1, [4, 6, 8], 2)]
    for kind, divisions, cutoff, dl, i, chis, gauges in cases:
        with tempfile.TemporaryDirectory(prefix="qlg-compression-result-") as temporary:
            child_output = Path(temporary) / "result.json"
            command = [sys.executable, "-m", "benchmarks.compare_compression", "--worker",
                       "--kind", kind, "--divisions", str(divisions), "--cutoff", str(cutoff),
                       "--dl", str(dl), "--boundary-i", str(i), "--chi", *map(str, chis),
                       "--seed", *map(str, args.seed), "--gauges", str(gauges),
                       "--max-entries", str(args.max_entries), "--output", str(child_output)]
            subprocess.run(command, check=True, timeout=600)
            report["results"].append(json.loads(child_output.read_text()))
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
