"""Compare cold full-block preparation at fixed inputs and CPU budgets.

Each layout gets its own empty vertex/booster cache. Repetitions reverse layout
order. Every ordered vertex tensor and all three boundary amplitudes must agree.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from qlg.jobs import atomic_json, source_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--dl", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="Validate existing complete cold reports and run missing cases")
    parser.add_argument("--layouts", nargs="+", default=["1:1", "1:2", "1:4", "2:1", "2:2", "4:1"],
                        help="processes:OpenMP_threads per process")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference = None
    report = dict(schema=1, source=source_version(), results=[])
    command = [sys.executable, "-m", "benchmarks.block_scan", "--cutoff", str(args.cutoff),
               "--dl", str(args.dl), "--timeout", "600"]
    for repeat in range(args.repeats):
        for setting in args.layouts[::1 if repeat % 2 == 0 else -1]:
            processes, threads = map(int, setting.split(":"))
            name = f"r{repeat}-p{processes}-t{threads}"
            run, output = root / name, root / (name + ".json")
            existing = run.exists()
            if existing and (not args.resume or not output.exists()):
                raise ValueError("Use an unused run-root, or --resume with complete cold reports")
            elapsed = None
            if not existing:
                before = time.perf_counter()
                subprocess.run(command + ["--run", str(run), "--output", str(output),
                                          "--processes", str(processes), "--threads", str(threads)],
                               check=True)
                elapsed = time.perf_counter() - before
            recorded = json.loads(output.read_text())
            # Older local reports may contain host/build diagnostics. Resume
            # their measurements without copying those diagnostics to exports.
            public_fields = ("schema", "gamma", "boundary_j", "accuracy", "backend_source_revision",
                             "rho_convention", "numpy", "source", "numeric_source_revision",
                             "processes", "threads", "blas_threads", "preparation_timeout_seconds", "results")
            row = {k: recorded[k] for k in public_fields if k in recorded}
            row.update(repeat=repeat, driver_wall_seconds=elapsed, loaded_existing_cold_report=existing)
            case = row["results"][0]
            if (case["K"], case["dl"], row["processes"], row["threads"]) != (args.cutoff, args.dl, processes, threads):
                raise ValueError("Existing report has different benchmark inputs")
            if sum(w["reused"] for w in case["workers"]) or case["status"] != "completed":
                raise AssertionError("Incomplete or warm benchmark")
            if reference is None:
                reference, reference_run = case, run
            relative_errors = [abs(math.expm1(s["log_absolute_amplitude"]
                                              - r["log_absolute_amplitude"]))
                               for s, r in zip(case["states"], reference["states"])]
            assert [s["sign"] for s in case["states"]] == [s["sign"] for s in reference["states"]]
            vertex_error = 0.
            for path in run.glob("dl_*/*.npz"):
                with np.load(path) as actual, np.load(reference_run / path.parent.name / path.name) as ref:
                    a, b = actual["tensor"], ref["tensor"]
                    error = float(np.max(np.abs(a - b)) / np.max(np.abs(b))) if np.any(b) else float(np.max(np.abs(a)))
                    vertex_error = max(vertex_error, error)
                    # Entries near cancellation zero need a tolerance scaled to
                    # the full tensor, not an arbitrary absolute amplitude.
                    np.testing.assert_allclose(a, b, rtol=1e-10,
                                               atol=1e-14 * float(np.max(np.abs(b))))
            assert vertex_error < 1e-10
            assert max(relative_errors) < 1e-10
            row.update(max_amplitude_relative_difference=max(relative_errors),
                       max_vertex_relative_maxnorm_difference=vertex_error)
            report["results"].append(row)
            atomic_json(args.output, report)
    # Reuse the baseline tensors in a fresh process with a changed CPU layout.
    output = root / "warm.json"
    subprocess.run(command + ["--run", str(reference_run), "--output", str(output),
                              "--processes", "2", "--threads", "2"], check=True)
    report["warm_reuse"] = json.loads(output.read_text())
    warm = report["warm_reuse"]["results"][0]
    assert sum(w["computed"] for w in warm["workers"]) == 0
    assert [s["sign"] for s in warm["states"]] == [s["sign"] for s in reference["states"]]
    assert max(abs(math.expm1(s["log_absolute_amplitude"] - r["log_absolute_amplitude"]))
               for s, r in zip(warm["states"], reference["states"])) < 1e-10
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
