"""Restartable full 16-simplex shell scan and CPU preparation benchmark.

Physical integer labels: K<=3, boundary j=1, gamma=1.2, canonical boundary
intertwiners i=0,1,2. Cache entries are full ordered native vertex tensors.
One native instance per subprocess; simultaneous workers never share boosters.
"""

import argparse
from itertools import product
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmarks.check_cutoffs import exact_value, native_data
from benchmarks.compression import prune_support
from benchmarks.refinement import build_block, layout
from qlg.backend import NativeBackend, backend_identity, digest
from qlg.jobs import atomic_json, cached_vertex, exclusive_lock, fingerprint, source_version
from qlg.spins import vertex_ranges

def required_spins(cutoff):
    graph = layout(2)
    needed = set()
    for node in graph["nodes"]:
        for labels in product(range(cutoff + 1), repeat=len(node["spin_indices"])):
            assignment = dict(zip(node["spin_indices"], labels))
            spins = tuple(2 * assignment[graph["face_indices"][f]]
                          if f in graph["face_indices"] else 2 for f in node["faces"])
            if all(vertex_ranges(spins)):
                needed.add(spins)
    return sorted(needed)


def cache_spec():
    tables = sorted(p for p in Path(os.environ["SL2CFOAM_TABLES"]).iterdir()
                    if p.suffix in (".3j", ".6j"))
    if sorted(p.suffix for p in tables) != [".3j", ".6j"]:
        raise ValueError("Select exactly one .3j and one .6j lookup table")
    return dict(schema=1, gamma=1.2, accuracy=2,
                backend=backend_identity(os.environ["SL2CFOAM_LIBRARY"]),
                tables={p.name: digest(p) for p in tables},
                wrapper_sha256=digest(Path(__file__).parents[1] / "qlg/backend.py"),
                numpy=np.__version__)


def checkpoint(run, spec, spins, dl, compute):
    directory = run / f"dl_{dl:03d}"
    directory.mkdir(exist_ok=True)
    identity = dict(cache_id=fingerprint(spec), dl=dl, spins=spins)
    return cached_vertex(directory / ("-".join(map(str, spins)) + ".npz"), identity, compute)


def unavailable():
    raise FileNotFoundError("Missing prepared vertex; run the vertices stage first")


class PreparedBackend:
    def __init__(self, native, run, spec):
        self.lib, self.conf = native.lib, native.conf
        self.run, self.spec = run, spec

    def vertex(self, spins, dl):
        tensor, _, reused = checkpoint(self.run, self.spec, spins, dl, unavailable)
        assert reused
        return tensor


def worker(args, spec):
    run = Path(args.run).resolve()
    start = time.perf_counter()
    if args.stage == "vertices":
        selected = required_spins(args.cutoff)[args.shard::args.processes]
        computed = reused = 0
        native_seconds = 0.
        data = Path(args.native)
        data.mkdir()
        native_data(data)
        # Factorial-table bound includes the largest auxiliary spin.
        max_two_spin = 4 * (2 * args.cutoff + 2 * args.dl[0]) + 8
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2,
                           max_two_spin, 128) as native:
            for spins in selected:
                _, seconds, hit = checkpoint(run, spec, spins, args.dl[0],
                                            lambda: native.vertex(spins, args.dl[0]))
                reused += hit
                computed += not hit
                native_seconds += 0 if hit else seconds
                # Progress survives a killed worker; atomic vertex files are authoritative.
                atomic_json(args.output, dict(status="running", computed=computed, reused=reused,
                                              selected=len(selected), native_seconds=native_seconds))
    else:
        dl = args.dl[0]
        with tempfile.TemporaryDirectory(prefix="qlg-assemble-") as temporary:
            data = Path(temporary)
            native_data(data)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2,
                               4 * (2 * args.cutoff + 2 * dl) + 8, 128) as native:
                prepared = PreparedBackend(native, run, spec)
                states = []
                for i in (0, 1, 2):
                    _, arrays, inputs, preparation = build_block(prepared, args.cutoff, dl, i)
                    arrays, inputs, support = prune_support(arrays, inputs)
                    value, methods = exact_value(arrays, inputs)
                    states.append(dict(boundary_i=i, sign=float(np.sign(value)),
                                       log_absolute_amplitude=(math.log(abs(value))
                                           + preparation["log_amplitude_scale"]) if value else None,
                                       preparation=preparation, support=support, exact_methods=methods))
        nonzero = [s["log_absolute_amplitude"] for s in states if s["sign"]]
        if not nonzero:
            raise FloatingPointError("All three boundary amplitudes vanish")
        scale = max(nonzero)
        vector = np.array([s["sign"] * math.exp(s["log_absolute_amplitude"] - scale)
                           if s["sign"] else 0. for s in states])
        vector /= np.linalg.norm(vector)
        row = dict(status="completed", K=args.cutoff, dl=dl, states=states,
                   normalized_three_state_amplitude=vector.tolist(),
                   peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   elapsed_seconds=time.perf_counter() - start)
        atomic_json(args.output, row)
        return
    atomic_json(args.output, dict(status="completed", computed=computed, reused=reused,
                                 selected=len(selected), native_seconds=native_seconds,
                                 elapsed_seconds=time.perf_counter() - start,
                                 peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def launch_group(commands, environment, timeout):
    """One bounded group; kill every owned worker before returning on timeout."""
    children = []
    start = time.perf_counter()
    try:
        for command in commands:
            children.append(subprocess.Popen(command, env=environment, start_new_session=True))
        for child in children:
            child.wait(timeout=max(.001, timeout - (time.perf_counter() - start)))
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for child in children:
            child.wait()
    if any(child.returncode for child in children):
        raise RuntimeError(f"Worker failed: {[child.returncode for child in children]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Persistent vertex-cache directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--dl", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800, help="Preparation wall limit per shell")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", choices=("vertices", "contract"), default="vertices")
    parser.add_argument("--native")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if min(args.dl) < 0 or max(args.dl) > 12 or min(args.processes, args.threads, args.timeout) <= 0:
        parser.error("Use shell cutoffs 0..12, positive CPU counts and timeout")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("Set OPENBLAS_NUM_THREADS=1 before starting Python")
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    spec = cache_spec()
    if args.worker:
        worker(args, spec)
        return
    # Let the group's finally block stop children on scheduler termination.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run / ".scan.lock"):
        manifest = run / "cache.json"
        if manifest.exists() and json.loads(manifest.read_text()) != spec:
            raise ValueError("Cache belongs to different numerical inputs/backend/tables")
        atomic_json(manifest, spec)
        # Binary/table fingerprints stay in the local cache manifest. Export
        # scientific inputs and public source revisions, without host diagnostics.
        report = dict(schema=1, gamma=1.2, boundary_j=1, accuracy="veryhigh",
                      backend_source_revision=spec["backend"]["build"].get("commit"),
                      rho_convention=spec["backend"]["build"].get("rho_convention"),
                      numpy=np.__version__, source=source_version(),
                      processes=args.processes, threads=args.threads, blas_threads=1,
                      preparation_timeout_seconds=args.timeout, results=[])
        common = [sys.executable, "-m", "benchmarks.block_scan", "--worker", "--run", str(run),
                  "--cutoff", str(args.cutoff), "--processes", str(args.processes)]
        for dl in args.dl:
            before = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="qlg-vertices-", dir=run) as temporary:
                directory = Path(temporary)
                commands = [common + ["--dl", str(dl), "--shard", str(i), "--native",
                            str(directory / f"native-{i}"), "--output", str(directory / f"{i}.json")]
                            for i in range(args.processes)]
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                try:
                    launch_group(commands, env, args.timeout)
                    status, reason = "completed", None
                except subprocess.TimeoutExpired:
                    status, reason = "resource_limit", "preparation wall limit"
                except RuntimeError as error:
                    status, reason = "worker_failure", str(error)
                rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
                preparation_seconds = time.perf_counter() - before
            row = dict(K=args.cutoff, dl=dl, status=status, reason=reason, workers=rows,
                       preparation_wall_seconds=preparation_seconds,
                       requested_core_seconds=preparation_seconds * args.processes * args.threads)
            report["results"].append(row)
            atomic_json(output, report)
            if status == "completed" and not args.prepare_only:
                row["status"] = "prepared"
                atomic_json(output, report)
                with tempfile.TemporaryDirectory(prefix="qlg-contract-result-") as temporary:
                    child_output = Path(temporary) / "result.json"
                    launch_group([common + ["--stage", "contract", "--dl", str(dl),
                                  "--output", str(child_output)]], dict(env, OMP_NUM_THREADS="1"), 300)
                    row.update(json.loads(child_output.read_text()))
                    atomic_json(output, report)
            print(f"K={args.cutoff} DL={dl}: {row['status']}; preparation "
                  f"{preparation_seconds:.3f}s, computed={sum(w['computed'] for w in rows)}, "
                  f"reused={sum(w['reused'] for w in rows)}", flush=True)
        if any(row["status"] != "completed" for row in report["results"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
