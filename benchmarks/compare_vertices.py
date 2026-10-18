"""Compare full vertex tensors with boundary-specific slices on real Delta4 data.

Run from the repository root with ``python -m benchmarks.compare_vertices``.
Each method/repetition has its own process and native booster cache. A cold
calculation is followed by a warm calculation; native vertex caching is off.
"""

import argparse
import ctypes as ct
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

import numpy as np

from qlg.amplitudes import delta4_sector, ring_amplitude
from qlg.backend import NativeBackend, Tensor, backend_identity, digest
from qlg.spins import delta4_bulk_spins, delta4_spins, vertex_ranges


def boundary_slice(backend, spins, dl, two_i):
    """Fix only the three external labels; keep both internal axes complete."""
    ranges = list(vertex_ranges(spins))
    for k in (0, 1, 4):
        if two_i not in ranges[k]:
            raise ValueError("Inadmissible boundary intertwiner")
        ranges[k] = (two_i,)
    bounds = [value for r in ranges for value in (r[0], r[-1])]
    shape = tuple(len(r) for r in reversed(ranges))
    ptr = backend.lib.sl2cfoam_vertex_range((ct.c_int * 10)(*spins), *bounds, dl, 1)
    if not ptr:
        raise RuntimeError("Native slice returned null")
    try:
        t = ptr.contents
        if t.num_keys != 5 or tuple(t.dims) != shape or t.dim != math.prod(shape):
            raise RuntimeError("Unexpected native slice layout")
        if tuple(t.strides) != (1,) + tuple(np.cumprod(shape[:-1])):
            raise RuntimeError("Unexpected native slice strides")
        array = np.ctypeslib.as_array(t.d, shape=(t.dim,)).copy().reshape(shape, order="F")
        if not np.isfinite(array).all():
            raise FloatingPointError("Non-finite native slice")
        return array
    finally:
        backend.lib.sl2cfoam_vertex_free(ptr)


def calculate(backend, two_j, dl, states, method):
    start = time.perf_counter()
    vertex_seconds = contraction_seconds = 0.0
    returned_bytes = max_tensor_bytes = 0
    contributions = {i: [] for i in states}
    for x in delta4_bulk_spins(two_j):
        spins = delta4_spins(two_j, x)
        if method == "full":
            before = time.perf_counter()
            tensor = backend.vertex(spins, dl)
            vertex_seconds += time.perf_counter() - before
            returned_bytes += tensor.nbytes
            max_tensor_bytes = max(max_tensor_bytes, tensor.nbytes)
        for i in states:
            if method == "slice":
                before = time.perf_counter()
                tensor = boundary_slice(backend, spins, dl, i)
                vertex_seconds += time.perf_counter() - before
                returned_bytes += tensor.nbytes
                max_tensor_bytes = max(max_tensor_bytes, tensor.nbytes)
            before = time.perf_counter()
            if method == "full":
                value = delta4_sector(tensor, [i // 2] * 12)
            else:
                matrix = tensor[0, :, :, 0, 0]
                value = ring_amplitude([matrix] * 4)
            contributions[i].append(float((x + 1) * value))
            contraction_seconds += time.perf_counter() - before
    amplitudes = [math.fsum(contributions[i]) for i in states]
    result = dict(seconds=time.perf_counter() - start, vertex_seconds=vertex_seconds,
                  contraction_seconds=contraction_seconds, returned_bytes=returned_bytes,
                  max_tensor_bytes=max_tensor_bytes, amplitudes=amplitudes,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if two_j == 2:
        reference = np.loadtxt(Path(__file__).resolve().parents[1] /
                               "qlg/reference/delta4_gamma1.2_j1.tsv")[dl]
        expected = reference[[i // 2 for i in states]]
        np.testing.assert_allclose(amplitudes, expected, rtol=1e-5, atol=0)
        result["published_reference_max_relative_error"] = float(
            np.max(np.abs(np.asarray(amplitudes) / expected - 1)))
    return result


def worker(args):
    with tempfile.TemporaryDirectory(prefix="qlg-bench-") as tmp:
        data = Path(tmp)
        for pattern in ("*.3j", "*.6j"):
            tables = list(Path(args.tables).glob(pattern))
            if len(tables) != 1:
                raise ValueError("Need exactly one table of each type")
            (data / tables[0].name).symlink_to(tables[0].resolve())
        max_spin = 4 * (3 * args.two_j[0] + 2 * args.shells[0]) + 8
        with NativeBackend(args.library, data, 1.2, 2, max_spin, 256) as native:
            function = native.lib.sl2cfoam_vertex_range
            function.argtypes = [ct.POINTER(ct.c_int)] + [ct.c_int] * 12
            function.restype = ct.POINTER(Tensor)
            rows = []
            for cache in ("cold", "warm"):
                row = calculate(native, args.two_j[0], args.shells[0], args.two_i, args.method)
                rows.append(dict(row, cache=cache))
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=os.environ.get("SL2CFOAM_LIBRARY"))
    parser.add_argument("--tables", default=os.environ.get("SL2CFOAM_TABLES"))
    parser.add_argument("--two-j", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--shells", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--two-i", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="benchmarks/results/local.json")
    parser.add_argument("--method", choices=["full", "slice"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.library or not args.tables:
        parser.error("Set SL2CFOAM_LIBRARY and SL2CFOAM_TABLES")
    if args.repeats < 1 or args.threads < 1 or any(j < 1 for j in args.two_j):
        parser.error("Positive repetitions, threads and twice-spins are required")
    if any(dl < 0 or dl > 25 for dl in args.shells):
        parser.error("This reference comparison supports shells 0 through 25")
    if any(i < 0 or i % 2 or i > 2 * min(args.two_j) for i in args.two_i):
        parser.error("Boundary labels must be admissible even twice-intertwiners")
    if args.method:
        worker(args)
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    results = []
    for j in args.two_j:
        for dl in args.shells:
            for repeat in range(args.repeats):
                methods = ("full", "slice") if repeat % 2 == 0 else ("slice", "full")
                for method in methods:
                    with tempfile.TemporaryDirectory() as tmp:
                        child_output = Path(tmp) / "result.json"
                        command = [sys.executable, "-m", "benchmarks.compare_vertices",
                                   "--library", str(args.library), "--tables", str(args.tables),
                                   "--two-j", str(j), "--shells", str(dl), "--two-i",
                                   *map(str, args.two_i), "--method", method,
                                   "--output", str(child_output)]
                        subprocess.run(command, env=env, check=True, timeout=300)
                        for row in json.loads(child_output.read_text()):
                            results.append(dict(row, two_j=j, dl=dl, method=method, repeat=repeat))
                    print(f"j={j/2:g} DL={dl} {method} repetition {repeat+1}: done", flush=True)
            rows = [r for r in results if r["two_j"] == j and r["dl"] == dl]
            for row in rows:
                np.testing.assert_allclose(row["amplitudes"], rows[0]["amplitudes"], rtol=1e-10, atol=0)
            # Keep completed cases if a subsequent native calculation fails.
            cpu = next((line.split(":", 1)[1].strip() for line in
                        Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
            report = dict(schema=1, platform=platform.platform(), cpu=cpu,
                          cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
                          numpy=np.__version__, threads=args.threads, blas_threads=1,
                          gamma=1.2, accuracy="veryhigh", two_i=args.two_i,
                          backend=backend_identity(args.library),
                          script_sha256=digest(__file__),
                          tables={p.name: digest(p) for p in sorted(Path(args.tables).iterdir())
                                  if p.suffix in (".3j", ".6j")},
                          results=results)
            output.write_text(json.dumps(report, indent=2) + "\n")
    for j in args.two_j:
        for dl in args.shells:
            for cache in ("cold", "warm"):
                medians = {method: statistics.median(r["seconds"] for r in results
                           if (r["two_j"], r["dl"], r["cache"], r["method"]) == (j, dl, cache, method))
                           for method in ("full", "slice")}
                print(f"j={j/2:g} DL={dl} {cache}: {medians}; full/slice={medians['full']/medians['slice']:.3f}")


if __name__ == "__main__":
    main()
