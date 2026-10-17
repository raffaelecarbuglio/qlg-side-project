"""Immutable run descriptions and restartable, independent cutoff tasks."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
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

from . import __version__
from .amplitudes import delta4_sector, signed_sum
from .backend import NativeBackend, backend_identity
from .spins import (delta4_bulk_spins, delta4_spins, integer, vertex_bytes,
                    vertex_ranges, vertex_shape)

SCHEMA = 1
ACCURACY = {"normal": 0, "high": 1, "veryhigh": 2}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_digest():
    root = Path(__file__).parent
    return fingerprint({p.name: _file_digest(p) for p in sorted(root.glob("*.py"))})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def exclusive_lock(path):
    with open(path, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_config(path):
    spec = json.loads(Path(path).read_text())
    allowed = {"model", "gamma", "two_j", "boundary", "two_spins", "two_intertwiners",
               "shells", "accuracy", "max_tensor_mb", "memory_mb_per_thread"}
    unknown = spec.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    model = spec.get("model")
    if model not in ("delta4", "vertex"):
        raise ValueError("model must be delta4 or vertex")
    gamma = spec.get("gamma", 1.2)
    if type(gamma) not in (float, int) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    shells = spec.get("shells", [0, 1, 2])
    if not isinstance(shells, list) or not shells:
        raise ValueError("shells must be a nonempty list")
    for dl in shells:
        integer(dl, "shell cutoff")
        if dl > 50:
            raise ValueError("This interface supports shell cutoffs <= 50")
    if len(set(shells)) != len(shells):
        raise ValueError("Duplicate shell cutoffs")
    result = {"model": model, "gamma": float(gamma), "shells": sorted(shells),
              "accuracy": spec.get("accuracy", "veryhigh"),
              "max_tensor_mb": spec.get("max_tensor_mb", 512),
              "memory_mb_per_thread": spec.get("memory_mb_per_thread", 256)}
    if result["accuracy"] not in ACCURACY:
        raise ValueError("accuracy must be normal, high or veryhigh")
    for key in ("max_tensor_mb", "memory_mb_per_thread"):
        integer(result[key], key, 1)
    if model == "delta4":
        if set(spec) & {"two_spins", "two_intertwiners"}:
            raise ValueError("Use two_j and boundary for Delta4")
        j = integer(spec.get("two_j", 2), "two_j", 1)
        boundary = spec.get("boundary", [{"name": f"i_{i}", "two_i": 2*i}
                                          for i in range(j + 1)])
        if not isinstance(boundary, list) or not boundary:
            raise ValueError("boundary must be a nonempty list")
        states, names = [], set()
        for state in boundary:
            if not isinstance(state, dict) or set(state) != {"name", "two_i"}:
                raise ValueError("Each boundary state needs exactly name and two_i")
            name = state["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Boundary names must be nonempty and unique")
            names.add(name)
            labels = state["two_i"]
            if type(labels) is int:
                labels = [labels] * 12
            if not isinstance(labels, list) or len(labels) != 12:
                raise ValueError("two_i must be one label or twelve labels")
            for label in labels:
                integer(label, "two_i")
                if label % 2 or label > 2*j:
                    raise ValueError("Boundary intertwiners must be integer i in [0,2j]")
            states.append({"name": name, "two_i": labels})
        result.update(two_j=j, boundary=states)
    else:
        if set(spec) & {"two_j", "boundary"}:
            raise ValueError("Use two_spins and two_intertwiners for a vertex")
        spins = spec.get("two_spins")
        if not isinstance(spins, list) or len(spins) != 10:
            raise ValueError("two_spins must contain ten labels")
        for spin in spins:
            integer(spin, "twice-spin")
        ranges = vertex_ranges(spins)
        if any(not r for r in ranges):
            raise ValueError("Inadmissible vertex spins")
        labels = spec.get("two_intertwiners", [r[0] for r in ranges])
        if not isinstance(labels, list) or len(labels) != 5:
            raise ValueError("two_intertwiners must contain five labels")
        for label, r in zip(labels, ranges):
            integer(label, "twice-intertwiner")
            if label not in r:
                raise ValueError("Inadmissible boundary intertwiner")
        result.update(two_spins=spins, two_intertwiners=labels)
    return result


def spin_tasks(spec):
    if spec["model"] == "delta4":
        return [(x, delta4_spins(spec["two_j"], x))
                for x in delta4_bulk_spins(spec["two_j"])]
    return [(None, tuple(spec["two_spins"]))]


def estimate(spec):
    tasks = spin_tasks(spec)
    return {"cutoff_tasks": len(spec["shells"]),
            "vertices_per_cutoff": len(tasks),
            "largest_output_tensor_mb": max(vertex_bytes(s) for _, s in tasks) / 2**20,
            "all_output_tensors_mb": sum(vertex_bytes(s) for _, s in tasks)
                                     * len(spec["shells"]) / 2**20,
            "note": "Output tensors only. Native workspace, tables and runtime require a benchmark."}


def prepare(config_path, run_path, library, tables):
    spec = load_config(config_path)
    run = Path(run_path).resolve()
    lib = Path(library).expanduser().resolve(strict=True)
    table_dir = Path(tables).expanduser().resolve(strict=True)
    # Match setup.c, which requires and selects both kinds of lookup table.
    selected = []
    for ext in ("3j", "6j"):
        matches = sorted(table_dir.glob(f"*.{ext}"))
        if len(matches) != 1:
            raise ValueError(f"Need exactly one .{ext} table in {table_dir}")
        selected.append({"path": str(matches[0]), "sha256": _file_digest(matches[0])})
    if estimate(spec)["largest_output_tensor_mb"] > spec["max_tensor_mb"]:
        raise ValueError("Output tensor exceeds max_tensor_mb; inspect the estimate first")
    identity = {"schema": SCHEMA, "spec": spec, "backend": backend_identity(lib),
                "tables": selected, "package_version": __version__,
                "code_sha256": code_digest(), "numpy_version": np.__version__}
    manifest = {**identity, "run_id": fingerprint(identity), "library": str(lib),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": source_version(), "estimate": estimate(spec)}
    run.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run / ".prepare.lock"):
        path = run / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["run_id"] != manifest["run_id"]:
                raise ValueError("Run directory belongs to different inputs/backend; choose a new directory")
            return old
        for dl in spec["shells"]:
            task = run / f"dl_{dl:03d}"
            data = task / "native"
            data.mkdir(parents=True, exist_ok=True)
            for table in selected:
                link = data / Path(table["path"]).name
                if not link.exists():
                    link.symlink_to(table["path"])
            (task / "vertices").mkdir(exist_ok=True)
        (run / "logs").mkdir(exist_ok=True)
        atomic_json(path, manifest)
    return manifest


def _file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_version():
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def read_manifest(run):
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported run schema")
    return manifest


def cached_vertex(path, identity, compute):
    """Only a complete atomic archive is reusable; metadata and shape must agree."""
    key = fingerprint(identity)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity"]) != key:
                raise ValueError(f"Incompatible checkpoint: {path}")
            tensor = data["tensor"]
            seconds = float(data["seconds"])
        reused = True
    else:
        start = time.perf_counter()
        tensor = compute()
        seconds = time.perf_counter() - start
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, tensor=tensor, identity=key, seconds=seconds)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        reused = False
    if tensor.shape != vertex_shape(identity["spins"]) or not np.isfinite(tensor).all():
        raise ValueError(f"Invalid tensor checkpoint: {path}")
    return tensor, seconds, reused


def worker(run_path, dl):
    run = Path(run_path).resolve()
    manifest = read_manifest(run)
    spec = manifest["spec"]
    if dl not in spec["shells"]:
        raise ValueError("Cutoff is not in this run's shell list")
    task = run / f"dl_{dl:03d}"
    with exclusive_lock(task / ".worker.lock"):
        if manifest["code_sha256"] != code_digest() or manifest["numpy_version"] != np.__version__:
            raise ValueError("Code or NumPy changed since prepare; use a new run directory")
        if (task / "result.json").exists():
            result = json.loads((task / "result.json").read_text())
            if result["run_id"] != manifest["run_id"]:
                raise ValueError("Incompatible result")
            print(f"DL={dl}: already complete", flush=True)
            return result
        if backend_identity(manifest["library"]) != manifest["backend"]:
            raise ValueError("Native library changed since prepare; use a new run directory")
        for table in manifest["tables"]:
            if _file_digest(table["path"]) != table["sha256"]:
                raise ValueError("Lookup table changed since prepare")
        tasks = spin_tasks(spec)
        # Conservative range for WIGXJPF's factorial tables, including auxiliary labels.
        max_two_spin = 4 * (max(max(s) for _, s in tasks) + 2*dl) + 8
        start = time.perf_counter()
        contributions = []
        timings = []
        with NativeBackend(manifest["library"], task / "native", spec["gamma"],
                           ACCURACY[spec["accuracy"]], max_two_spin,
                           spec["memory_mb_per_thread"]) as backend:
            for x, spins in tasks:
                identity = {"run_id": manifest["run_id"], "dl": dl, "spins": spins}
                path = task / "vertices" / ("vertex.npz" if x is None else f"x_{x:04d}.npz")
                tensor, seconds, reused = cached_vertex(path, identity,
                                                        lambda: backend.vertex(spins, dl))
                timings.append({"two_x": x, "seconds": seconds, "reused": reused,
                                "shape": list(tensor.shape), "bytes": tensor.nbytes})
                if spec["model"] == "delta4":
                    values = {b["name"]: float((x + 1) * delta4_sector(
                        tensor, [label // 2 for label in b["two_i"]])) for b in spec["boundary"]}
                else:
                    ranges = vertex_ranges(spins)
                    indices = [r.index(i) for r, i in zip(ranges, spec["two_intertwiners"])]
                    values = {"vertex": float(tensor[tuple(reversed(indices))])}
                if not all(math.isfinite(v) for v in values.values()):
                    raise FloatingPointError("Non-finite contracted amplitude")
                contributions.append({"two_x": x, "values": values})
                print(f"DL={dl} two_x={x}: {'cached' if reused else 'computed'} ({seconds:.3f}s)", flush=True)
        totals, cancellations = {}, {}
        for name in contributions[0]["values"]:
            totals[name], cancellations[name] = signed_sum(c["values"][name] for c in contributions)
        result = {"schema": SCHEMA, "run_id": manifest["run_id"], "dl": dl,
                  "amplitudes": totals, "bulk_sector_cancellation": cancellations,
                  "contributions": contributions, "vertices": timings,
                  "wall_seconds": time.perf_counter() - start,
                  "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "host": platform.node(), "python": sys.version, "numpy": np.__version__,
                  "source": source_version(),
                  "threads": {k: os.environ.get(k) for k in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        atomic_json(task / "result.json", result)
        return result
