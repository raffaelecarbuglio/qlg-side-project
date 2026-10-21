"""Shard the union of both five-cell vertex frames across independent jobs.

Each tensor is owned by exactly one shard. All workers write complete atomic
files to the same strict cache used by check_fivecell. Rerunning a shard
reuses its already completed tensors.
"""

import argparse
import os
from pathlib import Path
import tempfile
import time

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.check_cutoffs import native_data
from benchmarks.pilot_fivecell import required_labels
from qlg.backend import NativeBackend
from qlg.jobs import atomic_json, exclusive_lock, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--two-k", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--dl", type=int, choices=range(13), required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Require 0 <= shard < shards")
    if (not os.environ.get("OMP_NUM_THREADS", "").isdigit()
            or int(os.environ["OMP_NUM_THREADS"]) < 1
            or os.environ.get("OPENBLAS_NUM_THREADS") != "1"
            or os.environ.get("MKL_NUM_THREADS", "1") != "1"):
        parser.error("Require positive OMP_NUM_THREADS and single-threaded BLAS")
    args.run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    needed = set()
    frame_counts = {}
    for ordering in ("ascending", "descending"):
        rows, labels = required_labels(args.two_k, ordering)
        frame_counts[ordering] = dict(assignments=len(rows), vertices=len(labels))
        needed.update(labels)
    ordered = sorted(needed)
    selected = ordered[args.shard::args.shards]
    report = dict(status="running", two_k=args.two_k, dl=args.dl,
                  shard=args.shard, shards=args.shards,
                  frame_counts=frame_counts, union_vertices=len(ordered),
                  selected=len(selected), union_sha256=fingerprint(ordered),
                  cache_id=fingerprint(identity), computed=0, reused=0,
                  native_seconds=0., elapsed_seconds=0.)
    start = time.perf_counter()
    lock = args.run / f".prepare-{args.two_k}-{args.dl}-{args.shard}.lock"
    with exclusive_lock(lock):
        atomic_json(args.output, report)
        with tempfile.TemporaryDirectory(prefix="qlg-fivecell-prepare-") as tmp:
            data = Path(tmp)
            native_data(data)
            max_two_spin = 4 * (args.two_k + 2 * args.dl) + 8
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2,
                               max(32, max_two_spin), 128) as backend:
                for index, spins in enumerate(selected, 1):
                    _, seconds, reused = checkpoint(
                        args.run, identity, spins, args.dl,
                        lambda: backend.vertex(spins, args.dl))
                    report["reused" if reused else "computed"] += 1
                    report["native_seconds"] += 0 if reused else seconds
                    report["position"] = index
                    report["elapsed_seconds"] = time.perf_counter() - start
                    atomic_json(args.output, report)
                    print(f"K={args.two_k/2:g} DL={args.dl} shard={args.shard}: "
                          f"{index}/{len(selected)} {'cached' if reused else 'new'} "
                          f"{seconds:.3f}s", flush=True)
        report["status"] = "completed"
        report["elapsed_seconds"] = time.perf_counter() - start
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
