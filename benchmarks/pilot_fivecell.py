"""Checkpointed native vertex timing pilot for the regulated five-cell foam.

Samples labels across the ordered vertex set, including the largest output
tensor. Every completed tensor is stored in the same strict cache used by the
full amplitude calculation, so a later full run can reuse it.
"""

import argparse
import math
import os
from pathlib import Path
import tempfile

from benchmarks.block_scan import cache_spec, checkpoint
from benchmarks.check_cutoffs import native_data
from benchmarks.check_fivecell import layout, spin_assignments
from qlg.backend import NativeBackend
from qlg.jobs import atomic_json, exclusive_lock, fingerprint
from qlg.spins import vertex_shape


def required_labels(two_k, ordering):
    graph = layout(ordering=ordering)
    rows = spin_assignments(graph, two_k)
    needed = sorted({tuple(s[f] for f in n["faces"]) for s in rows
                     for n in graph["nodes"]})
    return rows, needed


def sample_indices(needed, samples):
    if not needed:
        return []
    points = {round(i * (len(needed) - 1) / max(1, samples - 1))
              for i in range(samples)}
    points.add(max(range(len(needed)),
                   key=lambda i: math.prod(vertex_shape(needed[i]))))
    return sorted(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Same cache directory as check_fivecell")
    parser.add_argument("--output", required=True)
    parser.add_argument("--two-k", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--dl", type=int, choices=range(13), required=True)
    parser.add_argument("--ordering", choices=("ascending", "descending"),
                        default="ascending")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    if args.samples < 1 or not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        parser.error("Require positive samples and OMP_NUM_THREADS")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("Require OPENBLAS_NUM_THREADS=1")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    identity = cache_spec()
    rows, labels = required_labels(args.two_k, args.ordering)
    selected = sample_indices(labels, args.samples)
    report = dict(status="running", two_k=args.two_k, dl=args.dl,
        ordering=args.ordering, samples_requested=args.samples,
        assignment_count=len(rows), distinct_ordered_vertices=len(labels),
        selected_indices=selected, cache_id=fingerprint(identity),
        total_output_tensor_bytes=sum(math.prod(vertex_shape(s)) * 8 for s in labels),
        measurements=[])
    with exclusive_lock(run / ".fivecell-pilot.lock"):
        atomic_json(args.output, report)
        with tempfile.TemporaryDirectory(prefix="qlg-fivecell-pilot-") as tmp:
            data = Path(tmp)
            native_data(data)
            max_two_spin = 4 * (args.two_k + 2 * args.dl) + 8
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2,
                               max(32, max_two_spin), 128) as backend:
                for i in selected:
                    spins = labels[i]
                    tensor, seconds, reused = checkpoint(
                        run, identity, spins, args.dl,
                        lambda: backend.vertex(spins, args.dl))
                    report["measurements"].append(dict(
                        index=i, spins=list(spins), shape=list(tensor.shape),
                        output_bytes=tensor.nbytes, seconds=seconds, reused=reused))
                    atomic_json(args.output, report)
                    print(f"DL={args.dl} K={args.two_k / 2:g} vertex {i + 1}/{len(labels)}:"
                          f" {seconds:.3f}s {'cached' if reused else 'new'}",
                          flush=True)
    report["status"] = "completed"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
