"""Measure exact zero-support trimming on the full K=2, 16-simplex block."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import opt_einsum as oe
import numpy as np

from benchmarks.compare_refinement import conditioned_methods, measure, search
from benchmarks.compression import prune_support
from benchmarks.refinement import build_block, expression
from qlg.backend import NativeBackend, backend_identity, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/support-local.json")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"Set {key}=1 before starting Python")
    with tempfile.TemporaryDirectory(prefix="qlg-support-") as temporary:
        data = Path(temporary)
        for table in Path(os.environ["SL2CFOAM_TABLES"]).iterdir():
            if table.suffix in (".3j", ".6j"):
                (data / table.name).symlink_to(table.resolve())
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 60, 256) as backend:
            _, arrays, inputs, preparation = build_block(backend, 2, 0, 1)
            before = time.perf_counter()
            trimmed, indices, support = prune_support(arrays, inputs)
            support["seconds"] = time.perf_counter() - before
            results = []
            for name, factors, inds in (("padded", arrays, inputs), ("trimmed", trimmed, indices)):
                for method in ("greedy", "random-greedy-64"):
                    path, info = search(inds, [a.shape for a in factors], method)
                    function = oe.contract_expression(expression(inds), *[a.shape for a in factors], optimize=path)
                    row = dict(kind=name, method=method, factor_bytes=sum(a.nbytes for a in factors),
                               **info, **measure(lambda: function(*factors), args.repeats))
                    results.append(row)
            for row in results:
                np.testing.assert_allclose(row["value"], results[0]["value"], rtol=5e-12, atol=0)
            conditioned = conditioned_methods(trimmed, indices, 5, 3, args.repeats)
            for method, row in conditioned.items():
                np.testing.assert_allclose(row["value"], results[0]["value"], rtol=5e-12, atol=0)
                results.append(dict(kind="trimmed", method=method,
                                    factor_bytes=sum(a.nbytes for a in trimmed), **row))
    report = dict(schema=1, gamma=1.2, boundary_j=1, boundary_i=1, K=2, dl=0,
                  preparation=preparation, exact_support=support,
                  backend=backend_identity(os.environ["SL2CFOAM_LIBRARY"]),
                  source_sha256={p: digest(Path(__file__).parents[1] / p) for p in
                                 ("benchmarks/compare_support.py", "benchmarks/compression.py",
                                  "benchmarks/refinement.py", "benchmarks/compare_refinement.py")},
                  results=results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(row["kind"], row["method"], row["median_seconds"], row["factor_bytes"])


if __name__ == "__main__":
    main()
