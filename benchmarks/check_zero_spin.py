"""Isolate support for zero-spin inputs in two explicitly selected Y-map builds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from benchmarks.check_cutoffs import native_data
from qlg.backend import NativeBackend, backend_identity
from qlg.jobs import atomic_json, source_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", nargs="+", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--two-j", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.worker:
        with tempfile.TemporaryDirectory(prefix="qlg-zero-spin-") as temporary:
            data = Path(temporary)
            native_data(data)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 64, 128) as backend:
                value = backend.vertex((args.two_j,) * 10, 0)
                atomic_json(args.output, dict(status="completed", shape=list(value.shape),
                                             max_absolute_value=float(np.max(np.abs(value)))))
        return
    if not args.libraries:
        parser.error("Select at least one compiled library explicitly")
    report = dict(source=source_version(), gamma=1.2, accuracy="veryhigh", dl=0, results=[])
    for library in args.libraries:
        build = backend_identity(library)["build"]
        for two_j in (1, 0):
            with tempfile.TemporaryDirectory(prefix="qlg-zero-result-") as temporary:
                output = Path(temporary) / "result.json"
                command = [sys.executable, "-m", "benchmarks.check_zero_spin", "--worker",
                           "--two-j", str(two_j), "--output", str(output)]
                env = dict(os.environ, SL2CFOAM_LIBRARY=str(Path(library).resolve()),
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    row = json.loads(output.read_text())
                elif "dsmall with rho == 0 not implemented" in result.stderr + result.stdout:
                    row = dict(status="unsupported", reason="dsmall with rho == 0 not implemented")
                else:
                    raise RuntimeError(f"Unexpected native failure, exit {result.returncode}")
            report["results"].append(dict(rho_convention=build.get("rho_convention"),
                backend_source_revision=build.get("commit"), two_j=two_j, **row))
            atomic_json(args.output, report)
            print(build.get("rho_convention"), f"j={two_j/2:g}", row["status"], flush=True)


if __name__ == "__main__":
    main()
