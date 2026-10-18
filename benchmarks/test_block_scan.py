"""Real EPRL checks for process/thread agreement and interruption recovery."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from benchmarks.block_scan import required_spins
from qlg.backend import digest


class BlockScanChecks(unittest.TestCase):
    def test_complete_vertex_sets(self):
        for k, count in ((1, 32), (2, 243), (3, 574)):
            spins = required_spins(k)
            self.assertEqual(len(spins), count)
            self.assertEqual(len(set(spins)), count)
            if k > 1:
                self.assertTrue(set(required_spins(k - 1)) < set(spins))

    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY"), "Native backend not configured")
    def test_threads_processes_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix="qlg-block-check-") as temporary:
            directory = Path(temporary)
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            reports, runs = [], []
            for processes, threads in ((1, 1), (1, 2), (2, 1)):
                run = directory / f"p{processes}t{threads}"
                output = directory / f"{run.name}.json"
                command = [sys.executable, "-m", "benchmarks.block_scan", "--run", str(run),
                           "--cutoff", "1", "--dl", "0", "--processes", str(processes),
                           "--threads", str(threads), "--output", str(output)]
                subprocess.run(command, env=env, check=True, capture_output=True, timeout=60)
                reports.append(json.loads(output.read_text())["results"][0])
                runs.append(run)
            # A direct earlier full-sum result, computed without these checkpoints.
            self.assertAlmostEqual(reports[0]["states"][1]["log_absolute_amplitude"],
                                   -350.95724075368616, places=10)
            for run, report in zip(runs, reports):
                self.assertEqual(sum(w["computed"] for w in report["workers"]), 32)
                for state, reference in zip(report["states"], reports[0]["states"]):
                    self.assertEqual(state["sign"], reference["sign"])
                    self.assertLess(abs(np.expm1(state["log_absolute_amplitude"]
                                                - reference["log_absolute_amplitude"])), 1e-11)
                for checkpoint in run.glob("dl_*/*.npz"):
                    with np.load(checkpoint) as actual, np.load(runs[0] / "dl_000" / checkpoint.name) as ref:
                        np.testing.assert_allclose(actual["tensor"], ref["tensor"], rtol=1e-11, atol=1e-25)
            # Same entries reused after changing both process and thread counts.
            command[command.index("--run") + 1] = str(runs[0])
            subprocess.run(command, env=env, check=True, capture_output=True, timeout=60)
            restarted = json.loads(output.read_text())["results"][0]
            self.assertEqual(sum(w["computed"] for w in restarted["workers"]), 0)
            self.assertEqual(sum(w["reused"] for w in restarted["workers"]), 32)

    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY"), "Native backend not configured")
    def test_killed_preparation_resumes_completed_vertices(self):
        with tempfile.TemporaryDirectory(prefix="qlg-interrupt-check-") as temporary:
            run, output = Path(temporary) / "cache", Path(temporary) / "result.json"
            command = [sys.executable, "-m", "benchmarks.block_scan", "--run", str(run),
                       "--cutoff", "2", "--dl", "2", "--prepare-only", "--output", str(output)]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            child = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 30
                while len(list(run.glob("dl_*/*.npz"))) < 3 and child.poll() is None:
                    if time.monotonic() > deadline:
                        self.fail("No completed vertex before interruption deadline")
                    time.sleep(.02)
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=15)
                self.assertNotEqual(child.returncode, 0)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            saved = {p: digest(p) for p in run.glob("dl_*/*.npz")}
            self.assertGreaterEqual(len(saved), 3)
            self.assertLess(len(saved), 243)
            subprocess.run(command, env=env, check=True, capture_output=True, timeout=120)
            row = json.loads(output.read_text())["results"][0]
            self.assertEqual(sum(w["reused"] for w in row["workers"]), len(saved))
            self.assertEqual(sum(w["computed"] for w in row["workers"]), 243 - len(saved))
            self.assertTrue(all(digest(p) == checksum for p, checksum in saved.items()))
            self.assertEqual(len(list(run.glob("dl_*/*.npz"))), 243)


if __name__ == "__main__":
    unittest.main()
