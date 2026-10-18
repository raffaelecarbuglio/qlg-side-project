"""Checks for custom full-sum complexes and fixed-path compression."""

from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import opt_einsum as oe

from benchmarks.check_cutoffs import native_data
from benchmarks.check_refinement import geometry_checks, intermediate_complex
from benchmarks.compression import compressed_contract, compression_network, compression_path
from benchmarks.refinement import build_block, complex_layout, expression, sector_factors
from qlg.backend import NativeBackend
from qlg.topology import edgewise_simplex, refine


class RefinementChecks(unittest.TestCase):
    def test_intermediate_geometry(self):
        vertices, parents = edgewise_simplex(2)
        parent_points = [np.array([[1, *vertices[i][1:]] for i in cell], dtype=float) for cell in parents]
        inverses = [np.linalg.inv(points) for points in parent_points]
        for steps, cells, faces in ((0, 16, 5), (1, 20, 8), (2, 25, 12), (3, 30, 16)):
            graph = intermediate_complex(steps)
            checks = geometry_checks(graph)
            self.assertEqual(checks["cells"], cells)
            self.assertEqual(checks["internal_faces"], faces)
            # Independent barycentric containment and volume partition checks.
            volumes = [0.] * len(parents)
            for cell in graph["cells"]:
                points = np.array([[1, *graph["vertices"][i][1:]] for i in cell], dtype=float)
                containing = [k for k, inverse in enumerate(inverses) if np.min(points @ inverse) >= -1e-13]
                self.assertEqual(len(containing), 1)
                volumes[containing[0]] += abs(np.linalg.det(points)) / 24
            np.testing.assert_allclose(volumes, [abs(np.linalg.det(p)) / 24 for p in parent_points],
                                       atol=1e-14, rtol=0)

    def test_builder_rejects_omitted_half_integer_sectors(self):
        cells = refine([tuple(range(5))], 0)
        graph = complex_layout(tuple(range(6)), cells)
        with self.assertRaisesRegex(ValueError, "half-integer"):
            build_block(None, graph=graph)

    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY"), "Native backend not configured")
    def test_full_twenty_cell_sum_and_fixed_path(self):
        # The pinned native library must not be reinitialized in one process.
        if os.environ.get("QLG_REFINEMENT_CHECK_CHILD") != "1":
            completed = subprocess.run(
                [sys.executable, "-m", "unittest",
                 "benchmarks.test_refinement.RefinementChecks.test_full_twenty_cell_sum_and_fixed_path"],
                env=dict(os.environ, QLG_REFINEMENT_CHECK_CHILD="1"),
                capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            return
        with tempfile.TemporaryDirectory(prefix="qlg-refinement-check-") as temporary:
            data = Path(temporary)
            native_data(data)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 60, 256) as backend:
                graph, arrays, inputs, _ = build_block(backend, graph=intermediate_complex(1))
                reference = float(oe.contract(expression(inputs), *arrays, optimize="greedy"))
                nf = len(graph["face_indices"])
                sector_arrays, sector_inputs = sector_factors(arrays, inputs, [0] * nf, nf)
                contract = oe.contract_expression(expression(sector_inputs),
                                                   *[a.shape for a in sector_arrays], optimize="greedy")
                sectors = [float(contract(*sector_factors(arrays, inputs, labels, nf)[0]))
                           for labels in product(range(2), repeat=nf)]
                self.assertLess(abs(math.fsum(sectors) / reference - 1), 1e-11)
                network = compression_network(arrays, inputs)
                original = [t.data.copy() for t in network]
                path = compression_path(network, 8)
                results = [compressed_contract(arrays, inputs, chi, network=network, path=path)
                           for chi in (16, 32)]
                self.assertEqual(results[0]["path"], results[1]["path"])
                for result in results:
                    estimate = result["sign"] * math.exp(result["log_absolute_value"])
                    self.assertLess(abs(estimate / reference - 1), 1e-10)
                for tensor, saved in zip(network, original):
                    np.testing.assert_array_equal(tensor.data, saved)


if __name__ == "__main__":
    unittest.main()
