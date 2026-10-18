"""Boundary pullback, exact BF gluing, and finite-spin completeness checks."""

from itertools import combinations, product
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from benchmarks.boundary_embedding import (check_embedding, coarse_boundary, embedding,
                                           routed_boundary)
from benchmarks.check_cutoffs import native_data
from benchmarks.matched_boundary import amplitude, spin_domains
from benchmarks.refinement import BasisMaps, complex_layout
from qlg.backend import NativeBackend
from qlg.spins import intertwiner_range
from qlg.topology import refine


class MatchedBoundaryChecks(unittest.TestCase):
    def test_does_not_claim_a_finite_bound_on_one_to_five(self):
        graph = complex_layout(tuple(range(6)), refine([tuple(range(5))], 0))
        outside = {f for t in graph["boundary_tetrahedra"]
                   for f in combinations(t, 3)}
        with self.assertRaisesRegex(ValueError, "No proved finite spin bound"):
            spin_domains(graph, {f: 2 for f in outside})

    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY"), "Native backend not configured")
    def test_boundary_pullbacks_and_full_bf_functionals(self):
        # One native instance per process; isolate this from other native suites.
        if os.environ.get("QLG_MATCHED_CHECK_CHILD") != "1":
            completed = subprocess.run(
                [sys.executable, "-m", "unittest",
                 "benchmarks.test_matched_boundary.MatchedBoundaryChecks.test_boundary_pullbacks_and_full_bf_functionals"],
                env=dict(os.environ, QLG_MATCHED_CHECK_CHILD="1"), capture_output=True,
                text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            return
        with tempfile.TemporaryDirectory(prefix="qlg-boundary-check-") as temporary:
            data = Path(temporary)
            native_data(data)
            with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], data, 1.2, 2, 128, 128) as backend:
                maps = BasisMaps(backend)
                for two_j in (1, 2):
                    coarse, cs, _ = amplitude(backend, maps, coarse_boundary(), two_j, None, data, {}, bf=True)
                    self.assertEqual(coarse.size, (two_j + 1)**5)
                    for kind, reverse in product(("split", "edgewise"), (False, True)):
                        spec = routed_boundary(kind, reverse)
                        checks = check_embedding(spec, maps, two_j)
                        self.assertLess(checks["max_absolute_error"], 1e-13)
                        self.assertEqual(checks["max_inherited_triangle_area_error"], 0)
                        fine, fs, info = amplitude(backend, maps, spec, two_j, None, data, {}, bf=True)
                        np.testing.assert_allclose(fine * math.exp(fs - cs), coarse, rtol=2e-11,
                                                   atol=1e-12 * np.max(np.abs(coarse)))
                        if two_j == 1 and kind == "edgewise" and reverse:
                            self.assertGreater(info["half_integer_internal_values"], 0)
                            # Independently enumerate the surviving global spin assignments.
                            graph = complex_layout(spec["vertices"], spec["cells"])
                            fixed = embedding(spec, maps, two_j)["spins"]
                            domains, _, initial = spin_domains(graph, fixed)
                            faces = tuple(graph["face_indices"])
                            tetrahedra = {t for node in graph["nodes"] for t in node["tetrahedra"]}
                            for values in product(*(initial[f] for f in faces)):
                                spins = dict(fixed)
                                spins.update(zip(faces, values))
                                if all(intertwiner_range(*(spins[f] for f in combinations(t, 3))) for t in tetrahedra):
                                    self.assertTrue(all(j in domains[f] for f, j in zip(faces, values)))


if __name__ == "__main__":
    unittest.main()
