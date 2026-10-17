from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qlg.amplitudes import delta4_sector, ring_amplitude, signed_sum
from qlg.analysis import collect, reference_value, relative_difference
from qlg.jobs import (atomic_json, cached_vertex, exclusive_lock, fingerprint,
                      load_config, prepare)
from qlg.spins import (delta4_bulk_spins, delta4_spins, intertwiner_range,
                       vertex_shape)
from qlg.topology import boundary, refinement_plan

ROOT = Path(__file__).resolve().parent.parent


class CoreTests(unittest.TestCase):
    def test_admissibility_and_half_integer_bulk(self):
        self.assertEqual(intertwiner_range(1, 1, 1, 1), (0, 2))
        self.assertEqual(intertwiner_range(1, 1, 1, 2), ())
        self.assertEqual(delta4_bulk_spins(1), (1, 3))
        self.assertEqual(delta4_bulk_spins(2), (0, 2, 4, 6))
        self.assertEqual(vertex_shape(delta4_spins(2, 6)), (3, 1, 1, 3, 3))

    def test_signed_contraction_against_explicit_sum(self):
        rng = np.random.default_rng(37)
        matrices = [rng.normal(size=(3, 3)) for _ in range(4)]
        expected = sum(matrices[0][a, b] * matrices[1][b, c] * matrices[2][c, d]
                       * matrices[3][d, a] for a, b, c, d in product(range(3), repeat=4))
        self.assertAlmostEqual(float(ring_amplitude(matrices)), expected)
        tensor = rng.normal(size=(2, 3, 3, 2, 2))
        indices = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
        matrices = [tensor[a, :, :, b, c] for a, b, c in np.array(indices).reshape(4, 3)]
        self.assertAlmostEqual(float(delta4_sector(tensor, indices)), float(ring_amplitude(matrices)))

    def test_cancellation_and_tiny_amplitudes(self):
        self.assertEqual(signed_sum([1e20, 1.0, -1e20])[0], 1.0)
        self.assertAlmostEqual(relative_difference(1e-38, 0.9e-38), 0.1)
        self.assertIsNone(relative_difference(0, 1e-38))
        self.assertEqual(relative_difference(0, 0), 0)

    def test_config_rejects_silent_mistakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for patch in ({"shells": [0, 0]}, {"shells": [True]}, {"gamma": -1},
                          {"two_j": 1.5}, {"two_js": [2]*10},
                          {"boundary": [{"name": "bad", "two_i": 1}]}):
                atomic_json(path, {"model": "delta4", **patch})
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_checkpoint_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.npz"
            identity = {"spins": [2]*10, "dl": 0}
            expected = np.arange(3**5).reshape((3,)*5).astype(float)
            first, _, reused = cached_vertex(path, identity, lambda: expected)
            self.assertFalse(reused)
            second, _, reused = cached_vertex(path, identity, lambda: self.fail("Recomputed checkpoint"))
            self.assertTrue(reused)
            np.testing.assert_array_equal(first, second)
            with self.assertRaises(ValueError):
                cached_vertex(path, {**identity, "dl": 1}, lambda: expected)
            with exclusive_lock(Path(tmp) / "lock"):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(Path(tmp) / "lock"):
                        pass

    def test_run_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "lib.so"
            library.write_bytes(b"identity only; never loaded")
            tables = root / "tables"
            tables.mkdir()
            for ext in ("3j", "6j"):
                (tables / f"test.{ext}").write_bytes(b"table")
            run = root / "run"
            m = prepare(ROOT / "configs/smoke.json", run, library, tables)
            self.assertEqual(prepare(ROOT / "configs/smoke.json", run, library, tables)["run_id"], m["run_id"])
            library.write_bytes(b"different library")
            with self.assertRaises(ValueError):
                prepare(ROOT / "configs/smoke.json", run, library, tables)

    def test_reference_report_and_missing_tasks(self):
        spec = load_config(ROOT / "configs/smoke.json")
        self.assertEqual(reference_value(spec, 15, "i_0"), 1.849656117029305e-37)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            atomic_json(run / "manifest.json", {"schema": 1, "run_id": "test", "spec": spec})
            for dl in (0, 1):
                values = {b["name"]: reference_value(spec, dl, b["name"]) for b in spec["boundary"]}
                atomic_json(run / f"dl_{dl:03d}" / "result.json",
                            {"run_id": "test", "dl": dl, "amplitudes": values,
                             "wall_seconds": 0.1, "max_rss_kb": 1024,
                             "bulk_sector_cancellation": dict.fromkeys(values, 1.0)})
            with self.assertRaises(ValueError):
                collect(run, reference=True)
            result = collect(run, allow_partial=True, reference=True)
            self.assertFalse(result["complete"])
            self.assertTrue(result["reference_pass"])
            self.assertEqual(result["missing_shells"], [2])
            self.assertTrue((run / "analysis/cutoff_convergence.png").exists())
            broken = json.loads((run / "dl_001/result.json").read_text())
            broken["amplitudes"]["i_0"] *= 2
            atomic_json(run / "dl_001/result.json", broken)
            self.assertFalse(collect(run, allow_partial=True, reference=True)["reference_pass"])

    def test_same_boundary_refinement_combinatorics(self):
        plan = refinement_plan(2)
        levels = plan["levels"]
        self.assertEqual([len(x["simplices"]) for x in levels], [1, 5, 9])
        self.assertEqual(levels[1]["internal_triangles"], 10)
        self.assertEqual(levels[1]["internal_tetrahedra"], 10)
        self.assertEqual(len(levels[0]["boundary_tetrahedra"]), 5)
        for level in levels:
            self.assertEqual(boundary(level["simplices"]), boundary(levels[0]["simplices"]))
        self.assertFalse(plan["amplitudes_implemented"])


if __name__ == "__main__":
    unittest.main()
