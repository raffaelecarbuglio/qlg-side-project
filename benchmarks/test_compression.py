"""Optional benchmark checks: python -m unittest benchmarks.test_compression."""

from itertools import product
import math
import os
import unittest

import numpy as np
import opt_einsum as oe

from benchmarks.compression import (compressed_contract, fixed_sector,
                                     local_contract, parity_structure, prune_support)
from benchmarks.refinement import build_block, expression, sector_factors
from qlg.backend import NativeBackend


class CompressionChecks(unittest.TestCase):
    def test_exact_reduction_preserves_a_hyperindex_sum(self):
        arrays = [np.array([[1., 0., 2.], [3., 0., 4.]]), np.array([2., 5., 6.]),
                  np.array([[1., 2.], [4., 8.], [2., 3.]])]
        inputs = [(0, 1), (1,), (1, 2)]
        expected = sum(arrays[0][a, b] * arrays[1][b] * arrays[2][b, c]
                       for a, b, c in product(range(2), range(3), range(2)))
        reduced, indices, support = prune_support(arrays, inputs)
        self.assertEqual(support["removed_values"], 1)
        reduced, indices, _ = local_contract(reduced, indices, max_entries=100)
        self.assertEqual(float(oe.contract(expression(indices), *reduced)), expected)
        zero, _, support = prune_support([np.zeros((2, 3)), np.ones(3)], [(0, 1), (1,)])
        self.assertTrue(support["zero"])
        self.assertEqual(float(zero[0]), 0)

    def test_later_refinement_has_half_integer_parity_sectors(self):
        self.assertEqual(parity_structure(2)["independent_parity_labels"], 0)
        larger = parity_structure(4)
        self.assertEqual(larger["parity_constraint_rank"], 225)
        self.assertEqual(larger["independent_parity_labels"], 35)
        self.assertEqual(larger["generator_rank"], 35)

    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY"), "Native backend not configured")
    def test_native_sector_and_compression_against_full_block(self):
        with NativeBackend(os.environ["SL2CFOAM_LIBRARY"], os.environ["SL2CFOAM_TABLES"],
                           1.2, 2, 60, 256) as backend:
            graph, arrays, inputs, info = build_block(backend, 1, 0, 1)
            nf = len(graph["face_indices"])
            sector, indices = sector_factors(arrays, inputs, [1] * nf, nf)
            expected = float(oe.contract(expression(indices), *sector, optimize="greedy"))
            _, fixed, fixed_inputs, scale = fixed_sector(backend, 2, 0, 1)
            value = float(oe.contract(expression(fixed_inputs), *fixed, optimize="greedy"))
            self.assertEqual(np.sign(value), np.sign(expected))
            self.assertAlmostEqual(math.log(abs(value)) + scale,
                                   math.log(abs(expected)) + info["log_amplitude_scale"], places=11)
            reference = float(oe.contract(expression(inputs), *arrays, optimize="greedy"))
            result = compressed_contract(arrays, inputs, chi=32)
            estimate = result["sign"] * math.exp(result["log_absolute_value"])
            self.assertLess(abs(estimate / reference - 1), 1e-10)


if __name__ == "__main__":
    unittest.main()
