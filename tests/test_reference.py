"""Deterministic mathematical checks for the separate NumPy reference.

These tests do not assert Qwen model quality, latency, or VRAM reductions.
"""

import unittest

import numpy as np

from sixfinger_moe_prototype import ExpertPartition, dense_ffn


class TestPartition(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261104)
        h, m = 24, 96
        self.gate = rng.normal(0, h ** -0.5, (m, h)).astype(np.float32)
        self.up = rng.normal(0, h ** -0.5, (m, h)).astype(np.float32)
        self.down = rng.normal(0, m ** -0.5, (h, m)).astype(np.float32)
        calibration = rng.normal(size=(32, h)).astype(np.float32)
        self.test_data = rng.normal(size=(8, h)).astype(np.float32)
        self.partition = ExpertPartition.from_dense(
            self.gate, self.up, self.down, calibration,
            experts=6, shared_fraction=0.25,
        )

    def test_every_neuron_is_partitioned_exactly_once(self):
        ids = np.concatenate((self.partition.shared, *self.partition.routed))
        np.testing.assert_array_equal(np.sort(ids), np.arange(self.gate.shape[0]))

    def test_all_experts_match_dense(self):
        for x in self.test_data:
            expected = dense_ffn(x, self.gate, self.up, self.down)
            actual, info = self.partition.forward_one(x, routed_k=6, omitted="mean")
            self.assertEqual(info["active_fraction"], 1.0)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_zero_mode_with_no_experts_uses_only_shared(self):
        x = self.test_data[0]
        actual, info = self.partition.forward_one(x, routed_k=0, omitted="zero")
        shared = self.partition.shared
        expected = dense_ffn(x, self.gate[shared], self.up[shared], self.down[:, shared])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertLess(info["active_fraction"], 1.0)

    def test_invalid_routing_is_rejected(self):
        with self.assertRaises(ValueError):
            self.partition.forward_one(self.test_data[0], routed_k=99)


if __name__ == "__main__":
    unittest.main()
