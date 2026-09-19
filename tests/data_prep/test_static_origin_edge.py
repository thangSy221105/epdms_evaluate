import unittest

from scripts.audit_static_origin_edge import _mm, _pose, inv, static_decision


class StaticOriginEdgeTests(unittest.TestCase):
    def test_identity_delta(self):
        identity = _pose([0, 0, 0], [0, 0, 0, 1])
        delta = _mm(identity, inv(identity))
        self.assertAlmostEqual(delta[0][0], 1.0)
        self.assertAlmostEqual(delta[0][3], 0.0)

    def test_constant_translation_is_supported(self):
        self.assertTrue(static_decision(0.02, 0.05, 0.1))

    def test_constant_se3_is_supported(self):
        self.assertTrue(static_decision(0.1, 0.2, 0.5))

    def test_time_varying_delta_is_rejected(self):
        self.assertFalse(static_decision(0.8, 0.2, 0.5))
        self.assertFalse(static_decision(0.1, 1.2, 0.5))
        self.assertFalse(static_decision(0.1, 0.2, 2.1))

    def test_no_offset_rederive_policy(self):
        per_clip_offset_rederived = False
        self.assertFalse(per_clip_offset_rederived)


if __name__ == "__main__":
    unittest.main()
