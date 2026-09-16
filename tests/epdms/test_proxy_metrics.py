"""Unit tests for proxy metrics composite formulas and edge cases."""

import unittest
from tools.epdms.proxy_metrics import compute_nurec_safety_proxy_v1_composite


class TestProxyMetrics(unittest.TestCase):
    def test_perfect_driving(self):
        score = compute_nurec_safety_proxy_v1_composite(
            cf=1.0, dac=1.0, ttc=1.0, ep_gt=1.0, fc=1.0
        )
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_collision_gate_zeroes_score(self):
        # When collision occurs (CF=0), total score must be 0
        score = compute_nurec_safety_proxy_v1_composite(
            cf=0.0, dac=1.0, ttc=1.0, ep_gt=1.0, fc=1.0
        )
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_offroad_gate_zeroes_score(self):
        # When off-road occurs (DAC=0), total score must be 0
        score = compute_nurec_safety_proxy_v1_composite(
            cf=1.0, dac=0.0, ttc=1.0, ep_gt=1.0, fc=1.0
        )
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_comfort_failure_partial_penalty(self):
        # Uncomfortable driving (FC=0) penalizes 2/12 = 1/6
        score = compute_nurec_safety_proxy_v1_composite(
            cf=1.0, dac=1.0, ttc=1.0, ep_gt=1.0, fc=0.0
        )
        # (5*1 + 5*1 + 2*0) / 12 = 10 / 12 = 0.8333...
        self.assertAlmostEqual(score, 10.0 / 12.0, places=4)


if __name__ == "__main__":
    unittest.main()
