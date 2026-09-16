"""Unit tests for kinematics and comfort evaluation."""

import unittest
import numpy as np
from tools.epdms.kinematics_numpy import compute_kinematics, ComfortThresholds
from tools.epdms.coordinates import derive_heading_from_xy


class TestKinematics(unittest.TestCase):
    def test_smooth_straight_motion(self):
        # 41 points at 10m/s straight line along x axis
        dt = 0.1
        t = np.arange(41) * dt
        x = 10.0 * t
        y = np.zeros_like(x)
        headings = derive_heading_from_xy(x, y)

        kin = compute_kinematics(x, y, headings, dt=dt)
        self.assertTrue(kin.comfort_pass)
        self.assertAlmostEqual(kin.max_abs_lateral_accel, 0.0, places=3)
        self.assertAlmostEqual(kin.max_abs_yaw_rate, 0.0, places=3)

    def test_extreme_braking_violation(self):
        # Sudden stop from 20m/s to 0m/s in 0.2s (accel ~ -10 m/s^2)
        dt = 0.1
        t = np.arange(41) * dt
        # First 10 steps speed 20, then abruptly drops to 0
        v = np.array([20.0] * 10 + [0.0] * 31)
        x = np.cumsum(v * dt)
        y = np.zeros_like(x)
        headings = derive_heading_from_xy(x, y)

        kin = compute_kinematics(x, y, headings, dt=dt)
        self.assertFalse(kin.comfort_pass)
        self.assertTrue(any("braking" in r or "jerk" in r for r in kin.failure_reasons))


if __name__ == "__main__":
    unittest.main()
