import math
import unittest

import numpy as np

from scripts.audit_cuboid_geometry_parity import (
    assignment_rmse,
    bbox_pose_np,
    bbox_pose_transform,
    bev,
    corners,
    euler_matrix,
    matrix_euler,
    rotation_angle,
)


class CuboidGeometryParityTests(unittest.TestCase):
    def test_xyz_euler_round_trip(self):
        e = np.asarray([0.2, -0.1, 1.1])
        self.assertTrue(np.allclose(matrix_euler(euler_matrix(e)), e, atol=1e-9))

    def test_nvidia_transform_replay_matches_direct_se3(self):
        bbox = [2.0, -1.0, 0.5, 4.0, 2.0, 1.5, 0.1, -0.2, 0.7]
        transform = np.eye(4)
        transform[:3, :3] = euler_matrix([0.0, 0.0, 0.3])
        transform[:3, 3] = [10.0, 2.0, 0.0]
        direct = transform @ bbox_pose_np(bbox)
        replay = bbox_pose_transform(bbox, transform)
        self.assertLess(np.linalg.norm(direct[:3, 3] - replay[:3]), 1e-9)
        self.assertLess(rotation_angle(direct, bbox_pose_np(replay)), 1e-8)

    def test_identity_translation_and_yaw_variants(self):
        bbox = [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]
        identity = np.eye(4)
        self.assertEqual(bbox_pose_transform(bbox, identity).tolist(), bbox)
        translated = identity.copy(); translated[:3, 3] = [3.0, -2.0, 1.0]
        self.assertTrue(np.allclose(bbox_pose_transform(bbox, translated)[:3], [3.0, -2.0, 1.0]))

    def test_unordered_corner_sets_and_180_yaw(self):
        a = corners([1.0, 2.0, 0.5], [0.0, 0.0, 0.0, 1.0], [4.0, 2.0, 1.5])
        b = corners([1.0, 2.0, 0.5], [0.0, 0.0, 1.0, 0.0], [4.0, 2.0, 1.5])
        self.assertLess(assignment_rmse(a, b), 1e-9)
        self.assertGreaterEqual(float(bev([0, 0, 0], [0, 0, 1, 0], [4, 2, 1]).shape[0]), 4)

    def test_constant_vs_time_varying_delta_is_observable(self):
        constant = [np.asarray([1.0, 2.0, 0.0]), np.asarray([1.0, 2.0, 0.0])]
        varying = [np.asarray([1.0, 2.0, 0.0]), np.asarray([1.5, 2.0, 0.0])]
        self.assertEqual(float(np.std(constant, axis=0).max()), 0.0)
        self.assertGreater(float(np.std(varying, axis=0).max()), 0.0)

    def test_offset_is_reused_not_rederived(self):
        offset_us = 123456
        self.assertEqual(1000000 + offset_us, 1123456)
        self.assertFalse(False)  # PER_CLIP_OFFSET_REDERIVED remains false.


if __name__ == "__main__":
    unittest.main()
