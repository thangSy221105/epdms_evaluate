import math
import unittest

import numpy as np

from tools.epdms.map_transform import (
    MapTransformError,
    interpolate_rig_world_pose,
    invert_se3,
    transform_polygon_to_ego_t0,
)


def pose(x=0.0, y=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class NuRecMapTransformTests(unittest.TestCase):
    def test_identity_transform_preserves_polygon(self):
        polygon = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        np.testing.assert_allclose(transform_polygon_to_ego_t0(polygon, np.eye(4)), polygon)

    def test_translation_is_inverted_to_ego_origin(self):
        world_to_ego = invert_se3(pose(x=10.0, y=-2.0))
        polygon = np.asarray([[10.0, -2.0], [12.0, -2.0], [12.0, 0.0]])
        np.testing.assert_allclose(transform_polygon_to_ego_t0(polygon, world_to_ego), [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], atol=1e-8)

    def test_full_se3_rotation_and_translation(self):
        world_to_ego = invert_se3(pose(x=10.0, y=0.0, yaw=math.pi / 2.0))
        polygon = np.asarray([[10.0, 0.0], [10.0, 2.0], [8.0, 2.0]])
        np.testing.assert_allclose(transform_polygon_to_ego_t0(polygon, world_to_ego), [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], atol=1e-8)

    def test_pose_interpolation_uses_se3(self):
        interpolated = interpolate_rig_world_pose([(0, pose()), (10, pose(x=10.0, yaw=math.pi))], 5)
        self.assertAlmostEqual(interpolated[0][3], 5.0)
        self.assertAlmostEqual(interpolated[1][0], 1.0, places=6)

    def test_out_of_range_is_fail_closed(self):
        with self.assertRaises(MapTransformError):
            interpolate_rig_world_pose([(10, pose()), (20, pose(x=1.0))], 9)

    def test_no_fitted_correction_is_represented_by_direct_inverse(self):
        raw = np.asarray(pose(x=4.0, y=3.0, yaw=0.2))
        rebased = invert_se3(raw) @ raw
        np.testing.assert_allclose(rebased, np.eye(4), atol=1e-8)


if __name__ == "__main__":
    unittest.main()

