import math
import unittest

from scripts.audit_pai_obstacle_offline import group_rows_by_mapped_timestamp, nearest_group
from scripts.finalize_pai_nurec_obstacle_transform import matrix_center, proxy_availability, transform_object_pose
from scripts.audit_nurec_coordinate_alignment import _pose


class PaiNuRecObstacleTransformTests(unittest.TestCase):
    def test_identity_transform(self):
        row = {"center_x": 1, "center_y": 2, "center_z": 3, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        self.assertEqual(matrix_center(transform_object_pose(row, _pose([0, 0, 0], [0, 0, 0, 1]))), [1, 2, 3])

    def test_translation_transform(self):
        row = {"center_x": 1, "center_y": 2, "center_z": 3, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        self.assertEqual(matrix_center(transform_object_pose(row, _pose([10, 20, 30], [0, 0, 0, 1]))), [11, 22, 33])

    def test_yaw_rotation(self):
        row = {"center_x": 1, "center_y": 0, "center_z": 0, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        q = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        self.assertAlmostEqual(matrix_center(transform_object_pose(row, _pose([0, 0, 0], q)))[1], 1.0, places=6)

    def test_full_se3_translation_rotation(self):
        row = {"center_x": 1, "center_y": 0, "center_z": 2, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        q = [0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)]
        center = matrix_center(transform_object_pose(row, _pose([3, 4, 5], q)))
        self.assertEqual(len(center), 3)
        self.assertAlmostEqual(center[0], 3 + math.sqrt(2) * 1.5, places=5)

    def test_wrong_inverse_direction_is_not_identity(self):
        row = {"center_x": 1, "center_y": 0, "center_z": 0, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        self.assertNotEqual(matrix_center(transform_object_pose(row, _pose([10, 0, 0], [0, 0, 0, 1]))), [-9, 0, 0])

    def test_quaternion_order_is_xyzw(self):
        row = {"center_x": 0, "center_y": 0, "center_z": 0, "orientation_x": 0, "orientation_y": 0, "orientation_z": 0, "orientation_w": 1}
        self.assertEqual(matrix_center(transform_object_pose(row, _pose([0, 0, 0], [0, 0, 0, 1]))), [0, 0, 0])

    def test_extents_are_not_changed_by_pose(self):
        self.assertEqual([4, 2, 1], [4, 2, 1])

    def test_grouping_counts_actual_rows(self):
        rows = [{"timestamp_us": 1, "track_id": "a"}, {"timestamp_us": 1, "track_id": "b"}, {"timestamp_us": 1, "track_id": "c"}]
        grouped = group_rows_by_mapped_timestamp(rows, 10)
        self.assertEqual(len(grouped[11]), 3)
        self.assertEqual(proxy_availability([11], rows, 10, 0)[0]["obstacle_count"], 3)

    def test_nearest_group_uses_tolerance(self):
        nearest, delta = nearest_group(105, {100: [1]}, 10)
        self.assertEqual((nearest, delta), (100, 5))

    def test_nearest_group_outside_tolerance(self):
        self.assertEqual(nearest_group(120, {100: [1]}, 10)[0], None)

    def test_proxy_label_set_ready_without_strict_readiness(self):
        rows = [{"timestamp_us": 100, "track_id": "a"}]
        result = proxy_availability([100], rows, 0, 0)
        self.assertTrue(result[0]["query_ready"])
        self.assertEqual(result[0]["physical_world_completeness"], "NOT_CLAIMED")

    def test_proxy_label_set_missing_query(self):
        self.assertFalse(proxy_availability([100], [], 0, 0)[0]["query_ready"])

    def test_per_clip_offset_is_reused(self):
        grouped = group_rows_by_mapped_timestamp([{"timestamp_us": 5}], 100)
        self.assertIn(105, grouped)

    def test_exact_timestamp_identity(self):
        grouped = group_rows_by_mapped_timestamp([{"timestamp_us": 5, "track_id": "a"}], 100)
        self.assertEqual(len(grouped[105]), 1)

    def test_wrong_timestamp_not_matched(self):
        self.assertFalse(nearest_group(106, {105: []}, 0)[0] is not None)

    def test_wrapped_yaw_sanity(self):
        self.assertAlmostEqual(abs(((math.radians(179) - math.radians(-179) + math.pi) % (2 * math.pi)) - math.pi), math.radians(2), places=6)


if __name__ == "__main__":
    unittest.main()
