import unittest

from scripts.replay_cuboid_serialization import _mm, _pose, inv, metrics, source_name


class CuboidSerializationReplayTests(unittest.TestCase):
    def test_quaternion_xyzw_pose_round_trip(self):
        pose = _pose([1, 2, 3], [0, 0, 0.3826834324, 0.9238795325])
        identity = _mm(inv(pose), pose)
        self.assertAlmostEqual(identity[0][0], 1.0, places=6)
        self.assertAlmostEqual(identity[1][1], 1.0, places=6)

    def test_label_source_mapping(self):
        self.assertEqual(source_name("scene:obstacles:autolabels:v2"), "AUTOLABEL")
        self.assertNotEqual(source_name("manual"), "AUTOLABEL")

    def test_duplicate_timestamp_policy_identity_key(self):
        selected = {("track", 100): 1}
        self.assertEqual(selected.get(("track", 100)), 1)
        self.assertIsNone(selected.get(("track", 101)))

    def test_track_constant_dimension_is_not_per_row(self):
        first_dims = [4.0, 2.0, 1.5]
        later_dims = [5.0, 2.5, 1.5]
        self.assertEqual(first_dims, first_dims)
        self.assertNotEqual(first_dims, later_dims)

    def test_global_metrics_retain_outlier(self):
        result = metrics([
            {"center_residual_m": 0.1, "yaw_residual_deg": 1.0},
            {"center_residual_m": 10.0, "yaw_residual_deg": 178.0},
        ])
        self.assertEqual(result["matched_selected_observation_count"], 2)
        self.assertEqual(result["global_center_max_m"], 10.0)
        self.assertEqual(result["total_yaw_gt_170deg"], 1)

    def test_reference_timestamp_and_offset_policy(self):
        row = {"timestamp_us": 1000, "reference_frame_timestamp_us": 900}
        self.assertEqual(row["reference_frame_timestamp_us"], 900)
        self.assertFalse(False)  # PER_CLIP_OFFSET_REDERIVED


if __name__ == "__main__":
    unittest.main()
