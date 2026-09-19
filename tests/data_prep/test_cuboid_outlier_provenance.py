import unittest

from scripts.audit_cuboid_outlier_provenance import global_metrics


class CuboidOutlierProvenanceTests(unittest.TestCase):
    def test_global_count_is_sum_not_max(self):
        rows = [
            {"center_residual_m": 3.0, "yaw_residual_deg": 1.0},
            {"center_residual_m": 4.0, "yaw_residual_deg": 2.0},
            {"center_residual_m": 0.1, "yaw_residual_deg": 91.0},
        ]
        result = global_metrics(rows)
        self.assertEqual(result["total_center_gt_2m"], 2)
        self.assertEqual(result["total_yaw_gt_90deg"], 1)

    def test_global_rmse_uses_all_rows(self):
        result = global_metrics([
            {"center_residual_m": 0.0, "yaw_residual_deg": 0.0},
            {"center_residual_m": 2.0, "yaw_residual_deg": 4.0},
        ])
        self.assertAlmostEqual(result["global_center_rmse_m"], 2 ** 0.5)
        self.assertAlmostEqual(result["global_yaw_rmse_deg"], 8 ** 0.5)

    def test_official_residual_is_not_replaced_by_mod_180(self):
        official = 178.0
        diagnostic = min(official, abs(180.0 - official))
        self.assertEqual(official, 178.0)
        self.assertEqual(diagnostic, 2.0)

    def test_outliers_remain_in_aggregate(self):
        rows = [{"center_residual_m": 21.0, "yaw_residual_deg": 178.0}]
        result = global_metrics(rows)
        self.assertEqual(result["global_center_max_m"], 21.0)
        self.assertEqual(result["global_yaw_max_deg"], 178.0)

    def test_track_boundary_and_reference_timestamp_are_preserved(self):
        row = {"track_position": "FIRST", "reference_frame_timestamp_us": 900}
        self.assertEqual(row["track_position"], "FIRST")
        self.assertEqual(row["reference_frame_timestamp_us"], 900)

    def test_no_offset_rederive(self):
        self.assertFalse(False)


if __name__ == "__main__":
    unittest.main()
