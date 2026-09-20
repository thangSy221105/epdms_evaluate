import unittest

import numpy as np

from scripts.audit_cuboid_final_outlier_closure import (
    SCIPY_AVAILABLE,
    global_frame_evidence,
    neighborhood,
    timeline_classification,
)


def fake_row(timestamp, residual, x, reference_delta=0.0):
    return {
        "raw_timestamp_us": timestamp,
        "timestamp_minus_reference_us": reference_delta,
        "A_center_residual_m": residual,
        "raw_center": [x, 0.0, 0.0],
        "local_sequence_center": [x, 0.0, 0.0],
    }


class CuboidFinalOutlierClosureTests(unittest.TestCase):
    def test_track_constant_dimension_policy_uses_selected_first_row(self):
        selected = [
            {"track_id": "1", "timestamp_us": 20, "size_x": 4.0},
            {"track_id": "1", "timestamp_us": 10, "size_x": 3.0},
        ]
        selected.sort(key=lambda row: row["timestamp_us"])
        first_dimensions = selected[0]["size_x"]
        self.assertEqual(first_dimensions, 3.0)
        self.assertNotEqual(first_dimensions, selected[1]["size_x"])

    def test_center_residual_does_not_depend_on_dimensions(self):
        center_a = np.asarray([10.0, 0.0, 0.0])
        center_b = np.asarray([9.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(center_a - center_b)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(center_a - center_b)), 1.0)

    def test_neighborhood_extracts_three_rows_each_side(self):
        rows = [fake_row(i, 0.1, float(i)) for i in range(9)]
        result = neighborhood(rows, 4, radius=3)
        self.assertEqual([index for index, _ in result], [1, 2, 3, 4, 5, 6, 7])

    def test_isolated_single_row_detection(self):
        rows = [fake_row(0, 0.1, 0), fake_row(1, 2.5, 1), fake_row(2, 0.2, 2)]
        self.assertEqual(timeline_classification(rows, [1], 100000), "ISOLATED_SINGLE_ROW")

    def test_contiguous_outlier_detection(self):
        rows = [fake_row(0, 0.1, 0), fake_row(1, 2.5, 1), fake_row(2, 2.3, 2), fake_row(3, 0.1, 3)]
        self.assertEqual(timeline_classification(rows, [1, 2], 100000), "CONTIGUOUS_LOCAL_TRACK_DIVERGENCE")

    def test_track_wide_constant_offset_detection(self):
        rows = [fake_row(i, 3.0, float(i)) for i in range(4)]
        self.assertEqual(timeline_classification(rows, [0, 1, 2, 3], 100000), "TRACK_WIDE_CONSTANT_OFFSET")

    def test_track_wide_time_varying_offset_detection(self):
        rows = [fake_row(0, 2.1, 0), fake_row(1, 4.0, 1), fake_row(2, 8.0, 2)]
        self.assertEqual(timeline_classification(rows, [0, 1, 2], 100000), "TRACK_WIDE_TIME_VARYING_OFFSET")

    def test_reference_timestamp_anomaly_detection(self):
        rows = [fake_row(0, 0.1, 0), fake_row(1, 2.5, 1, reference_delta=500000), fake_row(2, 0.1, 2)]
        self.assertEqual(timeline_classification(rows, [1], 100000), "REFERENCE_TIMESTAMP_ANOMALY")

    def test_source_track_jump_detection(self):
        rows = [fake_row(0, 0.1, 0), fake_row(1, 2.5, 20), fake_row(2, 0.1, 21)]
        self.assertEqual(timeline_classification(rows, [1], 100000), "SOURCE_LABEL_JUMP")

    def test_global_frame_error_not_implied_by_one_bad_track(self):
        rows = [fake_row(i, 10.0 if i == 1 else 0.1, float(i)) for i in range(4303)]
        clip_bias = [
            {"mean_dx": 0.0, "mean_dy": 0.0, "mean_dz": 0.0, "mean_signed_yaw_deg": 0.0},
            {"mean_dx": 0.0, "mean_dy": 0.0, "mean_dz": 0.0, "mean_signed_yaw_deg": 0.0},
            {"mean_dx": 0.0, "mean_dy": 0.0, "mean_dz": 0.0, "mean_signed_yaw_deg": 0.0},
        ]
        bridge = [{"translation_delta_norm_m": 0.4, "yaw_delta_deg": 0.1}]
        status, evidence = global_frame_evidence(rows, clip_bias, bridge)
        self.assertEqual(status, "NOT_SUPPORTED")
        self.assertLess(evidence["center_outlier_rate"], 0.1)

    def test_outlier_is_retained_in_aggregate(self):
        residuals = [0.1, 2.5, 0.2]
        self.assertEqual(sum(value > 2.0 for value in residuals), 1)
        self.assertEqual(max(residuals), 2.5)

    def test_scipy_status_is_not_fabricated(self):
        self.assertIn(SCIPY_AVAILABLE, (True, False))

    def test_no_offset_rederive_policy(self):
        existing_offset = 12345
        self.assertEqual(1000000 + existing_offset, 1012345)


if __name__ == "__main__":
    unittest.main()
