import unittest

import numpy as np

from tools.epdms.lane_metrics import (
    LaneCenterline,
    LK_AMBIGUOUS,
    LK_READY,
    compute_lane_direction_alignment_diagnostic,
    compute_lk_proxy,
    derive_lane_centerline,
)
from tools.epdms.metric_vector import build_partial_metric_vector


def lane_bundle():
    derived = derive_lane_centerline(
        [{"x": 0, "y": 1}, {"x": 10, "y": 1}],
        [{"x": 0, "y": -1}, {"x": 10, "y": -1}],
        sample_spacing_m=1.0,
    )
    center, corridor = derived
    return {"lanes": [LaneCenterline("lane-0", center, corridor, "STRAIGHT", 0)], "intersections": []}


class PartialVectorTests(unittest.TestCase):
    def test_centerline_resampling_is_deterministic(self):
        first = derive_lane_centerline(
            [{"x": 0, "y": 1}, {"x": 10, "y": 1}],
            [{"x": 0, "y": -1}, {"x": 10, "y": -1}],
        )
        second = derive_lane_centerline(
            [{"x": 0, "y": 1}, {"x": 10, "y": 1}],
            [{"x": 10, "y": -1}, {"x": 0, "y": -1}],
        )
        self.assertTrue(np.allclose(first[0], second[0]))

    def test_centered_trajectory_passes_lk(self):
        result = compute_lk_proxy(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), lane_bundle())
        self.assertEqual(result["status"], LK_READY)
        self.assertEqual(result["lk_proxy"], 1.0)

    def test_short_deviation_does_not_fail_continuous_window(self):
        trajectory = np.asarray([[0.0, 0.0], [1.0, 0.8], [2.0, 0.0], [3.0, 0.0]])
        result = compute_lk_proxy(trajectory, lane_bundle())
        self.assertEqual(result["lk_proxy"], 1.0)

    def test_continuous_deviation_fails_lk(self):
        trajectory = np.asarray([[0.0, 0.0]] + [[float(i) * 0.3, 0.8] for i in range(1, 25)])
        result = compute_lk_proxy(trajectory, lane_bundle())
        self.assertEqual(result["lk_proxy"], 0.0)

    def test_intersection_frames_reset_violation(self):
        bundle = lane_bundle()
        bundle["intersections"] = [np.asarray([[1.5, -2], [2.5, -2], [2.5, 2], [1.5, 2]], dtype=float)]
        trajectory = np.asarray([[float(i) * 0.2, 0.8] for i in range(20)])
        result = compute_lk_proxy(trajectory, bundle)
        self.assertEqual(result["intersection_excluded_frames"], 5)
        self.assertEqual(result["lk_proxy"], 1.0)

    def test_missing_or_ambiguous_lane_fails_closed(self):
        missing = compute_lk_proxy(np.asarray([[0.0, 0.0]]), {"lanes": [], "intersections": []})
        self.assertIsNone(missing["lk_proxy"])
        ambiguous_lane = lane_bundle()
        ambiguous_lane["lanes"].append(ambiguous_lane["lanes"][0])
        ambiguous = compute_lk_proxy(np.asarray([[0.0, 0.0], [1.0, 0.0]]), ambiguous_lane)
        self.assertEqual(ambiguous["status"], LK_AMBIGUOUS)
        self.assertIsNone(ambiguous["lk_proxy"])

    def test_direction_output_is_diagnostic_only(self):
        result = compute_lane_direction_alignment_diagnostic(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            {"lane_headings_rad": [0.0, 0.0]},
        )
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY_NO_LEGAL_DIRECTION_PROOF")
        self.assertEqual(result["lane_direction_alignment_fraction"], 1.0)

    def test_official_fields_stay_null_and_old_proxy_is_preserved(self):
        vector = build_partial_metric_vector(
            {"record_key": "clip|mode|0", "clip_id": "clip", "mode": "mode", "alpha": 0.0,
             "collision_free_proxy": 1.0, "dac_proxy": 0.0, "ttc_proxy": 1.0,
             "progress_gt_proxy": 0.9, "future_comfort_proxy": 1.0},
            lk_result={"lk_proxy": 1.0, "status": "READY"},
        )
        for field in ("nc", "dac", "ddc", "tlc", "ttc", "ep", "lk", "hc", "ec"):
            self.assertIsNone(vector[field])
        self.assertEqual(vector["dac_proxy"], 0.0)
        self.assertIsNone(vector["official_epdms_stage1"])
        self.assertIn("lk", vector["missing_components"])


if __name__ == "__main__":
    unittest.main()
