"""Regression tests covering all 18 peer review findings for EPDMS evaluation pipeline."""

import math
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.epdms.coordinates import derive_heading_from_xy
from tools.epdms.geometry_numpy import (
    compute_exact_box_distance,
    get_oriented_box_corners,
    is_point_on_segment,
    point_in_polygon_ray_casting,
    sat_box_intersection,
)
from tools.epdms.kinematics_numpy import ComfortThresholds, KinematicProfile, compute_kinematics
from tools.epdms.proxy_metrics import (
    compute_collision_free_proxy,
    compute_dac_proxy,
    compute_nurec_safety_proxy_v1_composite,
    compute_progress_gt_proxy,
    compute_ttc_proxy,
)
from tools.epdms.schemas import EvaluationScoreRecord, VehicleParameters
from tools.epdms.score_record import evaluate_single_condition
from tools.epdms.io_jsonl import AtomicJsonlWriter, iter_jsonl
from tools.epdms.config import EvaluationConfig
from tools.epdms.aggregate import aggregate_by_group, compute_paired_deltas, compute_paired_summary


class TestPeerReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.vehicle = VehicleParameters()

    # 1. Heading derivation: Rotation invariance on stationary segment
    def test_01_heading_stationary_segment_rotation_invariance(self):
        theta = np.pi / 3.0
        steps = 40
        x = np.zeros(steps)
        y = np.zeros(steps)
        for i in range(20):
            x[i] = i * 0.5 * np.cos(theta)
            y[i] = i * 0.5 * np.sin(theta)
        for i in range(20, 40):
            x[i] = x[19]
            y[i] = y[19]

        headings = derive_heading_from_xy(x, y)
        self.assertEqual(len(headings), steps)
        for h in headings[19:]:
            diff = abs((h - theta + np.pi) % (2 * np.pi) - np.pi)
            self.assertLess(diff, 1e-4)

    # 2. Point-in-polygon boundary consistency
    def test_02_point_in_polygon_boundary_consistency(self):
        poly = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
        on_edges = [
            (2.5, 0.0),
            (5.0, 2.5),
            (2.5, 5.0),
            (0.0, 2.5),
            (0.0, 0.0),
            (5.0, 5.0),
        ]
        for px, py in on_edges:
            self.assertTrue(point_in_polygon_ray_casting(px, py, poly), f"Failed on edge ({px}, {py})")

    # 3. SAT separated box exact distance
    def test_03_sat_separated_exact_euclidean_distance(self):
        box_a = get_oriented_box_corners(0.0, 0.0, 0.0, 4.0, 2.0)
        box_b = get_oriented_box_corners(7.0, 0.0, 0.0, 4.0, 2.0)
        is_col, clearance = sat_box_intersection(box_a, box_b, touch_is_collision=True)
        self.assertFalse(is_col)
        self.assertAlmostEqual(clearance, 3.0, places=4)

    # 4. Kinematics finite check rejects NaN / Inf
    def test_04_kinematics_finite_check(self):
        x = np.linspace(0, 10, 40)
        y = np.linspace(0, 10, 40)
        headings = np.zeros(40)
        x[15] = np.nan
        with self.assertRaises(ValueError):
            compute_kinematics(x, y, headings, dt=0.1)

        x2 = np.linspace(0, 10, 40)
        y2 = np.linspace(0, 10, 40)
        y2[20] = np.inf
        with self.assertRaises(ValueError):
            compute_kinematics(x2, y2, headings, dt=0.1)

    # 5. Comfort metrics NaN handling
    def test_05_comfort_metrics_nan_handling(self):
        x = np.linspace(0, 10, 40)
        y = np.linspace(0, 10, 40)
        headings = np.zeros(40)
        headings[5] = np.nan
        with self.assertRaises(ValueError):
            compute_kinematics(x, y, headings, dt=0.1)

    # 6. Required data contract: Missing map returns None, not 1.0
    def test_06_drivable_area_missing_map_returns_none(self):
        x = np.linspace(0, 10, 40)
        y = np.zeros(40)
        headings = np.zeros(40)
        timestamps = np.arange(40) * 100_000
        dac_score, _, _ = compute_dac_proxy(
            x, y, headings, timestamps, road_polygons=[], vehicle=self.vehicle, t0_us=0, require_drivable_geometry=True
        )
        self.assertIsNone(dac_score)

        dac_score_none, _, _ = compute_dac_proxy(
            x, y, headings, timestamps, road_polygons=None, vehicle=self.vehicle, t0_us=0, require_drivable_geometry=True
        )
        self.assertIsNone(dac_score_none)

    # 7. Required data contract: Missing context obstacles returns None, not 1.0
    def test_07_collision_missing_context_returns_none(self):
        x = np.linspace(0, 10, 40)
        y = np.zeros(40)
        headings = np.zeros(40)
        timestamps = np.arange(40) * 100_000
        speeds = np.ones(40) * 5.0

        cf, _, _, _, _ = compute_collision_free_proxy(
            x, y, headings, timestamps, obstacles=None, vehicle=self.vehicle, t0_us=0, context_present=False
        )
        self.assertIsNone(cf)

        ttc, _, _, _ = compute_ttc_proxy(
            x, y, headings, speeds, timestamps, obstacles=None, vehicle=self.vehicle, t0_us=0, context_present=False
        )
        self.assertIsNone(ttc)

    # 8. Required data contract: Missing GT returns None, not 1.0
    def test_08_progress_missing_gt_returns_none(self):
        x = np.linspace(0, 10, 40)
        y = np.zeros(40)
        prog_none, _, _, _ = compute_progress_gt_proxy(x, y, gt_xyz=None)
        self.assertIsNone(prog_none)

        prog_empty, _, _, _ = compute_progress_gt_proxy(x, y, gt_xyz=np.zeros((0, 3)))
        self.assertIsNone(prog_empty)

    # 9. Progress GT prepends ego origin at t=0
    def test_09_progress_gt_prepends_origin(self):
        x = np.linspace(1, 40, 40)
        y = np.zeros(40)
        # GT starts at (1.0, 0.0) and extends to (40.0, 0.0)
        gt_xyz = np.column_stack([np.linspace(1, 40, 40), np.zeros(40), np.zeros(40)])
        prog, pred_m, gt_m, _ = compute_progress_gt_proxy(x, y, gt_xyz, prepend_t0=True)
        self.assertIsNotNone(prog)
        self.assertAlmostEqual(prog, 1.0, places=3)
        self.assertAlmostEqual(gt_m, 40.0, places=3)

    # 10. Event timestamps use real dt / microsecond offsets
    def test_10_event_timestamps_use_real_delta_t(self):
        x = np.zeros(40)
        y = np.zeros(40)
        headings = np.zeros(40)
        t0 = 1_000_000_000
        timestamps = np.array([t0 + i * 100_000 for i in range(40)])
        obs = [{
            "timestamp_micros": t0 + 500_000,
            "obstacle": {
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }]
        cf, event_time, _, _, _ = compute_collision_free_proxy(
            x, y, headings, timestamps, obstacles=obs, vehicle=self.vehicle, t0_us=t0, context_present=True
        )
        self.assertEqual(cf, 0.0)
        self.assertAlmostEqual(event_time, 0.5, places=4)

    # 11. Horizon contract: Trajectory < 40 waypoints rejected as invalid
    def test_11_insufficient_waypoints_rejected(self):
        short_traj = [{"x_m": i * 0.5, "y_m": 0.0} for i in range(35)]
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "clean_waypoints": short_traj
        }
        rec = evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertIn("InsufficientWaypointsError", rec.failure_reason)

    # 12. Guidance fallback rejection: missing guided trajectory when alpha > 0
    def test_12_guidance_fallback_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.5,
            "clean_waypoints": [{"x_m": 0.0, "y_m": 0.0} for _ in range(40)]
        }
        rec = evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertIn("missing_guided_trajectory", rec.failure_reason)

    # 13. Input coordinate NaN/Inf rejected
    def test_13_input_nan_coordinates_rejected(self):
        traj = [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        traj[10]["x_m"] = np.nan
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "clean_waypoints": traj
        }
        rec = evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertIn("Non-finite coordinate", rec.failure_reason)

    # 14. Error boundary for invalid / non-numeric alpha
    def test_14_error_boundary_catches_invalid_alpha(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": "not_a_number",
            "clean_waypoints": [{"x_m": 0.0, "y_m": 0.0} for _ in range(40)]
        }
        rec = evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "ValueError")

    # 15. Official profiles raise NotImplementedError
    def test_15_official_profiles_raise_not_implemented(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "clean_waypoints": [{"x_m": 0.0, "y_m": 0.0} for _ in range(40)]
        }
        with self.assertRaises(NotImplementedError):
            evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle, metric_profile="navsim_v2_full")

        with self.assertRaises(NotImplementedError):
            evaluate_single_condition(pred_row, context_row=None, gt_row=None, vehicle=self.vehicle, metric_profile="navsim_v2_stage1")

    # 16. Resume fingerprint mismatch rejection
    def test_16_resume_fingerprint_mismatch_rejection(self):
        cfg_dict = {
            "metric_profile": "nurec_safety_proxy_v1",
            "horizon_s": 4.0,
            "frequency_hz": 10.0,
            "strict_mode": True,
            "paths": {},
            "vehicle": {},
            "proxy": {},
        }
        cfg1 = EvaluationConfig(cfg_dict)
        fp1 = cfg1.compute_effective_fingerprint({"source": "hash_a"})
        fp2 = cfg1.compute_effective_fingerprint({"source": "hash_b"})
        self.assertNotEqual(fp1, fp2)

    # 17. Atomic Jsonl writer rejects NaN
    def test_17_atomic_writer_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "scores.jsonl"
            writer = AtomicJsonlWriter(out_file)
            bad_record = {"clip_id": "c1", "score": float("nan")}
            with self.assertRaises(ValueError):
                writer.write(bad_record)
            writer.close()

    # 18. Aggregator numeric stability with None and Bootstrap CI
    def test_18_aggregator_safe_sort_with_none_and_bootstrap_ci(self):
        records = [
            {"mode": "cross_scene", "alpha": 0.0, "clip_id": "c1", "rule_group": None, "nurec_safety_proxy_v1": 0.95, "valid": True},
            {"mode": "cross_scene", "alpha": 0.5, "clip_id": "c1", "rule_group": "group_a", "nurec_safety_proxy_v1": 0.90, "valid": True},
            {"mode": "cross_scene", "alpha": 0.0, "clip_id": "c2", "rule_group": "group_b", "nurec_safety_proxy_v1": 0.85, "valid": True},
            {"mode": "cross_scene", "alpha": 0.5, "clip_id": "c2", "rule_group": None, "nurec_safety_proxy_v1": 0.88, "valid": True},
        ]
        agg = aggregate_by_group(records, group_keys=["rule_group", "alpha"])
        self.assertTrue(len(agg) > 0)

        paired = compute_paired_deltas(records)
        summary = compute_paired_summary(paired, bootstrap_iterations=100)
        self.assertTrue(len(summary) > 0)
        self.assertIn("ci95_lower", summary[0])
        self.assertIn("ci95_upper", summary[0])


if __name__ == "__main__":
    unittest.main()
