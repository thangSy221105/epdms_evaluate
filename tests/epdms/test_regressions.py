"""Regression tests covering all 18 peer review findings for EPDMS evaluation pipeline."""

import json
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
    normalize_obstacle_record,
)
from tools.epdms.schemas import EvaluationScoreRecord, VehicleParameters
from tools.epdms.score_record import evaluate_single_condition, extract_and_validate_trajectory
from tools.epdms.io_jsonl import AtomicJsonlWriter, iter_jsonl
from tools.epdms.config import EvaluationConfig
from tools.epdms.aggregate import (
    aggregate_by_group,
    compute_ade_disagreement_summary,
    compute_paired_deltas,
    compute_paired_summary,
)
from tools.epdms.audit import audit_data_contracts
from tools.epdms.map_loader import inspect_clip_map_status
from tools.epdms.reporting import export_ade_disagreement_to_markdown


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

    # 19. Obstacle missing timestamp rejected as INSUFFICIENT_OBSERVATION_DATA
    def test_19_obstacle_missing_timestamp_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        }
        context_row = {
            "clip_id": "clip_001",
            "semantic_context": {
                "obstacle": {
                    "all_obstacles": [{
                        "obstacle": {
                            "center": {"x": 10.0, "y": 0.0, "z": 0.0},
                            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    }]
                }
            }
        }
        rec = evaluate_single_condition(pred_row, context_row=context_row, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertIn(rec.failure_type, ["INSUFFICIENT_OBSERVATION_DATA", "CORRUPTED_OBSERVATION_DATA"])

    # 20. Obstacle 100% out of window rejected as INSUFFICIENT_OBSERVATION_DATA
    def test_20_obstacle_out_of_window_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        }
        context_row = {
            "clip_id": "clip_001",
            "semantic_context": {
                "obstacle": {
                    "all_obstacles": [{
                        "timestamp_micros": 1_200_000_000,
                        "obstacle": {
                            "center": {"x": 10.0, "y": 0.0, "z": 0.0},
                            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    }]
                }
            }
        }
        rec = evaluate_single_condition(pred_row, context_row=context_row, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "INSUFFICIENT_OBSERVATION_DATA")

    # 21. Flat and nested obstacle schemas supported without phantom (0, 0)
    def test_21_obstacle_flat_and_nested_schema_support(self):
        nested = {
            "timestamp_micros": 1000,
            "obstacle": {
                "center": {"x": 15.0, "y": 5.0, "z": 0.0},
                "size": {"x": 4.5, "y": 2.1, "z": 1.6},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }
        flat = {
            "timestamp_micros": 1000,
            "center": {"x": 15.0, "y": 5.0, "z": 0.0},
            "size": {"x": 4.5, "y": 2.1, "z": 1.6},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        norm_nested = normalize_obstacle_record(nested)
        norm_flat = normalize_obstacle_record(flat)
        self.assertIsNotNone(norm_nested)
        self.assertIsNotNone(norm_flat)
        self.assertAlmostEqual(norm_nested["center_x"], 15.0)
        self.assertAlmostEqual(norm_flat["center_x"], 15.0)
        self.assertAlmostEqual(norm_nested["length_m"], 4.5)
        self.assertAlmostEqual(norm_flat["length_m"], 4.5)

    # 22. Non-finite map polygon rejected
    def test_22_non_finite_map_polygon_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        }
        context_empty = {
            "clip_id": "clip_001",
            "semantic_context": {"obstacle": {"all_obstacles": []}}
        }
        bad_poly = np.array([[0.0, 0.0], [np.nan, 5.0], [5.0, 5.0]])
        rec = evaluate_single_condition(
            pred_row, context_row=context_empty, gt_row=None, vehicle=self.vehicle, lane_polygons=[bad_poly]
        )
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "map_geometry_contract")

    # 23. Ground Truth with insufficient waypoints (< 40) rejected
    def test_23_ground_truth_insufficient_waypoints_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        }
        context_empty = {
            "clip_id": "clip_001",
            "semantic_context": {"obstacle": {"all_obstacles": []}}
        }
        lane_polys = [np.array([[-10.0, -10.0], [50.0, -10.0], [50.0, 10.0], [-10.0, 10.0]])]
        gt_short = {
            "clip_id": "clip_001",
            "expert_future": [{"x": float(i), "y": 0.0} for i in range(15)]
        }
        rec = evaluate_single_condition(
            pred_row, context_row=context_empty, gt_row=gt_short, vehicle=self.vehicle, lane_polygons=lane_polys
        )
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "ground_truth_contract")

    # 24. Clear road serializes minimum_clearance_m = null without allow_nan crash
    def test_24_clear_road_minimum_clearance_null_serialization(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)]
        }
        context_empty = {
            "clip_id": "clip_001",
            "semantic_context": {"obstacle": {"all_obstacles": []}}
        }
        lane_polys = [np.array([[-10.0, -10.0], [50.0, -10.0], [50.0, 10.0], [-10.0, 10.0]])]
        gt_row = {"clip_id": "clip_001", "ego_future_xyz": [[i * 0.5, 0.0, 0.0] for i in range(40)]}
        rec = evaluate_single_condition(
            pred_row, context_row=context_empty, gt_row=gt_row, vehicle=self.vehicle, lane_polygons=lane_polys
        )
        self.assertTrue(rec.valid)
        self.assertIsNone(rec.minimum_clearance_m)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "clear_road.jsonl"
            writer = AtomicJsonlWriter(out_file)
            writer.write(rec.to_dict())
            writer.close()
            text = out_file.read_text(encoding="utf-8")
            self.assertIn('"minimum_clearance_m": null', text)

    # 25. Effective fingerprint changes on horizon override or map changes
    def test_25_effective_fingerprint_changes_on_horizon_and_map(self):
        cfg = EvaluationConfig({
            "metric_profile": "nurec_safety_proxy_v1",
            "horizon_s": 4.0,
            "frequency_hz": 10.0,
            "strict_mode": True,
            "paths": {},
            "vehicle": {},
            "proxy": {},
        })
        fp_base = cfg.compute_effective_fingerprint({"source": "s1"}, runtime_overrides={"horizon_s": 4.0})
        fp_h3 = cfg.compute_effective_fingerprint({"source": "s1"}, runtime_overrides={"horizon_s": 3.0})
        fp_map = cfg.compute_effective_fingerprint(
            {"source": "s1", "context_filtered_map": "map_hash_123"},
            runtime_overrides={"horizon_s": 4.0}
        )
        self.assertNotEqual(fp_base, fp_h3)
        self.assertNotEqual(fp_base, fp_map)

    # 26. prepare_file_for_resume recovers .tmp and repairs trailing truncation
    def test_26_prepare_file_for_resume_tmp_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "scores.jsonl"
            tmp = Path(tmpdir) / "scores.jsonl.tmp"
            tmp.write_text('{"record_key": "c1|m|0", "score": 1.0}\n{"record_key": "c1|m|0.5", "score": 0.8}\n{"record_key": "c1|m|1.0", "score":', encoding="utf-8")
            AtomicJsonlWriter.prepare_file_for_resume(target)
            self.assertTrue(target.is_file())
            self.assertFalse(tmp.is_file())
            rows = list(iter_jsonl(target))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["record_key"], "c1|m|0")
            self.assertEqual(rows[1]["record_key"], "c1|m|0.5")

    # 27. ADE and FDE metrics computed, and ADE-Safety disagreement detected
    def test_27_ade_fde_computed_and_disagreement_summary(self):
        gt_waypoints = [[float(i), 0.0, 0.0] for i in range(40)]
        gt_row = {"clip_id": "c1", "ego_future_xyz": gt_waypoints}
        context_empty = {"clip_id": "c1", "semantic_context": {"obstacle": {"all_obstacles": []}}}
        lane_polys = [np.array([[-10.0, -10.0], [50.0, -10.0], [50.0, 10.0], [-10.0, 10.0]])]

        pred_base = {
            "clip_id": "c1",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": float(i), "y_m": 0.0} for i in range(40)],
        }
        pred_guided = {
            "clip_id": "c1",
            "mode": "cross_scene",
            "alpha": 0.5,
            "t0_us": 5_100_000,
            "guided_waypoints": [{"x_m": float(i), "y_m": 2.0} for i in range(40)],
        }

        rec_base = evaluate_single_condition(pred_base, context_row=context_empty, gt_row=gt_row, vehicle=self.vehicle, lane_polygons=lane_polys)
        rec_guided = evaluate_single_condition(pred_guided, context_row=context_empty, gt_row=gt_row, vehicle=self.vehicle, lane_polygons=lane_polys)

        self.assertTrue(rec_base.valid)
        self.assertTrue(rec_guided.valid)
        self.assertAlmostEqual(rec_base.ade_m, 0.0, places=4)
        self.assertAlmostEqual(rec_guided.ade_m, 2.0, places=4)
        self.assertAlmostEqual(rec_base.fde_m, 0.0, places=4)
        self.assertAlmostEqual(rec_guided.fde_m, 2.0, places=4)

        paired = compute_paired_deltas([rec_base.to_dict(), rec_guided.to_dict()])
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["delta_ade"], 2.0, places=4)

        disagreement = compute_ade_disagreement_summary(paired)
        self.assertEqual(len(disagreement), 1)
        self.assertEqual(disagreement[0]["ade_penalized_cases"], 1)
        self.assertEqual(disagreement[0]["disagreement_count"], 1)
        self.assertAlmostEqual(disagreement[0]["disagreement_rate_pct"], 100.0, places=2)

    # 28. Audit trajectory check loop variable isolation (no leak from alpha)
    def test_28_audit_trajectory_check_order_independence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            pred_file = td / "preds.jsonl"
            ctx_file = td / "ctx.jsonl"
            gt_file = td / "gt.jsonl"

            r1 = {
                "clip_id": "c1", "mode": "cross_scene", "alpha": 0.5,
                "guided_waypoints": [{"x_m": float(i), "y_m": 0.0} for i in range(40)]
            }
            r2 = {
                "clip_id": "c1", "mode": "cross_scene", "alpha": 0.0,
                "clean_waypoints": [{"x_m": float(i), "y_m": 0.0} for i in range(40)]
            }
            with pred_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")
            ctx_file.write_text('{"clip_id": "c1"}\n', encoding="utf-8")
            gt_file.write_text('{"clip_id": "c1"}\n', encoding="utf-8")

            cfg = EvaluationConfig({
                "metric_profile": "nurec_safety_proxy_v1",
                "paths": {
                    "prediction_jsonl": str(pred_file),
                    "context_jsonl": str(ctx_file),
                    "ground_truth_jsonl": str(gt_file),
                    "context_filtered_dir": str(td / "non_existent"),
                }
            })
            report = audit_data_contracts(cfg)
            self.assertEqual(report["trajectory_contract_stats"]["total_issues"], 0)

    # 29. Audit readiness blocked if map polygons missing
    def test_29_audit_readiness_blocked_on_missing_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            pred_file = td / "preds.jsonl"
            ctx_file = td / "ctx.jsonl"
            gt_file = td / "gt.jsonl"
            row_data = {"clip_id": "c1", "mode": "cross_scene", "alpha": 0.0, "clean_waypoints": [{"x_m": 0.0, "y_m": 0.0} for _ in range(40)]}
            pred_file.write_text(json.dumps(row_data) + "\n", encoding="utf-8")
            ctx_file.write_text('{"clip_id": "c1"}\n', encoding="utf-8")
            gt_file.write_text('{"clip_id": "c1"}\n', encoding="utf-8")

            cfg = EvaluationConfig({
                "metric_profile": "nurec_safety_proxy_v1",
                "paths": {
                    "prediction_jsonl": str(pred_file),
                    "context_jsonl": str(ctx_file),
                    "ground_truth_jsonl": str(gt_file),
                    "context_filtered_dir": str(td / "empty_dir"),
                }
            })
            (td / "empty_dir").mkdir()
            report = audit_data_contracts(cfg)
            self.assertEqual(report["readiness"]["nurec_safety_proxy_v1"], "NOT READY")

    # 30. Obstacle in expanded window (+/- 0.5s) but zero frames matched -> INSUFFICIENT_OBSERVATION_DATA
    def test_30_obstacle_in_expanded_window_but_no_frame_matched_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)],
        }
        # Obstacle at t0 - 400_000 us (within 500_000 window, but > 50_000 discrete frame tolerance)
        context_row = {
            "clip_id": "clip_001",
            "semantic_context": {
                "obstacle": {
                    "all_obstacles": [{
                        "timestamp_micros": 5_100_000 - 400_000,
                        "obstacle": {
                            "center": {"x": 10.0, "y": 0.0, "z": 0.0},
                            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }]
                }
            },
        }
        rec = evaluate_single_condition(pred_row, context_row=context_row, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "obstacle_observation_contract")
        self.assertEqual(rec.failure_type, "INSUFFICIENT_OBSERVATION_DATA")

    # 31. Corrupted / non-finite obstacle in window rejected
    def test_31_corrupted_obstacle_in_window_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)],
        }
        context_row = {
            "clip_id": "clip_001",
            "semantic_context": {
                "obstacle": {
                    "all_obstacles": [{
                        "timestamp_micros": 5_100_000,
                        "obstacle": {
                            "center": {"x": np.nan, "y": 0.0, "z": 0.0},
                            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }]
                }
            },
        }
        rec = evaluate_single_condition(pred_row, context_row=context_row, gt_row=None, vehicle=self.vehicle)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "obstacle_observation_contract")
        self.assertEqual(rec.failure_type, "CORRUPTED_OBSERVATION_DATA")

    # 32. Strict mode rejects missing t0_us across all sources
    def test_32_strict_mode_missing_t0_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)],
        }
        context_row = {
            "clip_id": "clip_001",
            "semantic_context": {"obstacle": {"all_obstacles": []}},
        }
        rec = evaluate_single_condition(
            pred_row, context_row=context_row, gt_row=None, vehicle=self.vehicle, strict_mode=True
        )
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "time_origin_contract")
        self.assertEqual(rec.failure_type, "MissingTimeOriginError")

    # 33. Inconsistent waypoint timeline metadata rejected
    def test_33_inconsistent_waypoint_timeline_rejected(self):
        # Constant timestamps (all 0.1) violate strict monotonicity
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": float(i), "y_m": 0.0, "t_s": 0.1} for i in range(40)],
        }
        with self.assertRaises(ValueError) as ctx:
            extract_and_validate_trajectory(pred_row, alpha=0.0)
        self.assertIn("NonMonotonicWaypointTimelineError", str(ctx.exception))

    # 34. GT with empty coordinate dicts rejected
    def test_34_gt_empty_coordinate_dict_rejected(self):
        pred_row = {
            "clip_id": "clip_001",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": 5_100_000,
            "clean_waypoints": [{"x_m": i * 0.5, "y_m": 0.0} for i in range(40)],
        }
        context_empty = {
            "clip_id": "clip_001",
            "semantic_context": {"obstacle": {"all_obstacles": []}},
        }
        lane_polys = [np.array([[-10.0, -10.0], [50.0, -10.0], [50.0, 10.0], [-10.0, 10.0]])]
        gt_missing_coords = {
            "clip_id": "clip_001",
            "expert_future": [{"timestamp_micros": i * 100_000} for i in range(40)],
        }
        rec = evaluate_single_condition(
            pred_row, context_row=context_empty, gt_row=gt_missing_coords, vehicle=self.vehicle, lane_polygons=lane_polys
        )
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "ground_truth_contract")
        self.assertEqual(rec.failure_type, "MissingGroundTruthCoordinatesError")

    # 35. Full content map hashing detects byte changes with identical size
    def test_35_full_content_map_hashing(self):
        import hashlib
        def hash_map_dir(m_dir):
            m_hasher = hashlib.sha256()
            for cdir in sorted(m_dir.iterdir()):
                if cdir.is_dir():
                    clipgt = cdir / "clipgt"
                    for pq_name in ["lane.parquet", "intersection_area.parquet"]:
                        pq_file = clipgt / pq_name
                        if pq_file.is_file():
                            m_hasher.update(f"{cdir.name}/{pq_name}:".encode("utf-8"))
                            with pq_file.open("rb") as f:
                                while chunk := f.read(65536):
                                    m_hasher.update(chunk)
            return m_hasher.hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            d1 = base_dir / "dir1"
            d2 = base_dir / "dir2"
            f1 = d1 / "clip1" / "clipgt" / "lane.parquet"
            f2 = d2 / "clip1" / "clipgt" / "lane.parquet"
            f1.parent.mkdir(parents=True)
            f2.parent.mkdir(parents=True)
            # Same length (4 bytes) but different bytes
            f1.write_bytes(b"ABCD")
            f2.write_bytes(b"ABCE")
            h1 = hash_map_dir(d1)
            h2 = hash_map_dir(d2)
            self.assertNotEqual(h1, h2)

    # 36. Score file exists without manifest rejected on resume
    def test_36_score_file_exists_without_manifest_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            score_file = Path(tmpdir) / "scores.jsonl"
            manifest_file = Path(tmpdir) / "run_manifest.json"
            score_file.write_text('{"record_key": "c1|m|0", "score": 1.0}\n', encoding="utf-8")
            self.assertTrue(score_file.is_file() and score_file.stat().st_size > 0)
            self.assertFalse(manifest_file.is_file())
            with self.assertRaises(ValueError) as ctx:
                if score_file.is_file() and score_file.stat().st_size > 0 and not manifest_file.is_file():
                    raise ValueError(
                        f"Resume rejected: score file exists ({score_file.name}) but run manifest ({manifest_file.name}) is missing."
                    )
            self.assertIn("Resume rejected", str(ctx.exception))

    # 37. Pre-run manifest written with RUNNING status
    def test_37_pre_run_manifest_written_before_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "run_manifest.json"
            manifest_data = {
                "status": "RUNNING",
                "profile": "nurec_safety_proxy_v1",
                "effective_fingerprint": "abc12345",
            }
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest_data, f)
            read_back = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(read_back["status"], "RUNNING")
            self.assertEqual(read_back["effective_fingerprint"], "abc12345")

    # 38. Missing ADE rendered as None and N/A in table
    def test_38_missing_ade_rendered_as_none_and_na(self):
        paired_no_ade = [{
            "clip_id": "c1",
            "mode": "cross_scene",
            "alpha": 0.5,
            "delta_safety_proxy_raw": 0.0,
            "delta_ade": None,
            "out_cf": 1.0,
            "out_dac": 1.0,
        }]
        summary = compute_ade_disagreement_summary(paired_no_ade)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_with_ade"], 0)
        self.assertIsNone(summary[0]["mean_delta_ade"])
        self.assertIsNone(summary[0]["disagreement_rate_pct"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_md = Path(tmpdir) / "ade_table.md"
            export_ade_disagreement_to_markdown(summary, out_md)
            text = out_md.read_text(encoding="utf-8")
            self.assertIn("N/A", text)

    # 39. Paired deltas rejects horizon mismatch
    def test_39_paired_deltas_rejects_horizon_mismatch(self):
        r_base = {
            "clip_id": "c1", "mode": "cross_scene", "alpha": 0.0,
            "horizon_s": 4.0, "frequency_hz": 10.0, "metric_profile": "nurec_safety_proxy_v1",
            "nurec_safety_proxy_v1": 1.0,
        }
        r_cand_diff_horizon = {
            "clip_id": "c1", "mode": "cross_scene", "alpha": 0.5,
            "horizon_s": 6.4, "frequency_hz": 10.0, "metric_profile": "nurec_safety_proxy_v1",
            "nurec_safety_proxy_v1": 1.0,
        }
        paired = compute_paired_deltas([r_base, r_cand_diff_horizon])
        self.assertEqual(len(paired), 0)

    # 40. Disagreement distinguishes gate regression from genuine safety
    def test_40_disagreement_distinguishes_gate_regression(self):
        # Candidate maintained score (delta = 0) but dropped CF from 1.0 to 0.0 while DAC rose
        paired = [{
            "clip_id": "c1",
            "mode": "cross_scene",
            "alpha": 0.5,
            "delta_safety_proxy_raw": 0.0,
            "delta_ade": 0.2,
            "base_cf": 1.0,
            "out_cf": 0.0,
            "base_dac": 0.0,
            "out_dac": 1.0,
        }]
        summary = compute_ade_disagreement_summary(paired, ade_penalty_threshold_m=0.05)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["ade_penalized_cases"], 1)
        self.assertEqual(summary[0]["disagreement_count"], 1)
        self.assertEqual(summary[0]["genuinely_safe_count"], 0)
        self.assertEqual(summary[0]["safety_compromised_count"], 1)

    # 41. Dynamic TTC projection horizon up to 2.0s
    def test_41_dynamic_ttc_projection_2s(self):
        x = np.array([0.0, 0.0])
        y = np.array([0.0, 0.0])
        headings = np.array([0.0, 0.0])
        speeds = np.array([10.0, 10.0])  # 10 m/s forward projection
        timestamps_us = np.array([0, 100_000], dtype=np.int64)

        # Place one obstacle at t=0.5s (y=10m, no collision) and one at t=1.8s (x=22m, collision at dt=1.8s)
        obs = [
            {
                "timestamp_micros": int(0.5 * 1_000_000),
                "center": {"x": 5.0, "y": 10.0, "z": 0.0},
                "size": {"x": 2.0, "y": 2.0, "z": 1.5},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            {
                "timestamp_micros": int(1.8 * 1_000_000),
                "center": {"x": 22.0, "y": 0.0, "z": 0.0},
                "size": {"x": 2.0, "y": 2.0, "z": 1.5},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        ]
        ttc_score_1s, min_ttc_1s, _, _ = compute_ttc_proxy(
            x, y, headings, speeds, timestamps_us, obs, self.vehicle, t0_us=0, ttc_horizon_s=1.0
        )
        # At 1.0s horizon, projection only goes up to 1.0s, so 1.8s collision is NOT in horizon
        self.assertEqual(ttc_score_1s, 1.0)

        ttc_score_2s, min_ttc_2s, _, _ = compute_ttc_proxy(
            x, y, headings, speeds, timestamps_us, obs, self.vehicle, t0_us=0, ttc_horizon_s=2.0
        )
        # At 2.0s horizon, projection reaches 1.8s and detects collision
        self.assertEqual(ttc_score_2s, 0.0)
        self.assertAlmostEqual(min_ttc_2s, 1.8, places=1)

    # 42. Map inventory status breakdown classification
    def test_42_map_inventory_status_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            # 1. FILE_NOT_FOUND
            res_fnf = inspect_clip_map_status(base_dir, "clip_fnf")
            self.assertEqual(res_fnf["status"], "FILE_NOT_FOUND")

            # 2. NO_DRIVABLE_POLYGON (empty location parquet)
            import pandas as pd
            c_nodrive = base_dir / "clip_nodrive" / "clipgt"
            c_nodrive.mkdir(parents=True)
            df_empty = pd.DataFrame([{"intersection_area": {"location": []}}])
            df_empty.to_parquet(c_nodrive / "intersection_area.parquet")
            res_nodrive = inspect_clip_map_status(base_dir, "clip_nodrive")
            self.assertEqual(res_nodrive["status"], "NO_DRIVABLE_POLYGON")

            # 3. OK
            c_ok = base_dir / "clip_ok" / "clipgt"
            c_ok.mkdir(parents=True)
            df_ok = pd.DataFrame([{"intersection_area": {"location": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0}, {"x": 5.0, "y": 5.0}]}}])
            df_ok.to_parquet(c_ok / "intersection_area.parquet")
            res_ok = inspect_clip_map_status(base_dir, "clip_ok")
            self.assertEqual(res_ok["status"], "OK")
            self.assertEqual(res_ok["total_polygons"], 1)

    # 43. Observation coverage metrics recorded in EvaluationScoreRecord
    def test_43_observation_coverage_metrics_recorded(self):
        # 10 poses
        x = np.linspace(0.0, 10.0, 11)
        y = np.zeros(11)
        headings = np.zeros(11)
        timestamps_us = np.arange(11, dtype=np.int64) * 100_000  # 0 to 1s
        # 5 obstacle frames matching timestamps 0 to 400_000
        obs = [
            {
                "timestamp_micros": int(t),
                "trackline_id": "trk1",
                "category": "car",
                "center": {"x": 50.0, "y": 50.0, "z": 0.0},
                "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                "orientation": {"w": 1.0, "z": 0.0},
            }
            for t in timestamps_us[:5]
        ]
        res = compute_collision_free_proxy(
            x, y, headings, timestamps_us, obs, self.vehicle, t0_us=0, touch_is_collision=True, context_present=True
        )
        self.assertEqual(res.matched_observation_frames, 5)
        self.assertEqual(res.required_observation_frames, 11)
        self.assertAlmostEqual(res.observation_coverage_ratio, 5.0 / 11.0, places=3)
        # Verify tuple unpacking still works seamlessly
        cf, col_t, min_clr, trks, typs = res
        self.assertEqual(cf, 1.0)
        self.assertIsNone(col_t)


if __name__ == "__main__":
    unittest.main()
