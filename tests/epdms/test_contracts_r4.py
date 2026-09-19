"""Dedicated regression test suite for Round 4 Contracts & Coverage fixes (Groups A, B, C, D)."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.epdms.schemas import VehicleParameters
from tools.epdms.observation_contract import (
    CorruptedObservationDataError,
    ObservationState,
    evaluate_query_coverage,
    index_and_filter_obstacles,
    normalize_and_validate_obstacle,
)
from tools.epdms.time_contract import (
    ConflictingTimeOriginError,
    MissingGroundTruthCoordinatesError,
    MissingTimeOriginError,
    NonMonotonicWaypointTimelineError,
    TimeContractError,
    TimelineHorizonMismatchError,
    resolve_time_origin,
    validate_and_normalize_timeline,
)
from tools.epdms.run_identity import (
    AmbiguousCheckpointError,
    RunIdentityError,
    compute_file_content_sha256,
    compute_map_directory_content_sha256,
    compute_run_effective_fingerprint,
    verify_resume_safety_before_recovery,
    write_manifest_atomic,
)
from tools.epdms.map_loader import inspect_clip_map_status, load_lane_polygons_for_clip
from tools.epdms.proxy_metrics import compute_collision_free_proxy, compute_ttc_proxy
from tools.epdms.score_record import evaluate_single_condition
from tools.epdms.io_jsonl import AtomicJsonlWriter, iter_jsonl


class TestContractsRound4(unittest.TestCase):
    def setUp(self):
        self.vehicle = VehicleParameters()
        self.t0 = 5_100_000
        self.timestamps_40 = np.array([self.t0 + (i + 1) * 100_000 for i in range(40)], dtype=np.int64)
        self.timestamps_41 = np.array([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)

    # =========================================================================
    # Group A: Observation Coverage and Obstacle Integrity
    # =========================================================================

    def test_A1_reject_1_of_41_frame_coverage_false_safe(self):
        """When obstacles only exist at 1 frame (e.g. frame 0), 40 frames are unobserved.
        In strict mode, CF must NOT return safe (cf=1.0); it must return None and be rejected as INSUFFICIENT_OBSERVATION_DATA.
        """
        x = np.linspace(0.0, 20.0, 41)
        y = np.zeros(41)
        headings = np.zeros(41)
        # Obstacle far away at frame 0 (no collision at t0)
        obs = [{
            "timestamp_micros": int(self.timestamps_41[0]),
            "trackline_id": "trk_001",
            "center": {"x": 100.0, "y": 100.0, "z": 0.0},
            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }]

        # Direct CF evaluation in strict mode
        cf_res = compute_collision_free_proxy(
            x, y, headings, self.timestamps_41, obs, self.vehicle,
            t0_us=self.t0, strict_mode=True
        )
        self.assertIsNone(cf_res[0], "CF score must be None due to incomplete observation coverage (1/41)")
        self.assertEqual(cf_res.cf_observed_frames, 1)
        self.assertEqual(cf_res.cf_missing_frames, 40)
        self.assertAlmostEqual(cf_res.cf_coverage_ratio, 1.0 / 41.0, places=4)

        # In pipeline condition evaluation
        pred_row = {
            "clip_id": "clip_test_a1",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": self.t0,
            "clean_waypoints": [{"x_m": float(x[i]), "y_m": float(y[i]), "timestamp_micros": int(self.timestamps_41[i])} for i in range(41)],
        }
        ctx_row = {
            "clip_id": "clip_test_a1",
            "semantic_context": {"obstacle": {"all_obstacles": obs}},
        }
        rec = evaluate_single_condition(pred_row, context_row=ctx_row, gt_row=None, vehicle=self.vehicle, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "coordinate_contract")
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_A2_confirmed_empty_all_frames_passes(self):
        """Confirmed empty timestamps allow achieving 100% coverage even with 0 obstacle boxes."""
        x = np.linspace(0.0, 20.0, 41)
        y = np.zeros(41)
        headings = np.zeros(41)
        # Empty obstacles list but confirmed empty scene
        cf_res = compute_collision_free_proxy(
            x, y, headings, self.timestamps_41, [], self.vehicle,
            t0_us=self.t0, strict_mode=True, confirmed_empty_scene=True
        )
        self.assertEqual(cf_res[0], 1.0)
        self.assertEqual(cf_res.cf_coverage_ratio, 1.0)

    def test_A3_missing_timestamp_obstacle_raises_corrupted_error(self):
        """Obstacle missing timestamp must raise CorruptedObservationDataError in strict mode."""
        corrupted_obs = [{
            "trackline_id": "trk_bad",
            "center": {"x": 10.0, "y": 0.0, "z": 0.0},
            "size": {"x": 4.0, "y": 2.0, "z": 1.5},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }]
        with self.assertRaises(CorruptedObservationDataError) as ctx:
            index_and_filter_obstacles(corrupted_obs, raise_on_corrupt=True)
        self.assertIn("missing required timestamp_micros", str(ctx.exception))

        pred_row = {
            "clip_id": "clip_test_a3",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": self.t0,
            "clean_waypoints": [{"x_m": float(i), "y_m": 0.0, "timestamp_micros": int(self.t0 + (i + 1) * 100_000)} for i in range(40)],
        }
        ctx_row = {
            "clip_id": "clip_test_a3",
            "semantic_context": {"obstacle": {"all_obstacles": corrupted_obs}},
        }
        rec = evaluate_single_condition(pred_row, context_row=ctx_row, gt_row=None, vehicle=self.vehicle, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_A4_separate_cf_and_ttc_coverage_evaluation(self):
        """TTC requires observation queries beyond CF horizon up to t0 + horizon + ttc_horizon."""
        x = np.linspace(0.0, 10.0, 5)
        y = np.zeros(5)
        headings = np.zeros(5)
        speeds = np.zeros(5)
        ts = np.array([self.t0 + i * 100_000 for i in range(5)], dtype=np.int64)

        # Observations present only during the 5 CF poses, none during future TTC projection
        obs = [{
            "timestamp_micros": int(t),
            "trackline_id": "trk_1",
            "center": {"x": 50.0, "y": 50.0, "z": 0.0},
            "size": {"x": 2.0, "y": 2.0, "z": 1.5},
            "orientation": {"w": 1.0, "z": 0.0},
        } for t in ts]

        cf_res = compute_collision_free_proxy(x, y, headings, ts, obs, self.vehicle, t0_us=self.t0, strict_mode=True)
        self.assertEqual(cf_res[0], 1.0)
        self.assertEqual(cf_res.cf_coverage_ratio, 1.0)

        # TTC projection with 1.0s horizon needs queries up to ts[-1] + 1.0s, which are missing
        ttc_res = compute_ttc_proxy(x, y, headings, speeds, ts, obs, self.vehicle, t0_us=self.t0, ttc_horizon_s=1.0, strict_mode=True)
        self.assertIsNone(ttc_res[0], "TTC must fail strict coverage because projections extend past observed window")
        self.assertGreater(ttc_res.ttc_missing_observations, 0)

    # =========================================================================
    # Group B: Timeline Normalization and Grid Standardization
    # =========================================================================

    def test_B1_single_time_contract_validates_monotonicity(self):
        """Non-monotonic timestamps in trajectory waypoints must be rejected."""
        non_monotonic_wps = [
            {"x_m": float(i), "y_m": 0.0, "timestamp_micros": self.t0 + (i if i != 5 else 4) * 100_000}
            for i in range(40)
        ]
        with self.assertRaises(NonMonotonicWaypointTimelineError):
            validate_and_normalize_timeline(
                non_monotonic_wps, target_poses=40, expected_frequency_hz=10.0,
                expected_horizon_s=4.0, strict_grid=True
            )

    def test_B2_timeline_horizon_mismatch_rejected(self):
        """Trajectory spanning 8s @ 5Hz (dt=0.2s) when expecting 4s @ 10Hz (dt=0.1s) must be rejected."""
        mismatched_dt_wps = [
            {"x_m": float(i), "y_m": 0.0, "timestamp_micros": self.t0 + (i + 1) * 200_000}  # 5Hz grid
            for i in range(40)
        ]
        with self.assertRaises(TimelineHorizonMismatchError):
            validate_and_normalize_timeline(
                mismatched_dt_wps, target_poses=40, expected_frequency_hz=10.0,
                expected_horizon_s=4.0, strict_grid=True
            )

    def test_B3_multi_source_t0_resolution(self):
        """resolve_time_origin resolves t0 and detects conflicts between prediction and context."""
        t0_pred = 5_100_000
        t0_ctx = 5_100_000
        resolved, src = resolve_time_origin({"t0_us": t0_pred}, {"t0_us": t0_ctx}, None, strict_mode=True)
        self.assertEqual(resolved, 5_100_000)
        self.assertEqual(src, "prediction")

        # Conflicting t0
        t0_ctx_bad = 6_000_000
        with self.assertRaises(ConflictingTimeOriginError):
            resolve_time_origin({"t0_us": t0_pred}, {"t0_us": t0_ctx_bad}, None, strict_mode=True)

    def test_B4_missing_t0_raises_in_strict_mode(self):
        """Missing t0 in strict mode raises MissingTimeOriginError without guessing 5_100_000."""
        with self.assertRaises(MissingTimeOriginError):
            resolve_time_origin(None, None, None, strict_mode=True)

    def test_B5_boolean_and_nan_coordinates_rejected(self):
        """Boolean values (True/False) or NaN in waypoint coordinates must be rejected."""
        bool_wps = [
            {"x_m": True if i == 2 else float(i), "y_m": 0.0, "timestamp_micros": self.t0 + (i + 1) * 100_000}
            for i in range(40)
        ]
        with self.assertRaises(TimeContractError):
            validate_and_normalize_timeline(bool_wps, target_poses=40, expected_frequency_hz=10.0, expected_horizon_s=4.0, strict_grid=True)

    # =========================================================================
    # Group C: Run Identity, Comprehensive Fingerprint & Checkpoint Recovery
    # =========================================================================

    def test_C1_tmp_without_manifest_rejected_before_recovery(self):
        """If scores.jsonl.tmp exists without a manifest, resume must be rejected BEFORE recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            score_dir = Path(tmpdir)
            tmp_score = score_dir / "epdms_scores_300.jsonl.tmp"
            tmp_score.write_text('{"record_key": "clip1|cross|0", "valid": true}\n', encoding="utf-8")

            with self.assertRaises(RunIdentityError) as ctx:
                verify_resume_safety_before_recovery(score_dir, expected_fingerprint="dummy_fp_123")
            self.assertIn("Resume rejected", str(ctx.exception))
            self.assertTrue(tmp_score.is_file(), "File must NOT have been moved or modified")

    def test_C2_errors_tmp_without_manifest_rejected(self):
        """If errors.jsonl.tmp exists without a manifest, resume must also be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            score_dir = Path(tmpdir)
            tmp_err = score_dir / "epdms_errors_300.jsonl.tmp"
            tmp_err.write_text('{"record_key": "clip1|cross|0", "valid": false}\n', encoding="utf-8")

            with self.assertRaises(RunIdentityError):
                verify_resume_safety_before_recovery(score_dir, expected_fingerprint="dummy_fp_123")

    def test_C3_fingerprint_mismatch_rejects_resume(self):
        """Existing artifacts with a mismatched manifest fingerprint must be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            score_dir = Path(tmpdir)
            score_file = score_dir / "epdms_scores_300.jsonl"
            score_file.write_text('{"record_key": "c1|m|0"}\n', encoding="utf-8")
            manifest_file = score_dir / "run_manifest.json"
            manifest_file.write_text(json.dumps({"effective_fingerprint": "old_hash_abc"}), encoding="utf-8")

            with self.assertRaises(RunIdentityError) as ctx:
                verify_resume_safety_before_recovery(score_dir, expected_fingerprint="new_hash_xyz")
            self.assertIn("mismatch", str(ctx.exception))

    def test_C4_atomic_manifest_writing(self):
        """write_manifest_atomic safely writes manifest without leaving lingering .tmp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "run_manifest.json"
            write_manifest_atomic(manifest_path, {"status": "RUNNING", "effective_fingerprint": "test_fp"})
            self.assertTrue(manifest_path.is_file())
            self.assertFalse(manifest_path.with_suffix(".tmp").is_file())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "RUNNING")
            self.assertEqual(loaded["effective_fingerprint"], "test_fp")

    # =========================================================================
    # Group D: Map Inventory Partial Error Tracking
    # =========================================================================

    def test_D1_partial_status_on_corrupted_polygon(self):
        """Parquet file with valid polygon AND corrupted polygon must return status='PARTIAL'."""
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_dir = Path(tmpdir) / "clip_d1" / "clipgt"
            clip_dir.mkdir(parents=True)
            # Row 0: valid triangle, Row 1: corrupted NaN vertex
            df = pd.DataFrame([
                {"intersection_area": {"location": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0}, {"x": 5.0, "y": 5.0}]}},
                {"intersection_area": {"location": [{"x": float("nan"), "y": 0.0}, {"x": 1.0, "y": 1.0}]}},
            ])
            df.to_parquet(clip_dir / "intersection_area.parquet")

            status_info = inspect_clip_map_status(Path(tmpdir), "clip_d1")
            self.assertEqual(status_info["status"], "PARTIAL")
            self.assertEqual(status_info["valid_polygon_count"], 1)
            self.assertEqual(status_info["invalid_polygon_count"], 1)
            self.assertFalse(status_info["usable_for_strict_scoring"])

    def test_D2_strict_scoring_rejects_partial_map(self):
        """When map_status is PARTIAL, evaluate_single_condition in strict mode rejects the condition."""
        pred_row = {
            "clip_id": "clip_d2",
            "mode": "cross_scene",
            "alpha": 0.0,
            "t0_us": self.t0,
            "clean_waypoints": [{"x_m": float(i) * 0.1, "y_m": 0.0, "timestamp_micros": self.t0 + (i + 1) * 100_000} for i in range(40)],
        }
        rec = evaluate_single_condition(
            pred_row=pred_row,
            context_row={"semantic_context": {"obstacle": {"all_obstacles": []}}},
            gt_row=None,
            vehicle=self.vehicle,
            lane_polygons=[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])],
            map_status="PARTIAL",
            strict_mode=True,
        )
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "map_geometry_contract")
        self.assertEqual(rec.failure_type, "PartialCorruptedMapError")


if __name__ == "__main__":
    unittest.main()
