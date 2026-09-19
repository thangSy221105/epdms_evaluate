"""Round-6 contract-consistency and readiness regression tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.epdms.audit import audit_data_contracts, audit_time_alignment
from tools.epdms.config import EvaluationConfig
from tools.epdms.observation_contract import build_ttc_projection_timestamps
from tools.epdms.proxy_metrics import compute_collision_free_proxy, compute_ttc_proxy
from tools.epdms.run_identity import (
    AmbiguousCheckpointError,
    compute_run_effective_fingerprint,
    load_latest_checkpoint_states,
    verify_resume_safety_before_recovery,
)
from tools.epdms.schemas import VehicleParameters
from tools.epdms.score_record import evaluate_single_condition
from tools.epdms.time_contract import (
    TimelineOriginMismatchError,
    normalize_trajectory_timeline,
    prepare_trajectory_window,
)
from scripts.evaluate_epdms import derive_scoring_status


class TestContractsRound6(unittest.TestCase):
    def setUp(self):
        self.t0 = 5_200_000
        self.vehicle = VehicleParameters()

    def _future(self, count=40, absolute=False, include_t0=False, malformed=None):
        start = 0 if include_t0 else 1
        values = list(range(start, count + 1))
        result = []
        for i, step in enumerate(values):
            if malformed == "nan" and i == 2:
                step_value = float("nan")
            else:
                step_value = step
            timestamp = self.t0 + step_value * 100_000 if absolute else step_value / 10.0
            key = "timestamp_micros" if absolute else "t_s"
            result.append({"x_m": float(i) * 0.1, "y_m": 0.0, key: timestamp})
        return result

    def _frames(self, frame="ar1_ego", anchor="rear_axle"):
        return {
            "coordinate_frame": frame,
            "reference_point": anchor,
            "obstacle_frame": frame,
            "map_frame": frame,
            "obstacle_anchor": anchor,
            "map_anchor": anchor,
        }

    def _scene(self, metadata=True, context_extra=None):
        frame = self._frames() if metadata else {}
        pred = {"clip_id": "r6", "mode": "cross_scene", "alpha": 0.0, "t0_us": self.t0, **frame,
                "clean_waypoints": self._future()}
        gt = {"clip_id": "r6", "t0_us": self.t0, **frame, "expert_future": self._future()}
        context = {"clip_id": "r6", "t0_us": self.t0, **frame,
                   "confirmed_empty_scene": True,
                   "semantic_context": {"obstacle": {"all_obstacles": []}}}
        if context_extra:
            context.update(context_extra)
        polygon = [np.asarray([[-20.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-20.0, 20.0]])]
        return pred, context, gt, polygon

    def test_gt_41_including_t0_matches_40_future_grid(self):
        gt = prepare_trajectory_window(self._future(40, include_t0=True), self.t0, 40, "ground_truth")
        pred = normalize_trajectory_timeline(self._future(), self.t0, role="prediction")
        normalized = normalize_trajectory_timeline(gt, self.t0, role="ground_truth")
        self.assertTrue(np.array_equal(pred.timestamps_us, normalized.timestamps_us[1:]))

    def test_gt_65_including_t0_crops_to_horizon(self):
        gt = prepare_trajectory_window(self._future(64, include_t0=True), self.t0, 40, "ground_truth")
        self.assertEqual(len(gt), 41)
        normalized = normalize_trajectory_timeline(gt, self.t0, role="ground_truth")
        self.assertEqual(int(normalized.timestamps_us[-1]), self.t0 + 4_000_000)

    def test_gt_64_future_only_crops_to_horizon(self):
        gt = prepare_trajectory_window(self._future(64), self.t0, 40, "ground_truth")
        self.assertEqual(len(gt), 40)
        normalized = normalize_trajectory_timeline(gt, self.t0, role="ground_truth")
        self.assertEqual(int(normalized.timestamps_us[-1]), self.t0 + 4_000_000)

    def test_prediction_65_including_t0_and_64_future_only(self):
        self.assertEqual(len(prepare_trajectory_window(self._future(64, include_t0=True), self.t0, 40)), 41)
        self.assertEqual(len(prepare_trajectory_window(self._future(64), self.t0, 40)), 40)

    def test_41_with_first_timestamp_01_is_rejected(self):
        with self.assertRaises(TimelineOriginMismatchError):
            prepare_trajectory_window(self._future(41), self.t0, 40, "ground_truth")

    def test_strict_missing_coordinate_metadata_is_invalid(self):
        pred, context, gt, polygon = self._scene(metadata=False)
        rec = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_strict_partial_coordinate_metadata_is_invalid(self):
        pred, context, gt, polygon = self._scene(metadata=True)
        context.pop("map_frame")
        rec = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_strict_matching_frames_and_anchors_pass(self):
        pred, context, gt, polygon = self._scene(metadata=True)
        rec = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertTrue(rec.valid, rec.failure_reason)

    def test_strict_obstacle_frame_difference_is_invalid(self):
        pred, context, gt, polygon = self._scene(metadata=True)
        context["obstacle_frame"] = "other_frame"
        rec = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_strict_map_frame_difference_is_invalid(self):
        pred, context, gt, polygon = self._scene(metadata=True)
        context["map_frame"] = "other_frame"
        rec = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_type, "COORDINATE_CONTRACT_UNRESOLVED")

    def test_empty_obstacles_full_confirmed_empty_frames_pass(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        result = compute_collision_free_proxy(np.zeros(41), np.zeros(41), np.zeros(41), ts, [], self.vehicle, self.t0, strict_mode=True, confirmed_empty_timestamps=set(map(int, ts)))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result.cf_confirmed_empty_frames, 41)

    def test_ttc_full_confirmed_empty_projection_passes(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        projected = build_ttc_projection_timestamps(ts, 1.0)
        result = compute_ttc_proxy(np.zeros(41), np.zeros(41), np.zeros(41), np.zeros(41), ts, [], self.vehicle, self.t0, strict_mode=True, confirmed_empty_timestamps=set(map(int, projected)))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result.ttc_confirmed_empty_observations, len(projected))

    def test_partial_confirmed_empty_frames_are_invalid(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        result = compute_collision_free_proxy(np.zeros(41), np.zeros(41), np.zeros(41), ts, [], self.vehicle, self.t0, strict_mode=True, confirmed_empty_timestamps=set(map(int, ts[:-1])))
        self.assertIsNone(result[0])

    def test_object_outside_cf_horizon_with_empty_attestation_passes_cf(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        obstacle = {"timestamp_micros": self.t0 + 5_000_000, "trackline_id": "outside", "center": {"x": 100.0, "y": 100.0}, "size": {"x": 2.0, "y": 2.0}, "orientation": {"w": 1.0}}
        result = compute_collision_free_proxy(np.zeros(41), np.zeros(41), np.zeros(41), ts, [obstacle], self.vehicle, self.t0, strict_mode=True, confirmed_empty_timestamps=set(map(int, ts)))
        self.assertEqual(result[0], 1.0)

    def test_audit_malformed_gt_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred_path, ctx_path, gt_path = root / "pred.jsonl", root / "ctx.jsonl", root / "gt.jsonl"
            pred, ctx, gt, _ = self._scene(metadata=True)
            gt["expert_future"] = self._future(40, malformed="nan")
            pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")
            ctx_path.write_text(json.dumps(ctx) + "\n", encoding="utf-8")
            gt_path.write_text(json.dumps(gt) + "\n", encoding="utf-8")
            cfg = EvaluationConfig({"paths": {"prediction_jsonl": str(pred_path), "context_jsonl": str(ctx_path), "ground_truth_jsonl": str(gt_path)}, "modes": ["cross_scene"], "alphas": [0.0]})
            report = audit_data_contracts(cfg)
            self.assertFalse(report["readiness"]["GT_TIMELINE_READY"])
            self.assertGreater(report["gt_timeline_stats"]["issue_count"], 0)

    def test_audit_one_of_41_observation_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, ctx, gt, _ = self._scene(metadata=True)
            obstacle = {"timestamp_micros": self.t0, "trackline_id": "one", "center": {"x": 100.0, "y": 100.0}, "size": {"x": 2.0, "y": 2.0}, "orientation": {"w": 1.0}}
            ctx["confirmed_empty_scene"] = False
            ctx["semantic_context"]["obstacle"]["all_obstacles"] = [obstacle]
            paths = {name: root / name for name in ("p", "c", "g")}
            paths["p"].write_text(json.dumps(pred) + "\n", encoding="utf-8")
            paths["c"].write_text(json.dumps(ctx) + "\n", encoding="utf-8")
            paths["g"].write_text(json.dumps(gt) + "\n", encoding="utf-8")
            cfg = EvaluationConfig({"paths": {"prediction_jsonl": str(paths["p"]), "context_jsonl": str(paths["c"]), "ground_truth_jsonl": str(paths["g"])}, "modes": ["cross_scene"], "alphas": [0.0]})
            report = audit_data_contracts(cfg)
            row = report["observation_coverage_stats"]["rows"][0]
            self.assertEqual(row["cf_observed_frames"], 1)
            self.assertGreater(row["cf_missing_frames"], 0)
            self.assertFalse(report["readiness"]["OBSERVATION_COVERAGE_CONTRACT_READY"])

    def test_audit_full_confirmed_empty_observation_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, ctx, gt, _ = self._scene(metadata=True)
            ctx["confirmed_empty_scene"] = True
            paths = {name: root / name for name in ("p", "c", "g")}
            paths["p"].write_text(json.dumps(pred) + "\n", encoding="utf-8")
            paths["c"].write_text(json.dumps(ctx) + "\n", encoding="utf-8")
            paths["g"].write_text(json.dumps(gt) + "\n", encoding="utf-8")
            cfg = EvaluationConfig({"paths": {"prediction_jsonl": str(paths["p"]), "context_jsonl": str(paths["c"]), "ground_truth_jsonl": str(paths["g"])}, "modes": ["cross_scene"], "alphas": [0.0]})
            report = audit_data_contracts(cfg)
            self.assertTrue(report["readiness"]["OBSERVATION_COVERAGE_CONTRACT_READY"])

    def test_audit_relative_and_absolute_timestamp_representations_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, ctx, gt, _ = self._scene(metadata=True)
            paths = {name: root / name for name in ("p", "c", "g")}
            paths["p"].write_text(json.dumps(pred) + "\n", encoding="utf-8")
            paths["c"].write_text(json.dumps(ctx) + "\n", encoding="utf-8")
            paths["g"].write_text(json.dumps(gt) + "\n", encoding="utf-8")
            cfg = EvaluationConfig({"paths": {"prediction_jsonl": str(paths["p"]), "context_jsonl": str(paths["c"]), "ground_truth_jsonl": str(paths["g"])}})
            relative = audit_time_alignment(cfg)["rows"][0]
            pred_abs, ctx_abs, gt_abs, _ = self._scene(metadata=True)
            pred_abs["clean_waypoints"] = self._future(40, absolute=True)
            gt_abs["expert_future"] = self._future(40, absolute=True)
            paths["p"].write_text(json.dumps(pred_abs) + "\n", encoding="utf-8")
            paths["g"].write_text(json.dumps(gt_abs) + "\n", encoding="utf-8")
            absolute = audit_time_alignment(cfg)["rows"][0]
            self.assertEqual(relative["prediction_first_time_us"], absolute["prediction_first_time_us"])
            self.assertEqual(relative["prediction_last_time_us"], absolute["prediction_last_time_us"])
            self.assertEqual(relative["time_alignment_status"], absolute["time_alignment_status"])

    def test_fingerprint_changes_with_clip_scope(self):
        base = {"metric_profile": "nurec_safety_proxy_v1", "horizon_s": 4.0, "frequency_hz": 10.0, "strict_mode": True, "proxy": {}, "vehicle": {}}
        self.assertNotEqual(compute_run_effective_fingerprint(base, {}, clip_scope=["a", "b", "c"]), compute_run_effective_fingerprint(base, {}, clip_scope=["a", "b", "c", "d"]))
        self.assertNotEqual(compute_run_effective_fingerprint(base, {}, clip_scope=["a", "b", "c"]), compute_run_effective_fingerprint(base, {}, clip_scope=["a", "b", "d"]))
        self.assertEqual(compute_run_effective_fingerprint(base, {}, clip_scope=["c", "a"]), compute_run_effective_fingerprint(base, {}, clip_scope=["a", "c"]))

    def test_manifest_status_partial_and_blocked(self):
        self.assertEqual(derive_scoring_status(48, 48, 0, 0), "COMPLETE")
        self.assertEqual(derive_scoring_status(48, 1, 47, 0), "PARTIAL")
        self.assertEqual(derive_scoring_status(48, 0, 48, 0), "BLOCKED")
        self.assertEqual(derive_scoring_status(48, 30, 10, 8), "PARTIAL")

    def test_retry_attempt_history_reaches_three_and_latest_valid_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epdms_errors_300.jsonl").write_text(json.dumps({"record_key": "A", "valid": False, "attempt_number": 1}) + "\n", encoding="utf-8")
            (root / "epdms_errors_300_attempt_2.jsonl").write_text(json.dumps({"record_key": "A", "valid": False, "attempt_number": 2}) + "\n", encoding="utf-8")
            (root / "epdms_errors_300_attempt_3.jsonl").write_text(json.dumps({"record_key": "A", "valid": True, "attempt_number": 3}) + "\n", encoding="utf-8")
            states = load_latest_checkpoint_states(root)
            self.assertEqual(states["A"]["attempt_number"], 3)
            self.assertTrue(states["A"]["valid"])

    def test_duplicate_inside_target_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epdms_scores_300.jsonl").write_text('{"record_key":"A"}\n{"record_key":"A"}\n', encoding="utf-8")
            (root / "run_manifest.json").write_text('{"effective_fingerprint":"fp"}', encoding="utf-8")
            with self.assertRaises(AmbiguousCheckpointError):
                verify_resume_safety_before_recovery(root, "fp")

    def test_duplicate_inside_tmp_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epdms_scores_300.jsonl.tmp").write_text('{"record_key":"B"}\n{"record_key":"B"}\n', encoding="utf-8")
            (root / "run_manifest.json").write_text('{"effective_fingerprint":"fp"}', encoding="utf-8")
            with self.assertRaises(AmbiguousCheckpointError):
                verify_resume_safety_before_recovery(root, "fp")

    def test_disjoint_target_tmp_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epdms_scores_300.jsonl").write_text('{"record_key":"A"}\n', encoding="utf-8")
            (root / "epdms_scores_300.jsonl.tmp").write_text('{"record_key":"B"}\n', encoding="utf-8")
            (root / "run_manifest.json").write_text('{"effective_fingerprint":"fp"}', encoding="utf-8")
            self.assertIsNotNone(verify_resume_safety_before_recovery(root, "fp"))


if __name__ == "__main__":
    unittest.main()
