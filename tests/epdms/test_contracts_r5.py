"""Round-5 regression tests for time, observation, identity and readiness contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.epdms.observation_contract import evaluate_query_coverage
from tools.epdms.proxy_metrics import compute_collision_free_proxy, compute_ttc_proxy
from tools.epdms.run_identity import (
    AmbiguousCheckpointError,
    compute_run_effective_fingerprint,
    verify_resume_safety_before_recovery,
)
from tools.epdms.schemas import VehicleParameters
from tools.epdms.score_record import evaluate_single_condition
from tools.epdms.time_contract import (
    InconsistentWaypointTimelineError,
    NonMonotonicWaypointTimelineError,
    TimelineHorizonMismatchError,
    TimelineOriginMismatchError,
    normalize_trajectory_timeline,
)


class TestContractsRound5(unittest.TestCase):
    def setUp(self):
        self.t0 = 5_100_000
        self.vehicle = VehicleParameters()
        self.future_ts = [self.t0 + (i + 1) * 100_000 for i in range(40)]

    def _wps(self, timestamp_key="t_s", values=None):
        values = values or [0.1 * (i + 1) for i in range(40)]
        return [{"x_m": i * 0.1, "y_m": 0.0, timestamp_key: t} for i, t in enumerate(values)]

    def test_gt_zero_repeated_rejected(self):
        with self.assertRaises(NonMonotonicWaypointTimelineError):
            normalize_trajectory_timeline(self._wps(values=[0.0] * 40), self.t0, role="ground_truth")

    def test_gt_nan_rejected(self):
        values = [0.1 * (i + 1) for i in range(40)]
        values[3] = float("nan")
        with self.assertRaises(ValueError):
            normalize_trajectory_timeline(self._wps(values=values), self.t0, role="ground_truth")

    def test_gt_string_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            normalize_trajectory_timeline(self._wps(values=["abc"] + [0.1 * (i + 2) for i in range(39)]), self.t0, role="ground_truth")

    def test_absolute_prediction_origin_is_bound_to_t0(self):
        wps = [{"x_m": i * 0.1, "y_m": 0.0, "timestamp_micros": 100_100_000 + i * 100_000} for i in range(40)]
        with self.assertRaises(TimelineOriginMismatchError):
            normalize_trajectory_timeline(wps, self.t0, role="prediction")

    def test_gt_shifted_from_prediction_is_rejected(self):
        wps = [{"x_m": i * 0.1, "y_m": 0.0, "timestamp_micros": self.t0 + 200_000 + i * 100_000} for i in range(40)]
        with self.assertRaises(TimelineOriginMismatchError):
            normalize_trajectory_timeline(wps, self.t0, role="ground_truth")

    def test_zero_start_is_only_valid_when_t0_is_included(self):
        with self.assertRaises(TimelineOriginMismatchError):
            normalize_trajectory_timeline(self._wps(values=[0.0] + [0.1 * (i + 2) for i in range(39)]), self.t0)
        values = [0.0] + [0.1 * (i + 1) for i in range(40)]
        result = normalize_trajectory_timeline(self._wps(values=values), self.t0)
        self.assertTrue(result.includes_t0)

    def test_same_common_grid_has_zero_ade(self):
        pred = self._wps()
        gt = self._wps()
        p = normalize_trajectory_timeline(pred, self.t0, role="prediction")
        g = normalize_trajectory_timeline(gt, self.t0, role="ground_truth")
        self.assertTrue(np.array_equal(p.timestamps_us, g.timestamps_us))
        self.assertAlmostEqual(float(np.mean(np.hypot(p.x - g.x, p.y - g.y))), 0.0)

    def test_different_timestamp_grid_is_not_equal(self):
        pred = normalize_trajectory_timeline(self._wps(), self.t0, role="prediction")
        gt = normalize_trajectory_timeline(self._wps(timestamp_key="t_s", values=[0.1 * (i + 1) for i in range(40)]), self.t0 + 100_000, role="ground_truth")
        self.assertFalse(np.array_equal(pred.timestamps_us, gt.timestamps_us))

    def test_empty_obstacles_without_evidence_are_not_safe(self):
        x = np.zeros(41); y = np.zeros(41); h = np.zeros(41); ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        result = compute_collision_free_proxy(x, y, h, ts, [], self.vehicle, self.t0, strict_mode=True)
        self.assertIsNone(result[0])
        self.assertEqual(result.cf_missing_frames, 41)
        ttc = compute_ttc_proxy(x, y, h, np.zeros(41), ts, [], self.vehicle, self.t0, strict_mode=True)
        self.assertIsNone(ttc[0])

    def test_empty_obstacles_with_explicit_evidence_pass(self):
        x = np.zeros(41); y = np.zeros(41); h = np.zeros(41); ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        result = compute_collision_free_proxy(x, y, h, ts, [], self.vehicle, self.t0, strict_mode=True, confirmed_empty_scene=True)
        self.assertEqual(result[0], 1.0)
        ttc = compute_ttc_proxy(x, y, h, np.zeros(41), ts, [], self.vehicle, self.t0, strict_mode=True, confirmed_empty_scene=True)
        self.assertEqual(ttc[0], 1.0)

    def test_coverage_one_of_forty_one_is_invalid(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        states, observed, empty, missing, ratio = evaluate_query_coverage(ts, {int(ts[0]): [{"x": 1}]}, ts[:1])
        self.assertEqual((observed, empty, missing), (1, 0, 40))
        self.assertLess(ratio, 1.0)

    def test_coverage_forty_one_confirmed_frames_is_valid(self):
        ts = np.asarray([self.t0 + i * 100_000 for i in range(41)], dtype=np.int64)
        states, observed, empty, missing, ratio = evaluate_query_coverage(ts, {}, np.array([], dtype=np.int64), confirmed_empty_timestamps=set(map(int, ts)))
        self.assertEqual((observed, empty, missing), (0, 41, 0))
        self.assertEqual(ratio, 1.0)

    def test_fingerprint_changes_with_implementation_version(self):
        import tools.epdms.run_identity as identity
        base = {"metric_profile": "nurec_safety_proxy_v1", "horizon_s": 4.0, "frequency_hz": 10.0, "strict_mode": True, "proxy": {}, "vehicle": {}}
        original = identity.METRIC_IMPLEMENTATION_VERSION
        try:
            one = compute_run_effective_fingerprint(base, {})
            identity.METRIC_IMPLEMENTATION_VERSION = "changed-for-test"
            two = compute_run_effective_fingerprint(base, {})
        finally:
            identity.METRIC_IMPLEMENTATION_VERSION = original
        self.assertNotEqual(one, two)

    def test_ambiguous_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epdms_scores_300.jsonl").write_text('{"record_key":"A"}\n', encoding="utf-8")
            (root / "epdms_scores_300.jsonl.tmp").write_text('{"record_key":"A"}\n{"record_key":"B"}\n', encoding="utf-8")
            (root / "run_manifest.json").write_text(json.dumps({"effective_fingerprint": "fp"}), encoding="utf-8")
            with self.assertRaises(AmbiguousCheckpointError):
                verify_resume_safety_before_recovery(root, "fp")

    def test_direct_clock_synthetic_scene_scores_end_to_end(self):
        frame = {"coordinate_frame": "ar1_ego", "reference_point": "rear_axle"}
        pred = {"clip_id": "r5", "mode": "cross_scene", "alpha": 0.0, "t0_us": self.t0, **frame, "clean_waypoints": self._wps()}
        gt = {"clip_id": "r5", "t0_us": self.t0, **frame, "expert_future": self._wps()}
        context = {"clip_id": "r5", "t0_us": self.t0, "coordinate_frame": "ar1_ego", "obstacle_frame": "ar1_ego", "map_frame": "ar1_ego", "confirmed_empty_scene": True, "semantic_context": {"obstacle": {"all_obstacles": []}}}
        polygon = [np.asarray([[-20.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-20.0, 20.0]])]
        record = evaluate_single_condition(pred, context, gt, self.vehicle, lane_polygons=polygon, strict_mode=True)
        self.assertTrue(record.valid, record.failure_reason)
        for key in ("collision_free_proxy", "ttc_proxy", "dac_proxy", "progress_gt_proxy", "future_comfort_proxy", "ade_m", "fde_m", "nurec_safety_proxy_v1"):
            self.assertIsNotNone(record.to_dict()[key])


if __name__ == "__main__":
    unittest.main()
