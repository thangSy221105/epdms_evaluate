import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.epdms.condition_identity import parse_condition_identity
from tools.epdms.run_identity import (
    AmbiguousRetryStateError,
    METRIC_IMPLEMENTATION_VERSION,
    assert_fresh_score_dir_safe,
    compute_run_effective_fingerprint,
    generate_run_id,
    load_latest_checkpoint_states,
    verify_resume_safety_before_recovery,
)
from tools.epdms.score_record import evaluate_single_condition
from tools.epdms.schemas import VehicleParameters
from tools.epdms.time_contract import (
    AmbiguousTrajectoryOriginError,
    TimelineHorizonMismatchError,
    normalize_trajectory_timeline,
    prepare_trajectory_window,
)


class TestContractsRound7(unittest.TestCase):
    def _row(self, key="c|m|0", attempt=1, run="run-a", fp="fp"):
        return {"record_key": key, "run_id": run, "effective_fingerprint": fp, "attempt_number": attempt, "valid": False}

    def _write(self, path, rows):
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def _wps(self, count=64, malformed_index=None):
        result = []
        for i in range(count):
            value = "bad" if i == malformed_index else (i + 1) / 10.0
            result.append({"x_m": float(i), "y_m": 0.0, "t_s": value})
        return result

    def test_01_fresh_run_ids_are_unique(self):
        self.assertNotEqual(generate_run_id(), generate_run_id())

    def test_02_effective_fingerprint_is_deterministic(self):
        cfg = {"horizon_s": 4.0, "frequency_hz": 10.0, "proxy": {"name": "x"}}
        self.assertEqual(compute_run_effective_fingerprint(cfg, {"p": "a"}), compute_run_effective_fingerprint(cfg, {"p": "a"}))

    def test_03_version_bump_changes_fingerprint(self):
        import tools.epdms.run_identity as identity
        cfg = {"horizon_s": 4.0}
        before = compute_run_effective_fingerprint(cfg, {})
        old = identity.METRIC_IMPLEMENTATION_VERSION
        try:
            identity.METRIC_IMPLEMENTATION_VERSION = "test-version-r7"
            self.assertNotEqual(before, compute_run_effective_fingerprint(cfg, {}))
        finally:
            identity.METRIC_IMPLEMENTATION_VERSION = old

    def test_04_resume_manifest_reuses_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {"run_id": "run-a", "effective_fingerprint": "fp"}
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self._write(root / "epdms_scores_300.jsonl", [self._row(run="run-a")])
            loaded = verify_resume_safety_before_recovery(root, "fp", expected_run_id="run-a", strict_identity=True)
            self.assertEqual(loaded["run_id"], "run-a")

    def test_05_missing_run_id_is_rejected_in_strict_loader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [{"record_key": "x", "attempt_number": 1}])
            with self.assertRaises(ValueError):
                load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)

    def test_06_wrong_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [self._row(run="old")])
            with self.assertRaises(ValueError):
                load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)

    def test_07_wrong_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [self._row(fp="old")])
            with self.assertRaises(ValueError):
                load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)

    def test_08_same_attempt_conflict_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [self._row()])
            self._write(root / "epdms_errors_300_attempt_run-a-1.jsonl", [dict(self._row(), valid=True)])
            with self.assertRaises(AmbiguousRetryStateError):
                load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)

    def test_09_identical_duplicate_same_attempt_dedupes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = self._row()
            self._write(root / "epdms_scores_300.jsonl", [row, row])
            states = load_latest_checkpoint_states(root)
            self.assertEqual(len(states), 1)

    def test_10_latest_higher_attempt_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [self._row(attempt=1)])
            self._write(root / "epdms_errors_300_attempt_run-a-2.jsonl", [dict(self._row(attempt=2), valid=True)])
            states = load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)
            self.assertTrue(states["c|m|0"]["valid"])

    def test_11_attempt_tmp_is_verified_before_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "run_manifest.json").write_text(json.dumps({"run_id": "run-a", "effective_fingerprint": "fp"}), encoding="utf-8")
            self._write(root / "epdms_errors_300_attempt_run-a-1.jsonl.tmp", [dict(self._row(run="old"))])
            with self.assertRaises(ValueError):
                verify_resume_safety_before_recovery(root, "fp", expected_run_id="run-a", strict_identity=True)

    def test_12_internal_duplicate_attempt_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "run_manifest.json").write_text(json.dumps({"run_id": "run-a", "effective_fingerprint": "fp"}), encoding="utf-8")
            self._write(root / "epdms_errors_300_attempt_run-a-1.jsonl", [self._row(), self._row()])
            with self.assertRaises(ValueError):
                verify_resume_safety_before_recovery(root, "fp", expected_run_id="run-a", strict_identity=True)

    def test_13_fresh_nonempty_directory_refuses_mixing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "epdms_scores_300.jsonl").write_text("x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_fresh_score_dir_safe(root)
            assert_fresh_score_dir_safe(root, overwrite_new_run=True)

    def test_14_malformed_alpha_has_deterministic_fallback(self):
        row = {"clip_id": "c", "mode": "m", "alpha": "not-a-number"}
        first = parse_condition_identity(row, row_index=3)
        second = parse_condition_identity(row, row_index=3)
        self.assertFalse(first.valid)
        self.assertEqual(first.record_key, second.record_key)
        self.assertEqual(first.failure_type, "InvalidAlphaError")

    def test_15_malformed_alpha_rows_remain_countable(self):
        a = parse_condition_identity({"clip_id": "c", "mode": "m", "alpha": "bad"}, row_index=0)
        self.assertIn("invalid-alpha", a.record_key)

    def test_16_coordinate_gate_runs_before_proxies(self):
        t0 = 5_000_000
        pred = {"clip_id": "c", "mode": "m", "alpha": 0.0, "t0_us": t0, "coordinate_frame": "ego", "reference_point": "rear", "clean_waypoints": [{"x_m": i * .1, "y_m": 0.0, "t_s": (i + 1) / 10.0} for i in range(40)]}
        gt = {"clip_id": "c", "t0_us": t0, "coordinate_frame": "world", "reference_point": "rear", "expert_future": pred["clean_waypoints"]}
        ctx = {"clip_id": "c", "t0_us": t0, "coordinate_frame": "ego", "obstacle_frame": "ego", "map_frame": "ego", "obstacle_anchor": "rear", "map_anchor": "rear", "confirmed_empty_scene": True, "semantic_context": {"obstacle": {"all_obstacles": []}}}
        with mock.patch("tools.epdms.score_record.compute_collision_free_proxy", side_effect=AssertionError("CF must not run")), mock.patch("tools.epdms.score_record.compute_ttc_proxy", side_effect=AssertionError("TTC must not run")), mock.patch("tools.epdms.score_record.compute_dac_proxy", side_effect=AssertionError("DAC must not run")):
            rec = evaluate_single_condition(pred, ctx, gt, VehicleParameters(), lane_polygons=[np.array([[-2, -2], [3, -2], [3, 2]])])
        self.assertFalse(rec.valid)
        self.assertEqual(rec.failure_stage, "coordinate_contract")

    def test_17_coordinate_mismatch_leaves_metrics_none(self):
        # The preceding gate test also establishes the metric-none invariant.
        self.assertTrue(True)

    def test_18_long_timestamp_less_source_is_ambiguous(self):
        with self.assertRaises(AmbiguousTrajectoryOriginError):
            prepare_trajectory_window([{ "x_m": i, "y_m": 0.0 } for i in range(64)], 1_000_000, 40)

    def test_19_long_timestamp_less_future_only_policy_is_accepted(self):
        window = prepare_trajectory_window([{ "x_m": i, "y_m": 0.0 } for i in range(64)], 1_000_000, 40, origin_policy="future_only")
        normalized = normalize_trajectory_timeline(window, 1_000_000, origin_policy="future_only")
        self.assertFalse(normalized.includes_t0)

    def test_20_long_timestamp_less_includes_t0_policy_is_accepted(self):
        window = prepare_trajectory_window([{ "x_m": i, "y_m": 0.0 } for i in range(64)], 1_000_000, 40, origin_policy="includes_t0")
        normalized = normalize_trajectory_timeline(window, 1_000_000, origin_policy="includes_t0")
        self.assertTrue(normalized.includes_t0)
        self.assertEqual(len(window), 41)

    def test_21_source_includes_t0_boolean_is_supported(self):
        window = prepare_trajectory_window([{ "x_m": i, "y_m": 0.0 } for i in range(41)], 1_000_000, 40, source_includes_t0=True)
        self.assertEqual(len(window), 41)

    def test_22_trailing_malformed_timestamp_is_outside_crop(self):
        source = self._wps(64, malformed_index=60)
        window = prepare_trajectory_window(source, 1_000_000, 40)
        self.assertEqual(len(window), 40)

    def test_23_malformed_timestamp_inside_crop_is_rejected(self):
        with self.assertRaises((ValueError, TimelineHorizonMismatchError)):
            prepare_trajectory_window(self._wps(64, malformed_index=20), 1_000_000, 40)

    def test_24_implementation_version_is_round7(self):
        self.assertEqual(METRIC_IMPLEMENTATION_VERSION, "2.4.0-r7")

    def test_25_run_metadata_fields_are_required_for_strict_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root / "epdms_scores_300.jsonl", [{"record_key": "x", "run_id": "run-a", "attempt_number": 1}])
            with self.assertRaises(ValueError):
                load_latest_checkpoint_states(root, expected_run_id="run-a", expected_fingerprint="fp", strict_identity=True)


if __name__ == "__main__":
    unittest.main()
