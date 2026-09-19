import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.data_prep import nurec


class TestNuRecConditionCoverage(unittest.TestCase):
    def _condition(self, clip="clip-a", mode="cross_scene", alpha=0.0, t0=100):
        return {"clip_id": clip, "mode": mode, "alpha": alpha, "t0_us": t0, "coordinate_frame": "ego", "reference_point": "rear"}

    def _run_audit(self, root, prediction_rows, evaluation_config=None):
        clip = root / "clip-a"
        (clip / "clipgt").mkdir(parents=True)
        pred = root / "pred.jsonl"
        gt = root / "gt.jsonl"
        pred.write_text("\n".join(json.dumps(row) for row in prediction_rows) + "\n", encoding="utf-8")
        gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100, "future_frame": "ego"}) + "\n", encoding="utf-8")
        with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "OK", "row_count": 0, "min": 100, "max": 200, "unique_count": 0, "field": "key.timestamp_micros"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 100, "max": 200}), mock.patch.object(nurec, "_parquet_timestamp_values", return_value=set()), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
            return nurec.audit_dataset(root, pred, gt, root / "audit", evaluation_config=evaluation_config)

    def test_01_sixteen_conditions_same_clip_are_kept(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            rows = [self._condition(alpha=alpha, mode=mode) for mode in ("cross_scene", "noisy", "opposite_action", "no_reasoning") for alpha in (0, 0.5, 1, 2)]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result, errors, duplicates, _, stats = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 16)
            self.assertEqual(stats["clip-a"], {"raw_condition_count": 16, "unique_condition_count": 16, "duplicate_condition_count": 0})
            self.assertEqual(duplicates, [])
            self.assertEqual(errors, [])

    def test_02_same_clip_different_mode_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text("\n".join(json.dumps(self._condition(mode=mode)) for mode in ("cross_scene", "noisy")) + "\n", encoding="utf-8")
            result, _, duplicates, _, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 2)
            self.assertFalse(duplicates)

    def test_03_same_clip_mode_different_alpha_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text("\n".join(json.dumps(self._condition(alpha=alpha)) for alpha in (0, 0.5)) + "\n", encoding="utf-8")
            result, _, duplicates, _, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 2)
            self.assertFalse(duplicates)

    def test_04_exact_condition_duplicate_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            row = self._condition(alpha=0.5)
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            result, errors, duplicates, _, stats = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 1)
            self.assertEqual(duplicates, ["clip-a|cross_scene|0.5"])
            self.assertEqual(errors[-1]["failure_type"], "PREDICTION_CONDITION_DUPLICATE")
            self.assertEqual(stats["clip-a"]["raw_condition_count"], 2)
            self.assertEqual(stats["clip-a"]["unique_condition_count"], 1)

    def test_05_missing_mode_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "alpha": 0}) + "\n", encoding="utf-8")
            result, errors, _, identity_errors, _ = nurec._load_prediction_conditions(path)
            self.assertFalse(result)
            self.assertEqual(errors[0]["failure_type"], "PREDICTION_IDENTITY_INVALID")
            self.assertIn("MISSING_MODE", errors[0]["failure_reason"])
            self.assertIn("clip-a", identity_errors)

    def test_06_missing_alpha_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "mode": "noisy"}) + "\n", encoding="utf-8")
            _, errors, _, _, _ = nurec._load_prediction_conditions(path)
            self.assertIn("MISSING_ALPHA", errors[0]["failure_reason"])

    def test_07_nonfinite_alpha_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "mode": "noisy", "alpha": "NaN"}) + "\n", encoding="utf-8")
            _, errors, _, _, _ = nurec._load_prediction_conditions(path)
            self.assertIn("INVALID_ALPHA", errors[0]["failure_reason"])

    def test_08_common_t0_is_returned(self):
        common, values, consistent = nurec._common_condition_value([self._condition(), self._condition(alpha=1)], "t0_us")
        self.assertEqual(common, 100)
        self.assertEqual(values, [100])
        self.assertTrue(consistent)

    def test_09_conflicting_t0_values_are_not_collapsed(self):
        common, values, consistent = nurec._common_condition_value([self._condition(), self._condition(alpha=1, t0=200)], "t0_us")
        self.assertIsNone(common)
        self.assertEqual(values, [100, 200])
        self.assertFalse(consistent)

    def test_09b_conflicting_t0_conditions_block_clip(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run_audit(Path(td), [self._condition(), self._condition(mode="noisy", t0=200)])
            self.assertIn("PREDICTION_T0_CONFLICT", result["contracts"]["clip-a"]["blockers"])

    def test_09c_conflicting_prediction_frames_block_clip(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [self._condition(), {**self._condition(mode="noisy"), "coordinate_frame": "world"}]
            result = self._run_audit(Path(td), rows)
            self.assertIn("PREDICTION_COORDINATE_METADATA_CONFLICT", result["contracts"]["clip-a"]["blockers"])

    def test_10_zero_evidence_is_incomplete(self):
        result = nurec._coverage_counts(list(range(41)), set(), set(), True, 50_000)
        self.assertEqual((result["observed"], result["empty"], result["missing"]), (0, 0, 41))

    def test_11_one_evidence_is_partial(self):
        result = nurec._coverage_counts(list(range(41)), {0}, set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (1, 40))

    def test_12_two_empty_evidence_frames_do_not_become_full(self):
        result = nurec._coverage_counts(list(range(41)), {0, 1}, set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (2, 39))

    def test_13_forty_of_forty_one_evidence_frames(self):
        result = nurec._coverage_counts(list(range(41)), set(range(40)), set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (40, 1))

    def test_14_full_cf_evidence_is_complete(self):
        result = nurec._coverage_counts(list(range(41)), set(range(41)), set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (41, 0))

    def test_15_duplicate_evidence_does_not_increase_count(self):
        result = nurec._coverage_counts(list(range(41)), {0, 0, 1}, set(), True, 0)
        self.assertEqual(result["matched"], 2)

    def test_16_outside_horizon_evidence_does_not_match(self):
        result = nurec._coverage_counts(list(range(41)), {1000}, set(), True, 0)
        self.assertEqual(result["matched"], 0)

    def test_17_partial_ttc_coverage_is_counted(self):
        required = list(range(51))
        result = nurec._coverage_counts(required, set(range(50)), set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (50, 1))

    def test_18_full_ttc_coverage_is_counted(self):
        required = list(range(51))
        result = nurec._coverage_counts(required, set(required), set(), True, 0)
        self.assertEqual((result["empty"], result["missing"]), (51, 0))

    def test_19_obstacle_timestamps_without_evidence_are_observed(self):
        result = nurec._coverage_counts([1, 2], set(), {1, 2}, True, 0)
        self.assertEqual((result["observed"], result["missing"]), (2, 0))

    def test_20_unverified_completeness_is_unknown_not_empty(self):
        result = nurec._coverage_counts([1, 2], {1, 2}, set(), False, 0)
        self.assertEqual((result["empty"], result["unknown"]), (0, 2))

    def test_21_complete_empty_evidence_is_empty(self):
        result = nurec._coverage_counts([1, 2], {1, 2}, set(), True, 0)
        self.assertEqual((result["empty"], result["unknown"]), (2, 0))

    def test_22_objects_and_empty_frames_are_separated(self):
        result = nurec._coverage_counts([1, 2, 3], {1, 2, 3}, {2}, True, 0)
        self.assertEqual((result["observed"], result["empty"]), (1, 2))

    def test_23_none_clip_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gt.jsonl"
            path.write_text(json.dumps({"clip_id": None}) + "\n", encoding="utf-8")
            _, errors, _ = nurec._jsonl_index_with_errors(path, "ground_truth")
            self.assertEqual(errors[0]["failure_type"], "MISSING_CLIP_ID")

    def test_24_whitespace_clip_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gt.jsonl"
            path.write_text(json.dumps({"clip_id": "   "}) + "\n", encoding="utf-8")
            _, errors, _ = nurec._jsonl_index_with_errors(path, "ground_truth")
            self.assertEqual(errors[0]["failure_type"], "MISSING_CLIP_ID")

    def test_25_empty_evidence_requires_exact_timestamp(self):
        result = nurec._coverage_counts([100_000], {100_040}, set(), True, 50)
        self.assertEqual((result["empty"], result["missing"]), (0, 1))

    def test_26_object_timestamp_uses_tolerance(self):
        result = nurec._coverage_counts([100_000], {100_000}, {100_040}, True, 50_000)
        self.assertEqual(result["observed"], 1)

    def test_27_object_timestamp_outside_tolerance_is_missing(self):
        result = nurec._coverage_counts([100_000], set(), {100_051}, True, 50)
        self.assertEqual(result["missing"], 1)

    def test_28_coverage_matches_scorer_exact_empty_semantics(self):
        from tools.epdms.observation_contract import evaluate_query_coverage
        query = np.asarray([100_000, 200_000], dtype=np.int64)
        obs_by_time = {200_049: [{}]}
        obs_ts = np.asarray([200_049], dtype=np.int64)
        _, observed, empty, missing, _ = evaluate_query_coverage(query, obs_by_time, obs_ts, {100_040}, half_step_us=50_000)
        result = nurec._coverage_counts([100_000, 200_000], {100_040, 200_000}, {200_049}, True, 50_000)
        self.assertEqual((result["observed"], result["empty"], result["missing"]), (observed, empty, missing))

    def test_28b_object_wins_over_empty_when_both_cover_query(self):
        result = nurec._coverage_counts([100_000], {100_000}, {100_049}, True, 50_000)
        self.assertEqual((result["observed"], result["empty"]), (1, 0))

    def test_28c_ttc_empty_requires_exact_timestamp(self):
        result = nurec._coverage_counts([100_000], {100_099}, set(), True, 100_000)
        self.assertEqual((result["empty"], result["missing"]), (0, 1))

    def test_28d_ttc_object_uses_100ms_tolerance(self):
        result = nurec._coverage_counts([100_000], {100_000}, {100_099}, True, 100_000)
        self.assertEqual(result["observed"], 1)

    def test_28e_ttc_object_over_100ms_is_missing(self):
        result = nurec._coverage_counts([100_000], set(), {200_001}, True, 100_000)
        self.assertEqual(result["missing"], 1)

    def test_28f_cf_object_uses_tolerance_without_frame_evidence(self):
        result = nurec._coverage_counts([100_000], set(), {100_040}, True, 50_000)
        self.assertEqual((result["observed"], result["missing"]), (1, 0))

    def test_28g_cf_object_outside_tolerance_without_frame_evidence_is_missing(self):
        result = nurec._coverage_counts([100_000], set(), {151_000}, True, 50_000)
        self.assertEqual((result["observed"], result["missing"]), (0, 1))

    def test_28h_ttc_object_uses_tolerance_without_frame_evidence(self):
        result = nurec._coverage_counts([100_000], set(), {199_000}, True, 100_000)
        self.assertEqual((result["observed"], result["missing"]), (1, 0))

    def test_28i_ttc_object_outside_tolerance_without_frame_evidence_is_missing(self):
        result = nurec._coverage_counts([100_000], set(), {201_000}, True, 100_000)
        self.assertEqual((result["observed"], result["missing"]), (0, 1))

    def test_28j_exact_empty_is_not_tolerance_matched(self):
        result = nurec._coverage_counts([100_000], {100_001}, set(), True, 50_000)
        self.assertEqual((result["empty"], result["missing"]), (0, 1))

    def test_28k_incomplete_object_is_observed(self):
        result = nurec._coverage_counts([100_000], set(), {100_000}, False, 50_000)
        self.assertEqual((result["observed"], result["unknown"], result["missing"]), (1, 0, 0))

    def test_28l_incomplete_empty_evidence_is_unknown(self):
        result = nurec._coverage_counts([100_000], {100_000}, set(), False, 50_000)
        self.assertEqual((result["empty"], result["unknown"], result["missing"]), (0, 1, 0))

    def test_28m_mixed_coverage_matches_scorer(self):
        result = nurec._coverage_counts([100_000, 200_000, 300_000], {300_000}, {100_040, 200_000}, True, 50_000)
        self.assertEqual((result["observed"], result["empty"], result["unknown"], result["missing"]), (2, 1, 0, 0))
        self.assertEqual(result["observed"] + result["empty"] + result["unknown"] + result["missing"], result["required"])

    def test_29_default_query_grid_provenance_is_explicit(self):
        cf, ttc, settings = nurec._canonical_query_timestamps(0)
        self.assertEqual(len(cf), 41)
        self.assertEqual(len(ttc), 51)
        self.assertEqual(settings["source"], "DEFAULT_EVALUATION_CONFIG")
        self.assertFalse(settings["verified"])

    def test_30_nondefault_horizon_changes_cf_grid(self):
        cf, _, settings = nurec._canonical_query_timestamps(0, {"horizon_s": 2.0, "frequency_hz": 10.0, "future_poses": 20, "ttc_horizon_s": 1.0, "source": "test", "verified": True})
        self.assertEqual(len(cf), 21)
        self.assertEqual(settings["horizon_s"], 2.0)

    def test_31_nondefault_frequency_changes_cf_grid(self):
        cf, _, _ = nurec._canonical_query_timestamps(0, {"horizon_s": 4.0, "frequency_hz": 5.0, "future_poses": 20, "ttc_horizon_s": 1.0, "source": "test", "verified": True})
        self.assertEqual(len(cf), 21)

    def test_32_nondefault_ttc_horizon_changes_projection_grid(self):
        _, ttc, _ = nurec._canonical_query_timestamps(0, {"horizon_s": 4.0, "frequency_hz": 10.0, "future_poses": 40, "ttc_horizon_s": 0.5, "source": "test", "verified": True})
        self.assertEqual(len(ttc), 45)

    def test_33_invalid_query_settings_fail_closed(self):
        with self.assertRaises(ValueError):
            nurec._canonical_query_timestamps(0, {"horizon_s": 0, "frequency_hz": 10, "future_poses": 0, "ttc_horizon_s": 1, "source": "test", "verified": True})

    def test_34_query_grid_provenance_is_written_to_contract(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run_audit(Path(td), [self._condition()], {"horizon_s": 2.0, "frequency_hz": 10.0, "future_poses": 20, "ttc_horizon_s": 0.5, "source": "test-config", "verified": True})
            grid = result["contracts"]["clip-a"]["query_grid"]
            self.assertEqual((grid["horizon_s"], grid["cf_required_frames"], grid["ttc_horizon_s"]), (2.0, 21, 0.5))
            self.assertEqual(grid["source"], "test-config")

    def test_35_no_config_adds_unverified_grid_blocker(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run_audit(Path(td), [self._condition()])
            contract = result["contracts"]["clip-a"]
            self.assertFalse(contract["query_grid"]["verified"])
            self.assertIn("QUERY_GRID_CONFIG_UNVERIFIED", contract["blockers"])
            self.assertFalse(contract["ready_for_proxy"])

    def test_36_verified_config_has_no_unverified_grid_blocker(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run_audit(Path(td), [self._condition()], {"horizon_s": 4.0, "frequency_hz": 10.0, "future_poses": 40, "ttc_horizon_s": 1.0, "source": "config.json", "verified": True})
            contract = result["contracts"]["clip-a"]
            self.assertTrue(contract["query_grid"]["verified"])
            self.assertNotIn("QUERY_GRID_CONFIG_UNVERIFIED", contract["blockers"])

    def test_37_invalid_config_adds_contract_blocker(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run_audit(Path(td), [self._condition()], {"horizon_s": 0, "frequency_hz": 10.0, "future_poses": 0, "ttc_horizon_s": 1.0, "source": "bad.json", "verified": True})
            self.assertIn("QUERY_GRID_CONTRACT_UNRESOLVED", result["contracts"]["clip-a"]["blockers"])

    def test_38_full_object_only_cf_is_ready_without_completeness(self):
        result = nurec._coverage_counts([100_000] * 41, set(), {100_000}, False, 50_000)
        self.assertTrue(nurec._coverage_ready(True, True, result))
        self.assertEqual((result["observed"], result["empty"], result["unknown"], result["missing"]), (41, 0, 0, 0))

    def test_39_full_object_only_ttc_is_ready_without_completeness(self):
        result = nurec._coverage_counts([100_000] * 51, set(), {100_000}, False, 100_000)
        self.assertTrue(nurec._coverage_ready(True, True, result))
        self.assertEqual((result["observed"], result["empty"], result["unknown"], result["missing"]), (51, 0, 0, 0))

    def test_40_partial_object_coverage_is_not_ready(self):
        result = nurec._coverage_counts([100_000, 200_000], set(), {100_000}, False, 0)
        self.assertFalse(nurec._coverage_ready(True, True, result))
        self.assertEqual((result["observed"], result["missing"]), (1, 1))

    def test_41_object_plus_unknown_frame_is_not_ready(self):
        result = nurec._coverage_counts([100_000, 200_000], {200_000}, {100_000}, False, 0)
        self.assertFalse(nurec._coverage_ready(True, True, result))
        self.assertEqual((result["observed"], result["unknown"], result["missing"]), (1, 1, 0))

    def test_42_object_plus_confirmed_empty_is_ready(self):
        result = nurec._coverage_counts([100_000, 200_000], {200_000}, {100_000}, True, 0)
        self.assertTrue(nurec._coverage_ready(True, True, result))
        self.assertEqual((result["observed"], result["empty"], result["unknown"], result["missing"]), (1, 1, 0, 0))

    def test_43_explicit_object_only_parity_case(self):
        result = nurec._coverage_counts([100_000, 200_000, 300_000], set(), {100_020, 200_000, 299_990}, False, 50_000)
        self.assertEqual((result["observed"], result["empty"], result["unknown"], result["missing"]), (3, 0, 0, 0))
        self.assertEqual(sum(result[key] for key in ("observed", "empty", "unknown", "missing")), result["required"])

    def test_44_every_coverage_result_preserves_state_invariant(self):
        for required, evidence, obstacles, complete, tolerance in (
            ([1, 2], set(), {1}, False, 0),
            ([1, 2], {2}, {1}, False, 0),
            ([1, 2], {2}, {1}, True, 0),
            ([1, 2], set(), set(), False, 0),
        ):
            result = nurec._coverage_counts(required, evidence, obstacles, complete, tolerance)
            self.assertEqual(sum(result[key] for key in ("observed", "empty", "unknown", "missing")), result["required"])


if __name__ == "__main__":
    unittest.main()
