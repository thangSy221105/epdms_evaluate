import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.data_prep import nurec


class TestNuRecConditionCoverage(unittest.TestCase):
    def _condition(self, clip="clip-a", mode="cross_scene", alpha=0.0, t0=100):
        return {"clip_id": clip, "mode": mode, "alpha": alpha, "t0_us": t0, "coordinate_frame": "ego", "reference_point": "rear"}

    def _run_audit(self, root, prediction_rows):
        clip = root / "clip-a"
        (clip / "clipgt").mkdir(parents=True)
        pred = root / "pred.jsonl"
        gt = root / "gt.jsonl"
        pred.write_text("\n".join(json.dumps(row) for row in prediction_rows) + "\n", encoding="utf-8")
        gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100, "future_frame": "ego"}) + "\n", encoding="utf-8")
        with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "OK", "row_count": 0, "min": 100, "max": 200, "unique_count": 0, "field": "key.timestamp_micros"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 100, "max": 200}), mock.patch.object(nurec, "_parquet_timestamp_values", return_value=set()), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
            return nurec.audit_dataset(root, pred, gt, root / "audit")

    def test_01_sixteen_conditions_same_clip_are_kept(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            rows = [self._condition(alpha=alpha, mode=mode) for mode in ("cross_scene", "noisy", "opposite_action", "no_reasoning") for alpha in (0, 0.5, 1, 2)]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result, errors, duplicates, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 16)
            self.assertEqual(duplicates, [])
            self.assertEqual(errors, [])

    def test_02_same_clip_different_mode_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text("\n".join(json.dumps(self._condition(mode=mode)) for mode in ("cross_scene", "noisy")) + "\n", encoding="utf-8")
            result, _, duplicates, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 2)
            self.assertFalse(duplicates)

    def test_03_same_clip_mode_different_alpha_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text("\n".join(json.dumps(self._condition(alpha=alpha)) for alpha in (0, 0.5)) + "\n", encoding="utf-8")
            result, _, duplicates, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 2)
            self.assertFalse(duplicates)

    def test_04_exact_condition_duplicate_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            row = self._condition(alpha=0.5)
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            result, errors, duplicates, _ = nurec._load_prediction_conditions(path)
            self.assertEqual(len(result["clip-a"]), 2)
            self.assertEqual(duplicates, ["clip-a|cross_scene|0.5"])
            self.assertEqual(errors[-1]["failure_type"], "PREDICTION_CONDITION_DUPLICATE")

    def test_05_missing_mode_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "alpha": 0}) + "\n", encoding="utf-8")
            result, errors, _, identity_errors = nurec._load_prediction_conditions(path)
            self.assertFalse(result)
            self.assertEqual(errors[0]["failure_type"], "PREDICTION_IDENTITY_INVALID")
            self.assertIn("MISSING_MODE", errors[0]["failure_reason"])
            self.assertIn("clip-a", identity_errors)

    def test_06_missing_alpha_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "mode": "noisy"}) + "\n", encoding="utf-8")
            _, errors, _, _ = nurec._load_prediction_conditions(path)
            self.assertIn("MISSING_ALPHA", errors[0]["failure_reason"])

    def test_07_nonfinite_alpha_is_identity_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text(json.dumps({"clip_id": "clip-a", "mode": "noisy", "alpha": "NaN"}) + "\n", encoding="utf-8")
            _, errors, _, _ = nurec._load_prediction_conditions(path)
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

    def test_19_obstacle_timestamps_without_evidence_are_not_observed(self):
        result = nurec._coverage_counts([1, 2], set(), {1, 2}, True, 0)
        self.assertEqual((result["observed"], result["missing"]), (0, 2))

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

    def test_25_timestamp_matching_uses_tolerance(self):
        result = nurec._coverage_counts([100_000], {100_040}, set(), True, 50)
        self.assertEqual(result["matched"], 1)

    def test_26_timestamp_outside_tolerance_is_missing(self):
        result = nurec._coverage_counts([100_000], {100_051}, set(), True, 50)
        self.assertEqual(result["missing"], 1)


if __name__ == "__main__":
    unittest.main()
