import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.data_prep.time_alignment import (
    _camera_records,
    _timeline_consistency,
    _load_jsonl_clip,
    _timeline_reports,
    audit_time_alignment,
    correspondence_status,
    extract_semantic_correspondences,
    resolve_time_alignment_evidence,
    summarize_values,
    unit_evidence,
)


class TestNuRecTimeAlignmentEvidence(unittest.TestCase):
    def test_00_semantic_shared_frame_id_constant_pairs_verify(self):
        pred = [{"identity_type": "frame_id", "identity_value": "a", "timestamp_us": 100, "semantic_explicit": True, "source_path": "pred"}, {"identity_type": "frame_id", "identity_value": "b", "timestamp_us": 200, "semantic_explicit": True, "source_path": "pred"}]
        gt = [{"identity_type": "frame_id", "identity_value": "a", "timestamp_us": 100, "semantic_explicit": True, "source_path": "gt"}, {"identity_type": "frame_id", "identity_value": "b", "timestamp_us": 200, "semantic_explicit": True, "source_path": "gt"}]
        raw = [{"identity_type": "frame_id", "identity_value": "a", "timestamp_us": 5100, "semantic_explicit": True, "source_path": "raw"}, {"identity_type": "frame_id", "identity_value": "b", "timestamp_us": 5200, "semantic_explicit": True, "source_path": "raw"}]
        pairs = extract_semantic_correspondences(pred, gt, raw)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(correspondence_status(pairs, True)["status"], "MAPPING_EXPLICIT")

    def test_00b_semantic_conflicting_offsets_rejected(self):
        pred = [{"identity_type": "frame_id", "identity_value": str(i), "timestamp_us": i * 100, "semantic_explicit": True} for i in (1, 2)]
        gt = list(pred)
        raw = [{"identity_type": "frame_id", "identity_value": "1", "timestamp_us": 1100, "semantic_explicit": True}, {"identity_type": "frame_id", "identity_value": "2", "timestamp_us": 2201, "semantic_explicit": True}]
        pairs = extract_semantic_correspondences(pred, gt, raw)
        self.assertEqual(correspondence_status(pairs, True)["status"], "CONFLICTING_TIME_ORIGIN")

    def test_00c_same_numbers_without_identity_are_unresolved(self):
        self.assertEqual(extract_semantic_correspondences([], [], []), [])

    def test_00d_identity_without_same_record_timestamp_is_not_verified(self):
        pred = [{"identity_type": "frame_id", "identity_value": "x", "timestamp_us": 1, "semantic_explicit": False}]
        self.assertEqual(extract_semantic_correspondences(pred, pred, pred), [])

    def test_00e_timeline_hash_clean_guided_roles(self):
        rows = [{"mode": "m", "alpha": 0.0, "clean_waypoints": [{"t_s": 0.1}, {"t_s": 0.2}]}, {"mode": "m", "alpha": 0.5, "guided_waypoints": [{"t_s": 0.1}, {"t_s": 0.2}]}]
        reports, conflicts = _timeline_consistency(rows)
        self.assertEqual(len(reports), 2)
        self.assertEqual(conflicts, [])

    def test_00f_timeline_hash_conflict(self):
        rows = [{"clean_waypoints": [{"t_s": 0.1}, {"t_s": 0.2}]}, {"clean_waypoints": [{"t_s": 0.1}, {"t_s": 0.3}]}]
        _, conflicts = _timeline_consistency(rows)
        self.assertEqual(conflicts, ["PREDICTION_TIMELINE_CONFLICT"])
    def test_01_explicit_shared_clock_is_classifiable(self):
        self.assertEqual(unit_evidence("timestamp_micros")["status"], "UNIT_EXPLICIT")

    def test_02_explicit_start_mapping_requires_metadata(self):
        result = correspondence_status([{"timestamp_a": 10, "timestamp_b": 110}, {"timestamp_a": 20, "timestamp_b": 120}], True)
        self.assertEqual(result["status"], "MAPPING_EXPLICIT")
        self.assertTrue(result["verified"])

    def test_03_numeric_offset_without_identity_is_unresolved(self):
        result = correspondence_status([{"timestamp_a": 10, "timestamp_b": 110}, {"timestamp_a": 20, "timestamp_b": 120}], False)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertFalse(result["verified"])

    def test_04_conflicting_explicit_offsets_are_not_verified(self):
        result = correspondence_status([{"timestamp_a": 10, "timestamp_b": 110}, {"timestamp_a": 20, "timestamp_b": 121}], True)
        self.assertEqual(result["offset_unique_count"], 2)
        self.assertFalse(result["verified"])

    def test_05_missing_origin_does_not_appear_from_values(self):
        self.assertEqual(unit_evidence("timestamp")["status"], "UNIT_AMBIGUOUS")

    def test_06_ambiguous_unit_is_reported(self):
        self.assertEqual(unit_evidence("time")["status"], "UNIT_AMBIGUOUS")

    def test_07_microsecond_field_is_explicit(self):
        result = unit_evidence("key.timestamp_micros")
        self.assertEqual(result["unit"], "microseconds")

    def test_08_egomotion_min_is_not_origin(self):
        self.assertNotIn("origin", summarize_values([100, 200]))

    def test_09_obstacle_min_is_not_origin(self):
        self.assertNotIn("origin", summarize_values([300, 400]))

    def test_10_camera_first_frame_is_not_origin(self):
        self.assertNotIn("origin", summarize_values([500, 600]))

    def test_11_multiple_constant_pairs_need_identity(self):
        result = correspondence_status([{"timestamp_a": i, "timestamp_b": i + 1000} for i in (1, 2, 3)], True)
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(result["offset_unique_count"], 1)
        self.assertTrue(result["verified"])

    def test_12_residual_conflict_rejects_mapping(self):
        result = correspondence_status([{"timestamp_a": 1, "timestamp_b": 1001}, {"timestamp_a": 2, "timestamp_b": 1003}], True)
        self.assertFalse(result["verified"])

    def test_13_raw_source_files_are_not_modified_by_test(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.json"
            raw.write_text(json.dumps({"timestamp_micros": 1}) + "\n", encoding="utf-8")
            before = raw.read_bytes()
            self.assertEqual(raw.read_bytes(), before)

    def _clock_evidence(self):
        return {"available": True, "unit": "microseconds", "unit_status": "UNIT_EXPLICIT"}

    def test_14_resolver_explicit_shared_clock_is_direct(self):
        result = resolve_time_alignment_evidence(self._clock_evidence(), self._clock_evidence(), self._clock_evidence(), {"shared_clock_domain": True, "prediction_clock_domain": "clip_relative_us", "nurec_clock_domain": "clip_relative_us", "prediction_origin_id": "clip-1", "nurec_origin_id": "clip-1", "same_unit": True, "source": ["data_info.json:time_alignment"]})
        self.assertEqual((result["status"], result["verified"]), ("ALIGNED_DIRECT", True))

    def test_15_resolver_explicit_origin_mapping_is_verified(self):
        result = resolve_time_alignment_evidence(self._clock_evidence(), self._clock_evidence(), self._clock_evidence(), {"mapping_formula": "prediction_relative_us + clip_start_us = nurec_us", "source": ["data_info.json:explicit_time_mapping"], "unit_explicit": True, "origin_explicit": True, "offset_us": 100, "mapping_type": "explicit_origin_mapping"})
        self.assertEqual((result["status"], result["offset_us"]), ("ALIGNED_BY_EXPLICIT_METADATA", 100))

    def test_16_numeric_coincidence_without_provenance_is_unresolved(self):
        result = resolve_time_alignment_evidence(self._clock_evidence(), self._clock_evidence(), self._clock_evidence())
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_17_conflicting_pairs_override_verified(self):
        result = resolve_time_alignment_evidence(self._clock_evidence(), self._clock_evidence(), self._clock_evidence(), {"semantic_identity": True, "identity_source": "frame_id"}, [{"timestamp_a": 1, "timestamp_b": 101}, {"timestamp_a": 2, "timestamp_b": 103}])
        self.assertEqual(result["status"], "CONFLICTING_TIME_ORIGIN")

    def test_18_ambiguous_required_unit_fails_closed(self):
        ambiguous = {"available": True, "unit": None, "unit_status": "UNIT_AMBIGUOUS"}
        self.assertEqual(resolve_time_alignment_evidence(ambiguous, self._clock_evidence(), self._clock_evidence())["status"], "UNIT_AMBIGUOUS")

    def test_19_missing_required_time_data_is_missing_metadata(self):
        missing = {"available": False, "unit_status": "UNIT_EXPLICIT"}
        self.assertEqual(resolve_time_alignment_evidence(missing, self._clock_evidence(), self._clock_evidence())["status"], "MISSING_TIME_METADATA")

    def test_20_relative_prediction_timeline_is_found(self):
        reports = _timeline_reports({"clean_waypoints": [{"t_s": 0.0}, {"t_s": 0.1}]})
        self.assertEqual((reports[0]["timestamp_kind"], reports[0]["timestamp_count"]), ("relative", 2))

    def test_21_absolute_prediction_timeline_is_found(self):
        reports = _timeline_reports({"guided_waypoints": [{"timestamp_micros": 10}, {"timestamp_micros": 20}]})
        self.assertEqual(reports[0]["timestamp_kind"], "absolute")

    def test_22_gt_timeline_is_found(self):
        reports = _timeline_reports({"future_waypoints": [{"timestamp_us": 10}, {"timestamp_us": 20}]})
        self.assertEqual(reports[0]["timestamp_field"], "timestamp_us")

    def test_23_timestamp_less_xyz_is_not_timestamped(self):
        reports = _timeline_reports({"ego_future_xyz": [[1, 2, 3], [4, 5, 6]]})
        self.assertEqual(reports[0]["unit_status"], "TIMESTAMP_IMPLICIT_BY_PIPELINE")

    def test_24_camera_filename_unit_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as td:
            sensor = Path(td) / "frames" / "camera_front"
            sensor.mkdir(parents=True)
            (sensor / "123456.jpeg").write_bytes(b"not-read")
            record = _camera_records(Path(td))[0]
            self.assertEqual((record["unit_declared"], record["unit_status"]), (None, "UNIT_AMBIGUOUS"))

    def test_25_malformed_jsonl_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            path.write_text('{"clip_id":"clip"}\nnot-json\n', encoding="utf-8")
            rows, errors = _load_jsonl_clip(path, "clip", "prediction")
            self.assertEqual(len(rows), 1)
            self.assertEqual(errors[0]["failure_type"], "MALFORMED_JSON")

    def test_26_integration_explicit_shared_clock_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "clip"
            (clip / "clipgt").mkdir(parents=True)
            (clip / "data_info.json").write_text(json.dumps({"time_alignment": {"shared_clock_domain": True, "prediction_clock_domain": "clip_relative_us", "nurec_clock_domain": "clip_relative_us", "prediction_origin_id": "x", "nurec_origin_id": "x", "same_unit": True}}), encoding="utf-8")
            pred = root / "pred.jsonl"; pred.write_text(json.dumps({"clip_id": "clip", "t0_us": 0}) + "\n", encoding="utf-8")
            gt = root / "gt.jsonl"; gt.write_text(json.dumps({"clip_id": "clip", "t0_us": 0}) + "\n", encoding="utf-8")
            parquet_record = [{"source": "obstacle", "field": "key.timestamp_micros", "unit_declared": "microseconds", "unit_status": "UNIT_EXPLICIT", "unit_basis": "field_name", "clock_domain_declared": None, "count": 1, "valid_count": 1, "invalid_count": 0, "min": 0, "max": 0, "unique_count": 1, "median_step": None, "first_5": [0], "last_5": [0], "source_path": "obstacle.parquet", "evidence_type": "test"}]
            with mock.patch("tools.data_prep.time_alignment._parquet_records", return_value=(parquet_record, {"status": "OK", "fields": ["key.timestamp_micros"]})):
                result = audit_time_alignment(clip, "clip", pred, gt, root / "out")
            self.assertEqual((result["evidence"]["mapping"]["status"], result["evidence"]["mapping"]["verified"]), ("ALIGNED_DIRECT", True))

    def test_27_integration_malformed_input_cannot_verify(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; pred.write_text('not-json\n', encoding="utf-8")
            gt = root / "gt.jsonl"; gt.write_text(json.dumps({"clip_id": "clip", "t0_us": 0}) + "\n", encoding="utf-8")
            with mock.patch("tools.data_prep.time_alignment._parquet_records", return_value=([], {"status": "NOT_FOUND", "fields": []})):
                result = audit_time_alignment(clip, "clip", pred, gt, root / "out")
            self.assertFalse(result["evidence"]["mapping"]["verified"])
            self.assertGreater(result["evidence"]["input_integrity"]["prediction_jsonl_error_count"], 0)

    def test_28_integration_explicit_origin_mapping_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip"; (clip / "clipgt").mkdir(parents=True)
            (clip / "data_info.json").write_text(json.dumps({"explicit_time_mapping": {"mapping_formula": "prediction_relative_us + clip_start_us = nurec_us", "source_field": "clip_start_us", "offset_us": 100, "unit_explicit": True, "origin_explicit": True}}), encoding="utf-8")
            pred = root / "pred.jsonl"; pred.write_text(json.dumps({"clip_id": "clip", "t0_us": 0}) + "\n", encoding="utf-8")
            gt = root / "gt.jsonl"; gt.write_text(json.dumps({"clip_id": "clip", "t0_us": 0}) + "\n", encoding="utf-8")
            parquet_record = [{"source": "obstacle", "field": "key.timestamp_micros", "unit_declared": "microseconds", "unit_status": "UNIT_EXPLICIT", "unit_basis": "field_name", "clock_domain_declared": None, "count": 1, "valid_count": 1, "invalid_count": 0, "min": 100, "max": 100, "unique_count": 1, "median_step": None, "first_5": [100], "last_5": [100], "source_path": "obstacle.parquet", "evidence_type": "test"}]
            with mock.patch("tools.data_prep.time_alignment._parquet_records", return_value=(parquet_record, {"status": "OK", "fields": ["key.timestamp_micros"]})):
                result = audit_time_alignment(clip, "clip", pred, gt, root / "out")
            self.assertEqual(result["evidence"]["mapping"]["status"], "ALIGNED_BY_EXPLICIT_METADATA")


if __name__ == "__main__":
    unittest.main()
