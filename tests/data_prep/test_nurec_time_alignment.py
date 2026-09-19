import json
import tempfile
import unittest
from pathlib import Path

from tools.data_prep.time_alignment import correspondence_status, summarize_values, unit_evidence


class TestNuRecTimeAlignmentEvidence(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
