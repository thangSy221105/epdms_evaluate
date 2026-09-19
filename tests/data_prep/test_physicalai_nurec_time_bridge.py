import tempfile
import unittest
from pathlib import Path

from scripts.audit_physicalai_nurec_time_bridge import (
    constant_delta_diagnostics,
    inspect_nurec_provenance,
    explicit_ncore_nurec_mapping,
    pai_to_ncore_timestamp_contract,
    per_sequence_mapping,
    t0_query_contract,
)


class TestPhysicalAINuRecTimeBridge(unittest.TestCase):
    def test_constant_delta_without_semantic_identity_is_diagnostic_only(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1200},
        ], semantic_verified=False)
        self.assertEqual(result["median_delta"], 1000)
        self.assertEqual(result["verification_status"], "DIAGNOSTIC_ONLY")

    def test_constant_delta_with_semantic_identity_can_verify(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1200},
        ], semantic_verified=True)
        self.assertEqual(result["verification_status"], "VERIFIED")

    def test_inconsistent_delta_is_not_verified(self):
        result = constant_delta_diagnostics([
            {"physicalai_timestamp_us": 100, "nurec_timestamp_us": 1100},
            {"physicalai_timestamp_us": 200, "nurec_timestamp_us": 1201},
        ], semantic_verified=True)
        self.assertEqual(result["verification_status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["delta_unique_count"], 2)

    def test_raw_file_is_not_modified_by_forensic_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.txt"
            path.write_text("raw", encoding="utf-8")
            before = path.read_bytes()
            constant_delta_diagnostics([], semantic_verified=False)
            self.assertEqual(path.read_bytes(), before)

    def test_t0_is_interpolatable_without_exact_row(self):
        result = t0_query_contract([4_900_000, 5_099_189, 5_109_194, 5_300_000])
        self.assertTrue(result["query_contract_found"])
        self.assertFalse(result["exact_row"])
        self.assertTrue(result["interpolatable"])

    def test_t0_out_of_range_is_not_interpolatable(self):
        self.assertFalse(t0_query_contract([0, 1_000])["interpolatable"])

    def test_provenance_not_inspected_vs_not_found(self):
        self.assertEqual(inspect_nurec_provenance(None)["status"], "NOT_INSPECTED")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_info.json"
            path.write_text('{"pose-range": {"start-timestamp_us": 1}}', encoding="utf-8")
            self.assertEqual(inspect_nurec_provenance(directory)["status"], "NOT_FOUND_AFTER_INSPECTION")

    def test_recursive_provenance_finds_source_clip_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text("nested:\n  source_clip_id: pilot-1\n", encoding="utf-8")
            result = inspect_nurec_provenance(directory)
            self.assertEqual(result["status"], "FOUND")
            self.assertTrue(any(item["value"] == "pilot-1" for item in result["candidates"]))

    def test_pai_to_ncore_filters_negative_rows_without_retiming(self):
        result = pai_to_ncore_timestamp_contract([-3, 0, 100, 200])
        self.assertEqual(result["numeric_retiming"], "NONE_FOR_RETAINED_ROWS")
        self.assertEqual(result["negative_ego_rows"], "FILTERED")
        self.assertEqual(result["scale"], 1.0)
        self.assertEqual(result["offset_us"], 0)
        self.assertEqual(result["retained_count"], 3)

    def test_explicit_ncore_nurec_offset_is_accepted(self):
        result = explicit_ncore_nurec_mapping({"source_clip_id": "a", "target_clip_id": "b", "offset_us": 42}, "a", "b")
        self.assertTrue(result["verified"])
        self.assertEqual(result["offset_us"], 42)

    def test_same_numeric_delta_without_semantic_pair_stays_diagnostic(self):
        result = constant_delta_diagnostics([{"physicalai_timestamp_us": 10, "nurec_timestamp_us": 110}], False)
        self.assertEqual(result["verification_status"], "DIAGNOSTIC_ONLY")

    def test_per_sequence_mapping_isolated(self):
        result = per_sequence_mapping([{ "sequence_id": "s1", "offset_us": 1 }, { "sequence_id": "s2", "offset_us": 2 }], "s2")
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["mapping"]["offset_us"], 2)

    def test_wrong_source_clip_is_rejected(self):
        result = explicit_ncore_nurec_mapping({"source_clip_id": "wrong", "target_clip_id": "b", "offset_us": 42}, "a", "b")
        self.assertFalse(result["verified"])
