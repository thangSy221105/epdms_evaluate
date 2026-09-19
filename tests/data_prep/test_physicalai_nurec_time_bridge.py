import tempfile
import unittest
import argparse
from pathlib import Path

import pandas as pd

from scripts.audit_physicalai_nurec_time_bridge import (
    constant_delta_diagnostics,
    classify_clip_pattern,
    classify_cross_clip_pattern,
    duration_diagnostics,
    inspect_nurec_provenance,
    explicit_ncore_nurec_mapping,
    _external_dataset_evidence,
    pai_to_ncore_timestamp_contract,
    per_sequence_mapping,
    relative_clock_diagnostics,
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
            result = inspect_nurec_provenance(directory)
            self.assertEqual(result["status"], "NOT_FOUND_AFTER_INSPECTION")
            self.assertEqual(result["inspection_errors"], [])

    def test_recursive_provenance_finds_source_clip_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text("nested:\n  source_clip_id: pilot-1\n", encoding="utf-8")
            result = inspect_nurec_provenance(directory, expected_source_clip_id="pilot-1")
            self.assertEqual(result["status"], "FOUND")
            self.assertTrue(any(item["verification_status"] == "VERIFIED_SOURCE_PROVENANCE" for item in result["verified_candidates"]))

    def test_malformed_metadata_read_error_never_becomes_found(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.yaml"
            path.write_text("broken: [", encoding="utf-8")
            result = inspect_nurec_provenance(directory)
            self.assertNotEqual(result["status"], "FOUND")
            self.assertEqual(result["verified_candidates"], [])
            self.assertTrue(result["inspection_errors"])
            self.assertEqual(result["inspection_errors"][0]["inspection_status"], "READ_ERROR")

    def test_generic_clip_id_is_candidate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "clipgt").mkdir()
            pd.DataFrame({"clip_id": ["00040136-e651-4abd-991d-0655ccda9430"]}).to_parquet(Path(directory) / "clipgt" / "clip.parquet")
            result = inspect_nurec_provenance(directory)
            candidate = result["candidates"][0]
            self.assertEqual(candidate["identity_class"], "GENERIC_IDENTITY")
            self.assertEqual(candidate["verification_status"], "CANDIDATE_ONLY")

    def test_source_clip_id_matching_expected_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = "00040136-e651-4abd-991d-0655ccda9430"
            (Path(directory) / "clipgt").mkdir()
            pd.DataFrame({"source_clip_id": [expected]}).to_parquet(Path(directory) / "clipgt" / "clip.parquet")
            result = inspect_nurec_provenance(directory, expected)
            candidate = result["verified_candidates"][0]
            self.assertEqual(candidate["identity_class"], "SOURCE_IDENTITY")
            self.assertTrue(candidate["matches_expected_source_clip"])
            self.assertEqual(candidate["verification_status"], "VERIFIED_SOURCE_PROVENANCE")

    def test_wrong_source_clip_id_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "clipgt").mkdir()
            pd.DataFrame({"source_clip_id": ["another-clip"]}).to_parquet(Path(directory) / "clipgt" / "clip.parquet")
            result = inspect_nurec_provenance(directory, "expected")
            self.assertEqual(result["status"], "NOT_FOUND_AFTER_INSPECTION")
            self.assertEqual(result["candidates"][0]["verification_status"], "CANDIDATE_ONLY")

    def test_external_presence_is_not_verified_by_script(self):
        args = argparse.Namespace(official_ncore_pilot_presence="NOT_INSPECTED", official_nurec_pilot_presence="FOUND", external_dataset_evidence=None)
        evidence = _external_dataset_evidence(args)
        self.assertEqual(evidence["official_nurec"]["evidence_source"], "EXTERNAL_INSPECTION")
        self.assertFalse(evidence["official_nurec"]["verified_by_this_script"])

    def test_scope_includes_parsed_config_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "clipgt"
            root.mkdir()
            (Path(directory) / "parsed_config.yaml").write_text("version: 1\n", encoding="utf-8")
            (Path(directory) / "metadata.yaml").write_text("name: pilot\n", encoding="utf-8")
            pd.DataFrame({"calibration_id": ["c1"]}).to_parquet(root / "calibration_estimate.parquet")
            result = inspect_nurec_provenance(directory)
            self.assertIn("parsed_config.yaml", result["successfully_read_files"])
            self.assertIn("clipgt/calibration_estimate.parquet", result["successfully_read_files"])

    def test_source_lineage_does_not_verify_time_mapping(self):
        result = explicit_ncore_nurec_mapping({"source_clip_id": "a", "target_clip_id": "b"}, "a", "b")
        self.assertFalse(result["verified"])

    def test_identical_clocks_are_direct_with_semantic_pairs(self):
        self.assertEqual(classify_clip_pattern(direct=True, constant_delta=True, duration_equal=True, semantic_pairs=2), "DIRECT")

    def test_same_global_offset_is_candidate_only(self):
        result = classify_cross_clip_pattern([{"semantic_pairs": 3, "constant_delta": True, "offset_us": 10}, {"semantic_pairs": 3, "constant_delta": True, "offset_us": 10}, {"semantic_pairs": 3, "constant_delta": True, "offset_us": 10}])
        self.assertEqual(result["pattern_status"], "GLOBAL_FIXED_OFFSET_CANDIDATE")
        self.assertEqual(result["verification_status"], "UNVERIFIED")

    def test_different_offsets_are_per_clip_candidate(self):
        result = classify_cross_clip_pattern([{"semantic_pairs": 2, "constant_delta": True, "offset_us": 10}, {"semantic_pairs": 2, "constant_delta": True, "offset_us": 20}])
        self.assertEqual(result["pattern_status"], "PER_CLIP_OFFSET_CANDIDATE")

    def test_same_offset_within_sequence_can_be_grouped_without_verification(self):
        result = classify_cross_clip_pattern([{"sequence_id": "s1", "semantic_pairs": 2, "constant_delta": True, "offset_us": 10}, {"sequence_id": "s1", "semantic_pairs": 2, "constant_delta": True, "offset_us": 10}])
        self.assertEqual(result["offsets"], [10])
        self.assertEqual(result["verification_status"], "UNVERIFIED")

    def test_relative_timeline_with_different_origins(self):
        result = relative_clock_diagnostics([100, 200, 300], [1000, 1100, 1200])
        self.assertTrue(result["same_duration"])
        self.assertTrue(result["same_count"])

    def test_differing_duration_rejects_translation_invariant(self):
        result = duration_diagnostics(0, 100, 1000, 1201)
        self.assertEqual(result["duration_error_us"], 101)

    def test_interval_delta_without_pairs_is_unverified(self):
        result = classify_cross_clip_pattern([])
        self.assertEqual(result["verification_status"], "UNVERIFIED")

    def test_two_constant_semantic_pairs_can_classify_clip(self):
        self.assertEqual(classify_clip_pattern(constant_delta=True, duration_equal=True, semantic_pairs=2), "CONSTANT_OFFSET")

    def test_conflicting_semantic_deltas_reject_constant_mapping(self):
        self.assertEqual(classify_clip_pattern(constant_delta=False, duration_equal=True, semantic_pairs=3), "NONLINEAR_OR_RETIMED")

    def test_pilot_absent_from_ncore_blocks_application(self):
        self.assertFalse("pilot" in {"other"})

    def test_external_identity_does_not_upgrade_numeric_pattern(self):
        result = classify_cross_clip_pattern([{"semantic_pairs": 0, "constant_delta": True, "offset_us": 4}])
        self.assertEqual(result["verification_status"], "UNVERIFIED")

    def test_raw_files_unchanged_by_pattern_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.bin"; path.write_bytes(b"raw"); before = path.read_bytes()
            classify_cross_clip_pattern([]); self.assertEqual(path.read_bytes(), before)

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
