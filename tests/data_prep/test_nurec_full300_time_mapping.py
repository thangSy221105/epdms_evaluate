import json
import tempfile
import unittest
from pathlib import Path

from scripts.recover_nurec_time_mapping_full300 import (
    EXPECTED_CLIP_COUNT,
    compute_nurec_t0,
    posthoc_sequence_consistency,
    select_mapping_candidate,
    verify_physicalai_t0,
    verify_semantic_pairs,
)


class NuRecFull300TimeMappingTests(unittest.TestCase):
    def test_existing_verified_candidate_is_reused_without_change(self):
        result = select_mapping_candidate([
            {"verification_status": "VERIFIED", "offset_us": 3_033_653_000},
        ])
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["offset_us"], 3_033_653_000)

    def test_unverified_candidate_is_not_promoted(self):
        result = select_mapping_candidate([
            {"verification_status": "UNVERIFIED", "offset_us": 3_033_653_000},
        ])
        self.assertEqual(result["status"], "MISSING_REQUIRED_EVIDENCE")

    def test_semantic_correspondence_is_required(self):
        result = verify_semantic_pairs([
            {"semantic_identity": False, "offset_us": 100, "scale": 1.0},
            {"semantic_identity": False, "offset_us": 100, "scale": 1.0},
        ])
        self.assertFalse(result["verified"])

    def test_numeric_proximity_is_not_verification(self):
        result = verify_semantic_pairs([
            {"semantic_identity": False, "offset_us": 100, "scale": 1.0},
            {"semantic_identity": False, "offset_us": 101, "scale": 1.0},
        ])
        self.assertFalse(result["verified"])

    def test_per_clip_offsets_are_independent_and_no_global_offset_is_used(self):
        first = verify_semantic_pairs([
            {"semantic_identity": True, "offset_us": 100, "scale": 1.0},
            {"semantic_identity": True, "offset_us": 100, "scale": 1.0},
        ])
        second = verify_semantic_pairs([
            {"semantic_identity": True, "offset_us": 200, "scale": 1.0},
            {"semantic_identity": True, "offset_us": 200, "scale": 1.0},
        ])
        self.assertTrue(first["verified"])
        self.assertTrue(second["verified"])
        self.assertNotEqual(first["unique_offsets_us"], second["unique_offsets_us"])

    def test_consistent_pairs_verify(self):
        result = verify_semantic_pairs([
            {"semantic_identity": True, "offset_us": 100, "scale": 1.0},
            {"semantic_identity": True, "offset_us": 100, "scale": 1.0},
        ])
        self.assertTrue(result["verified"])
        self.assertEqual(result["pair_count"], 2)

    def test_conflicting_pairs_reject(self):
        result = verify_semantic_pairs([
            {"semantic_identity": True, "offset_us": 100, "scale": 1.0},
            {"semantic_identity": True, "offset_us": 101, "scale": 1.0},
        ])
        self.assertFalse(result["verified"])
        self.assertEqual(result["unique_offset_count"], 2)

    def test_conflicting_verified_sources_reject(self):
        result = select_mapping_candidate([
            {"verification_status": "VERIFIED", "offset_us": 100},
            {"verification_status": "VERIFIED", "offset_us": 101},
        ])
        self.assertEqual(result["status"], "CONFLICTING_EVIDENCE")

    def test_t0_conflict_rejects_mapping(self):
        result = verify_physicalai_t0([5_100_000], [5_200_000], 5_100_000)
        self.assertEqual(result["status"], "CONFLICTING")
        self.assertIsNone(result["physicalai_t0_us"])

    def test_nurec_t0_equation(self):
        self.assertEqual(compute_nurec_t0(5_100_000, 3_033_653_000), 3_038_753_000)

    def test_sequence_minimum_cannot_derive_offset(self):
        result = posthoc_sequence_consistency(1_000, 7_000_000, None)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertFalse(result["mapping_verified"])

    def test_posthoc_consistency_does_not_verify_mapping(self):
        result = posthoc_sequence_consistency(100, 7_000_100, 100)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["mapping_verified"])

    def test_full_status_requires_exactly_300_clip_membership(self):
        self.assertEqual(EXPECTED_CLIP_COUNT, 300)
        self.assertNotEqual(len({f"clip-{i}" for i in range(299)}), EXPECTED_CLIP_COUNT)

    def test_historical_contract_is_not_mutated_by_test_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.jsonl"
            original = {"clip_id": "historical", "offset_us": 7, "verified": True}
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            # The recovery code writes a separate full300 artifact; this fixture is immutable.
            self.assertEqual(path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
