import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.recover_nurec_time_evidence_exhaustive import (
    FINAL_STATUSES,
    _clip_metadata_inventory,
    _candidate_from_row,
    _posthoc_support,
    collect_timestamp_fields,
    inspect_source_files,
)


class ExhaustiveNuRecTimeEvidenceTests(unittest.TestCase):
    def _paths(self, root):
        return Path(root) / "historical.jsonl", Path(root) / "historical.csv"

    def test_explicit_source_is_inspected_without_filename_regex(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene_manifest.yaml"
            path.write_text('{"clip_id":"a","t0_us":5100000}\n', encoding="utf-8")
            contract, match = self._paths(directory)
            inventory, candidates = inspect_source_files([path], {"a"}, contract, match, set())
            self.assertEqual(inventory[0]["read_status"], "READ")
            self.assertTrue(candidates)
            self.assertEqual(candidates[0]["verification_status"], "UNVERIFIED")

    def test_nested_timestamp_fields_are_inventoried(self):
        summaries = collect_timestamp_fields({"pose": {"T_rig_world_timestamps_us": [10, 20, 30]}})
        self.assertEqual(len(summaries), 1)
        accumulator = next(iter(summaries.values()))
        self.assertEqual(accumulator.count, 3)
        self.assertEqual(accumulator.minimum, 10)
        self.assertEqual(accumulator.maximum, 30)

    def test_numeric_only_evidence_cannot_verify(self):
        candidate = _candidate_from_row(
            Path("unrelated.json"),
            {"clip_id": "a", "physicalai_t0_us": 10, "nurec_t0_us": 110, "offset_us": 100, "verified": True},
            1,
            {"a"},
            Path("historical.jsonl"),
            Path("historical.csv"),
        )
        self.assertEqual(candidate["verification_status"], "UNVERIFIED")
        self.assertEqual(candidate["verification_level"], "LEVEL_0_NUMERIC_ONLY")

    def test_complete_explicit_origin_metadata_can_verify(self):
        candidate = _candidate_from_row(
            Path("arbitrary_name.json"),
            {
                "clip_id": "a", "physicalai_t0_us": 10, "nurec_t0_us": 110, "offset_us": 100,
                "scale": 1.0, "verified": True, "verification_method": "producer_manifest",
                "source_clock_domain": "explicit PAI to NuRec origin",
            },
            1,
            {"a"},
            Path("historical.jsonl"),
            Path("historical.csv"),
        )
        self.assertEqual(candidate["verification_status"], "VERIFIED")
        self.assertEqual(candidate["verification_level"], "LEVEL_2_EXPLICIT_ORIGIN_METADATA")

    def test_two_semantic_pairs_verify_from_nonhistorical_file(self):
        candidate = _candidate_from_row(
            Path("producer_manifest.json"),
            {
                "clip_id": "a", "offset_us": 100, "scale": 1.0, "verified": True,
                "semantic_identity_source": "shared frame IDs",
                "semantic_pairs": [
                    {"semantic_identity": True, "offset_us": 100},
                    {"semantic_identity": True, "offset_us": 100},
                ],
            },
            1,
            {"a"},
            Path("historical.jsonl"),
            Path("historical.csv"),
        )
        self.assertEqual(candidate["verification_status"], "VERIFIED")
        self.assertEqual(candidate["verification_level"], "LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET")

    def test_one_semantic_pair_does_not_verify(self):
        candidate = _candidate_from_row(
            Path("producer_manifest.json"),
            {"clip_id": "a", "offset_us": 100, "verified": True, "semantic_pairs": [{"semantic_identity": True, "offset_us": 100}]},
            1,
            {"a"},
            Path("historical.jsonl"),
            Path("historical.csv"),
        )
        self.assertEqual(candidate["verification_status"], "UNVERIFIED")

    def test_different_semantic_pair_offsets_reject(self):
        candidate = _candidate_from_row(
            Path("producer_manifest.json"),
            {
                "clip_id": "a", "offset_us": 100, "verified": True,
                "semantic_pairs": [
                    {"semantic_identity": True, "offset_us": 100},
                    {"semantic_identity": True, "offset_us": 101},
                ],
            },
            1,
            {"a"},
            Path("historical.jsonl"),
            Path("historical.csv"),
        )
        self.assertEqual(candidate["verification_status"], "UNVERIFIED")

    def test_filename_does_not_determine_verification(self):
        row = {"clip_id": "a", "offset_us": 100, "verified": True}
        first = _candidate_from_row(Path("time_mapping.csv"), row, 1, {"a"}, Path("h.jsonl"), Path("m.csv"))
        second = _candidate_from_row(Path("notes.csv"), row, 1, {"a"}, Path("h.jsonl"), Path("m.csv"))
        self.assertEqual(first["verification_status"], second["verification_status"])

    def test_historical_mapping_remains_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            contract, match = self._paths(directory)
            contract.write_text("{}\n", encoding="utf-8")
            match.write_text("clip_id,offset_start_us,offset_equal\na,100,True\n", encoding="utf-8")
            candidate = _candidate_from_row(match, {"clip_id": "a", "offset_start_us": "100", "offset_equal": "True"}, 2, {"a"}, contract, match)
            self.assertEqual(candidate["verification_level"], "HISTORICAL_ACCEPTED")

    def test_rig_or_sequence_minimum_cannot_derive_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence_tracks.json"
            path.write_text(json.dumps({"tracks_timestamps_us": [1000, 2000, 3000]}), encoding="utf-8")
            result = _posthoc_support(path, None)
            self.assertFalse(result["mapping_verified"])
            self.assertEqual(result["status"], "UNAVAILABLE")

    def test_posthoc_support_uses_five_seconds_and_cannot_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence_tracks.json"
            path.write_text(json.dumps({"tracks_timestamps_us": [100, 5_000_100]}), encoding="utf-8")
            result = _posthoc_support(path, 100)
            self.assertEqual(result["query_max_us"], 5_000_100)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["mapping_verified"])

    def test_final_status_set_is_closed(self):
        self.assertEqual(len(FINAL_STATUSES), 6)
        self.assertIn("MISSING_REQUIRED_EVIDENCE", FINAL_STATUSES)

    def test_canonical_style_rows_require_verified_flag(self):
        row = {"clip_id": "a", "verified": False, "offset_us": 100}
        candidate = _candidate_from_row(Path("manifest.json"), row, 1, {"a"}, Path("h.jsonl"), Path("m.csv"))
        self.assertFalse(candidate["verification_flag"])

    def test_lightweight_inventory_has_one_row_per_manifest_clip(self):
        manifest = {
            "a": {"clip_id": "a", "nurec_clip_dir": "C:/does/not/exist"},
            "b": {"clip_id": "b", "nurec_clip_dir": "C:/does/not/exist"},
        }
        rows = _clip_metadata_inventory(manifest, {"a": {"record_available": True}}, [])
        self.assertEqual([row["clip_id"] for row in rows], ["a", "b"])
        self.assertTrue(rows[0]["context_full_record_available"])
        self.assertFalse(rows[1]["context_full_record_available"])


if __name__ == "__main__":
    unittest.main()
