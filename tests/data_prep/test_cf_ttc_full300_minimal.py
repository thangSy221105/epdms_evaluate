import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cf_ttc_full300_minimal import (
    EXPECTED_CF_QUERIES_PER_CLIP,
    EXPECTED_TTC_QUERIES_PER_CLIP,
    build_full300_manifest,
    load_verified_time_mapping,
    run_audit,
)


def _sequence(path: Path, timestamps=(0, 100_000, 200_000)):
    value = {
        "tracks_data": {
            "tracks_id": ["track-1"],
            "tracks_poses": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in timestamps]],
            "tracks_timestamps_us": [list(timestamps)],
            "tracks_label_class": ["vehicle"],
            "tracks_flags": [[0]],
        },
        "cuboidtracks_data": {"cuboids_dims": [[4.0, 2.0, 1.5]]},
    }
    path.write_text(json.dumps(value), encoding="utf-8")


class Full300MinimalReadinessTests(unittest.TestCase):
    def test_mapping_loader_rejects_unverified_and_keeps_no_global_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.jsonl"
            path.write_text(json.dumps({"clip_id": "a", "physicalai_t0_us": 5_100_000, "nurec_t0_us": 9, "offset_us": 3, "verified": False, "source": "x"}) + "\n", encoding="utf-8")
            mapping, errors = load_verified_time_mapping(path)
            self.assertFalse(mapping["a"]["mapping_valid"])
            self.assertTrue(errors)

    def test_manifest_keeps_missing_sequence_and_mapping_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").mkdir()
            _sequence(root / "a" / "sequence_tracks.json")
            manifest, inventory, mapping_rows = build_full300_manifest(
                ["a", "b"],
                {"a": {"t0_values": {5_100_000}, "record_count": 1}, "b": {"t0_values": {5_100_000}, "record_count": 1}},
                {"a": {"t0_values": {5_100_000}, "record_count": 1}, "b": {"t0_values": {5_100_000}, "record_count": 1}},
                root,
                {"a": {"physicalai_t0_us": 5_100_000, "nurec_t0_us": 1_000_000, "offset_us": 0, "verified": True, "source": "existing", "mapping_valid": True, "mapping_status": "VERIFIED_REUSED_ARTIFACT"}},
            )
            self.assertTrue(manifest[0]["stage1_input_ready"])
            self.assertIn("SEQUENCE_TRACKS_MISSING", manifest[1]["input_blockers"])
            self.assertIn("TIME_MAPPING_MISSING", manifest[1]["input_blockers"])
            self.assertEqual(len(inventory), 2)
            self.assertEqual(len(mapping_rows), 2)

    def test_audit_uses_production_grid_and_missing_is_not_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = root / "a"
            clip.mkdir()
            _sequence(clip / "sequence_tracks.json", timestamps=(1_000_000,))
            manifest = [{
                "clip_id": "a", "sequence_tracks_path": str(clip / "sequence_tracks.json"),
                "sequence_tracks_available": True, "time_mapping_verified": True,
                "nurec_t0_us": 2_000_000, "offset_us": 0, "stage1_input_ready": True,
            }]
            summary = run_audit(manifest, root / "out", expected_clip_count=1)
            self.assertEqual(summary["ACTUAL_CF_QUERY_COUNT"], EXPECTED_CF_QUERIES_PER_CLIP)
            self.assertEqual(summary["ACTUAL_TTC_QUERY_COUNT"], EXPECTED_TTC_QUERIES_PER_CLIP)
            self.assertGreater(summary["TOTAL_CF_MISSING_COUNT"], 0)
            self.assertEqual(summary["TOTAL_CF_CONFIRMED_EMPTY_COUNT"], 0)
            self.assertFalse(summary["CF_DATA_READY_FULL_300"])

    def test_full300_readiness_requires_exact_clip_count_and_zero_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = root / "a"
            clip.mkdir()
            _sequence(clip / "sequence_tracks.json", timestamps=[2_000_000 + i * 100_000 for i in range(41)])
            manifest = [{
                "clip_id": "a", "sequence_tracks_path": str(clip / "sequence_tracks.json"),
                "sequence_tracks_available": True, "time_mapping_verified": True,
                "nurec_t0_us": 2_000_000, "offset_us": 0, "stage1_input_ready": True,
            }]
            summary = run_audit(manifest, root / "out", expected_clip_count=2)
            self.assertFalse(summary["CF_DATA_READY_FULL_300"])
            self.assertFalse(summary["TTC_DATA_READY_FULL_300"])

    def test_five_clip_pilot_regression_totals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = []
            timestamps = [1_000_000 + i * 100_000 for i in range(51)]
            for index in range(5):
                clip = root / f"pilot-{index}"
                clip.mkdir()
                _sequence(clip / "sequence_tracks.json", timestamps=timestamps)
                manifest.append({
                    "clip_id": f"pilot-{index}",
                    "sequence_tracks_path": str(clip / "sequence_tracks.json"),
                    "sequence_tracks_available": True,
                    "time_mapping_verified": True,
                    "nurec_t0_us": 1_000_000,
                    "offset_us": 0,
                    "stage1_input_ready": True,
                })
            summary = run_audit(manifest, root / "out", expected_clip_count=5)
            self.assertEqual(summary["TOTAL_CF_REQUIRED_QUERY_COUNT"], 205)
            self.assertEqual(summary["TOTAL_CF_OBJECT_PRESENT_COUNT"], 205)
            self.assertEqual(summary["TOTAL_CF_MISSING_COUNT"], 0)
            self.assertEqual(summary["TOTAL_TTC_REQUIRED_QUERY_COUNT"], 255)
            self.assertEqual(summary["TOTAL_TTC_OBJECT_PRESENT_COUNT"], 255)
            self.assertEqual(summary["TOTAL_TTC_MISSING_COUNT"], 0)

    def test_fallback_plan_contains_only_affected_clip_and_no_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = [{
                "clip_id": "missing", "sequence_tracks_path": str(root / "missing" / "sequence_tracks.json"),
                "sequence_tracks_available": False, "time_mapping_verified": False,
                "nurec_t0_us": None, "offset_us": None, "stage1_input_ready": False,
            }]
            summary = run_audit(manifest, root / "out", expected_clip_count=1)
            plan = (root / "out" / "minimal_fallback_data_plan.csv").read_text(encoding="utf-8")
            self.assertIn("missing", plan)
            self.assertIn("False", plan)
            self.assertEqual(summary["MINIMAL_FALLBACK_CLIP_COUNT"], 1)
            self.assertEqual(summary["FULL_CLIP_DOWNLOAD_REQUIRED_COUNT"], 0)


if __name__ == "__main__":
    unittest.main()
