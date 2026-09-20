import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cf_ttc_13clip_empty_attestation import classify_gap
from scripts.audit_cf_ttc_full300_minimal import (
    load_confirmed_empty_sidecar,
    run_audit,
)
from scripts.prepare_nurec_obstacles import build_scorer_query_grid


def _sequence(path: Path, timestamps=(1_000_000,)):
    value = {
        "tracks_data": {
            "tracks_id": ["track-1"],
            "tracks_poses": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in timestamps]],
            "tracks_timestamps_us": [list(timestamps)],
            "tracks_label_class": ["vehicle"],
            "tracks_flags": [[0 for _ in timestamps]],
        },
        "cuboidtracks_data": {"cuboids_dims": [[4.0, 2.0, 1.5]]},
    }
    path.write_text(json.dumps(value), encoding="utf-8")


class CFTTCAttestationTests(unittest.TestCase):
    def _item(self, timestamp=2_000_000):
        return {
            "clip_id": "clip-a",
            "query_timestamp_us": timestamp,
            "required_by_cf": True,
            "required_by_ttc": False,
            "nearest_object_timestamp_us": None,
            "object_delta_us": None,
            "prior_classifications": {"POSSIBLE_EMPTY_FRAME"},
        }

    def test_sensor_range_does_not_attest_label_empty(self):
        result = classify_gap(
            self._item(),
            {
                "object_timestamps": {2_100_000: 1},
                "label_ranges": [{"source_file": "sequence_tracks.json", "min_us": 1_900_000, "max_us": 2_300_000}],
                "explicit_empty_timestamps": {},
                "frame_ranges": [{"path": "camera.frame-range", "start_us": 1_900_000, "end_us": 2_300_000}],
                "sensor_sources": ["data_info.json"],
                "label_source_errors": [],
                "source_files": ["sequence_tracks.json", "data_info.json"],
            },
        )
        self.assertTrue(result["sensor_frame_present"])
        self.assertFalse(result["label_frame_attested"])
        self.assertEqual(result["classification"], "UNRESOLVED")

    def test_exact_zero_object_frame_is_confirmed_empty(self):
        result = classify_gap(
            self._item(),
            {
                "object_timestamps": {2_100_000: 1},
                "label_ranges": [{"source_file": "labels.json", "min_us": 1_000_000, "max_us": 3_000_000}],
                "explicit_empty_timestamps": {2_000_000: "labels.json:empty_frames[0]"},
                "frame_ranges": [],
                "sensor_sources": [],
                "label_source_errors": [],
                "source_files": ["labels.json"],
            },
        )
        self.assertEqual(result["classification"], "CONFIRMED_EMPTY")
        self.assertEqual(result["label_object_count"], 0)

    def test_nearest_empty_timestamp_is_not_exact_attestation(self):
        result = classify_gap(
            self._item(),
            {
                "object_timestamps": {2_100_000: 1},
                "label_ranges": [{"source_file": "labels.json", "min_us": 1_000_000, "max_us": 3_000_000}],
                "explicit_empty_timestamps": {2_000_001: "labels.json:empty_frames[0]"},
                "frame_ranges": [],
                "sensor_sources": [],
                "label_source_errors": [],
                "source_files": ["labels.json"],
            },
        )
        self.assertEqual(result["classification"], "UNRESOLVED")

    def test_object_and_empty_evidence_conflict_fails_closed(self):
        result = classify_gap(
            self._item(),
            {
                "object_timestamps": {2_000_000: 3},
                "label_ranges": [{"source_file": "labels.json", "min_us": 1_000_000, "max_us": 3_000_000}],
                "explicit_empty_timestamps": {2_000_000: "labels.json:empty_frames[0]"},
                "frame_ranges": [],
                "sensor_sources": [],
                "label_source_errors": [],
                "source_files": ["labels.json"],
            },
        )
        self.assertEqual(result["classification"], "UNRESOLVED")
        self.assertTrue(result["object_empty_evidence_conflict"])

    def test_outside_label_timeline_is_not_empty(self):
        result = classify_gap(
            self._item(),
            {
                "object_timestamps": {3_000_000: 1},
                "label_ranges": [{"source_file": "labels.json", "min_us": 3_000_000, "max_us": 4_000_000}],
                "explicit_empty_timestamps": {},
                "frame_ranges": [],
                "sensor_sources": [],
                "label_source_errors": [],
                "source_files": ["labels.json"],
            },
        )
        self.assertEqual(result["classification"], "OUTSIDE_LABEL_TIMELINE")

    def test_sidecar_rejects_unverified_rows_and_keeps_verified_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.jsonl"
            path.write_text(
                json.dumps({"clip_id": "bad", "confirmed_empty_timestamps_us": [1], "source": "x", "verified": False})
                + "\n"
                + json.dumps({
                    "clip_id": "good",
                    "confirmed_empty_timestamps_us": [2],
                    "source": "frame_manifest.json",
                    "attestation_type": "AUTHORITATIVE_FRAME_LEVEL_ZERO_OBJECT_LABEL_SET",
                    "verified": True,
                })
                + "\n",
                encoding="utf-8",
            )
            by_clip, sources, errors = load_confirmed_empty_sidecar(path)
            self.assertNotIn("bad", by_clip)
            self.assertEqual(by_clip["good"], {2})
            self.assertEqual(sources[("good", 2)], "frame_manifest.json")
            self.assertTrue(errors)

    def test_exact_empty_sidecar_can_close_both_grids_without_claiming_world_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = root / "clip-a"
            clip.mkdir()
            _sequence(clip / "sequence_tracks.json")
            cf, ttc, _ = build_scorer_query_grid(2_000_000, include_t0=True)
            manifest = [{
                "clip_id": "clip-a",
                "sequence_tracks_path": str(clip / "sequence_tracks.json"),
                "sequence_tracks_available": True,
                "time_mapping_verified": True,
                "nurec_t0_us": 2_000_000,
                "offset_us": 0,
                "stage1_input_ready": True,
            }]
            summary = run_audit(
                manifest,
                root / "out",
                expected_clip_count=1,
                confirmed_empty_by_clip={"clip-a": set(cf + ttc)},
                confirmed_empty_sources={("clip-a", timestamp): "frame_manifest.json" for timestamp in set(cf + ttc)},
            )
            self.assertEqual(summary["TOTAL_CF_REQUIRED_QUERY_COUNT"], 41)
            self.assertEqual(summary["TOTAL_TTC_REQUIRED_QUERY_COUNT"], 51)
            self.assertEqual(summary["TOTAL_CF_MISSING_COUNT"], 0)
            self.assertEqual(summary["TOTAL_TTC_MISSING_COUNT"], 0)
            self.assertTrue(summary["CF_DATA_READY_FULL_300"])
            self.assertTrue(summary["TTC_DATA_READY_FULL_300"])
            self.assertEqual(summary["PHYSICAL_WORLD_OBSTACLE_COMPLETENESS"], "NOT_CLAIMED")


if __name__ == "__main__":
    unittest.main()
