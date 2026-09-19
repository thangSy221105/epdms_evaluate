import unittest

from scripts.audit_nurec_observation_completeness import audit_tracks, frame_completeness, query_completeness


def make_row(track="a", ts=0, index=0, center=None, quaternion=None, dimensions=None, category="car"):
    return {"track_id": track, "timestamp_us": ts, "center": center or [0.0, 0.0, 0.0], "quaternion": quaternion or [0.0, 0.0, 0.0, 1.0], "dimensions": dimensions or [4.0, 2.0, 1.5], "category": category, "source_pose_index": index}


class ObservationCompletenessTests(unittest.TestCase):
    def test_regular_track(self):
        tracks, cadence = audit_tracks([make_row(ts=i * 100_000, index=i) for i in range(4)])
        self.assertEqual(tracks[0]["track_completeness_status"], "COMPLETE_ON_OBSERVED_GRID")
        self.assertEqual(cadence["global_expected_step_us"], 100_000)

    def test_interior_gap(self):
        tracks, _ = audit_tracks([make_row(ts=0, index=0), make_row(ts=100_000, index=1), make_row(ts=300_000, index=2)])
        self.assertEqual(tracks[0]["track_completeness_status"], "HAS_INTERIOR_GAPS")
        self.assertEqual(tracks[0]["interior_missing_slot_count"], 1)

    def test_duplicate_timestamp(self):
        tracks, _ = audit_tracks([make_row(ts=0, index=0), make_row(ts=0, index=1)])
        self.assertEqual(tracks[0]["duplicate_timestamp_count"], 1)

    def test_nonmonotonic_timestamp(self):
        tracks, _ = audit_tracks([make_row(ts=100, index=0), make_row(ts=0, index=1)])
        self.assertEqual(tracks[0]["track_completeness_status"], "INVALID_TIMESTAMPS")

    def test_single_observation(self):
        self.assertEqual(audit_tracks([make_row()])[0][0]["track_completeness_status"], "SINGLE_OBSERVATION_ONLY")

    def test_missing_geometry(self):
        self.assertEqual(audit_tracks([make_row(center=[float("nan"), 0, 0])])[0][0]["track_completeness_status"], "INVALID_TIMESTAMPS")

    def test_invalid_quaternion(self):
        self.assertEqual(audit_tracks([make_row(quaternion=[0, 0, 0, 0])])[0][0]["track_completeness_status"], "INVALID_TIMESTAMPS")

    def test_invalid_extent(self):
        self.assertEqual(audit_tracks([make_row(dimensions=[4, 0, 1])])[0][0]["track_completeness_status"], "INVALID_TIMESTAMPS")

    def test_irregular_cadence(self):
        status = audit_tracks([make_row(ts=0, index=0), make_row(ts=100_000, index=1), make_row(ts=150_000, index=2), make_row(ts=300_000, index=3)])[0][0]["track_completeness_status"]
        self.assertIn(status, {"IRREGULAR_CADENCE", "HAS_INTERIOR_GAPS"})

    def test_object_frame_requires_attestation(self):
        frame = frame_completeness([100], [make_row(ts=100)], 100_000)[0]
        self.assertEqual(frame["frame_annotation_status"], "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN")
        self.assertFalse(frame["completeness_verified"])

    def test_empty_frame_with_attestation_ready(self):
        frames = [{"timestamp_us": 100, "object_count": 0, "completeness_verified": True, "frame_annotation_status": "CONFIRMED_EMPTY"}]
        self.assertTrue(query_completeness([100], frames, 0)[0]["query_ready"])

    def test_missing_object_rows_unknown(self):
        self.assertEqual(frame_completeness([100], [], 100_000)[0]["frame_annotation_status"], "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN")

    def test_sensor_timestamp_is_not_annotation_complete(self):
        self.assertFalse(frame_completeness([100], [], 100_000)[0]["completeness_verified"])

    def test_interval_only_is_not_complete(self):
        self.assertFalse(frame_completeness([100], [make_row(ts=100)], 100_000)[0]["completeness_verified"])

    def test_object_count_preserved_but_unknown(self):
        frame = frame_completeness([100], [make_row(ts=100), make_row(track="b", ts=100)], 0)[0]
        self.assertEqual(frame["object_count"], 2)
        self.assertFalse(frame["completeness_verified"])

    def test_complete_objects_query_ready(self):
        frames = [{"timestamp_us": 100, "object_count": 1, "completeness_verified": True, "frame_annotation_status": "COMPLETE_OBJECTS_PRESENT"}]
        self.assertTrue(query_completeness([100], frames, 0)[0]["query_ready"])

    def test_one_unknown_query_blocks_cf(self):
        frames = [{"timestamp_us": 100, "object_count": 1, "completeness_verified": True, "frame_annotation_status": "COMPLETE_OBJECTS_PRESENT"}, {"timestamp_us": 200, "object_count": 1, "completeness_verified": False, "frame_annotation_status": "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN"}]
        self.assertFalse(all(r["query_ready"] for r in query_completeness([100, 200], frames, 0)))

    def test_out_of_range_query_not_ready(self):
        frames = [{"timestamp_us": 100, "object_count": 1, "completeness_verified": True, "frame_annotation_status": "COMPLETE_OBJECTS_PRESENT"}]
        self.assertFalse(query_completeness([200], frames, 10)[0]["query_ready"])

    def test_ttc_unknown_not_ready(self):
        frames = [{"timestamp_us": 100, "object_count": 1, "completeness_verified": False, "frame_annotation_status": "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN"}]
        self.assertFalse(query_completeness([100], frames, 0)[0]["query_ready"])

    def test_object_evidence_one_does_not_equal_complete(self):
        frames = frame_completeness([100, 200], [make_row(ts=100), make_row(ts=200)], 0)
        queries = query_completeness([100, 200], frames, 0)
        self.assertEqual(sum(r["object_count"] > 0 for r in queries) / 2, 1.0)
        self.assertFalse(any(r["query_ready"] for r in queries))

    def test_no_extrapolation_before_or_after_track(self):
        tracks, _ = audit_tracks([make_row(ts=500_000, index=0), make_row(ts=600_000, index=1)])
        self.assertEqual(tracks[0]["interior_missing_slot_count"], 0)


if __name__ == "__main__":
    unittest.main()
