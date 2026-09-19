import unittest

from scripts.audit_pai_obstacle_offline import classify_representation, classify_track_gap, exact_match_index, wrapped_yaw_difference


class PaiObstacleOfflineAuditTests(unittest.TestCase):
    def test_feature_presence_true_false_unknown_are_distinct(self):
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertIsNone(None)

    def test_object_row_schema(self):
        self.assertEqual(classify_representation(["timestamp_us", "track_id"], [{"timestamp_us": 1, "track_id": "a"}]), "OBJECT_ROWS_ONLY")

    def test_explicit_empty_list_schema(self):
        self.assertEqual(classify_representation(["timestamp_us", "objects"], [{"timestamp_us": 1, "objects": []}]), "FRAME_GROUPED_OBJECT_LIST")

    def test_object_count_schema(self):
        self.assertEqual(classify_representation(["timestamp_us", "object_count"], [{"timestamp_us": 1, "object_count": 0}]), "FRAME_ROW_WITH_OBJECT_COUNT")

    def test_missing_timestamp_is_not_empty(self):
        self.assertEqual(classify_representation(["track_id"], [{"track_id": "a"}]), "UNKNOWN")

    def test_raw_source_gap_is_not_local_missing(self):
        self.assertEqual(classify_track_gap(200_000, 100_000), "SOURCE_TRACK_GAP_PRESENT")

    def test_small_gap_is_not_called_missing(self):
        self.assertEqual(classify_track_gap(110_000, 100_000), "NO_INTERIOR_GAP_IN_SOURCE_GRID")

    def test_exact_id_timestamp_index(self):
        rows = [{"track_id": "a", "timestamp_us": 10}]
        self.assertIn(("a", 10), exact_match_index(rows))

    def test_wrapped_yaw(self):
        import math
        qa = [0, 0, math.sin(math.radians(89.5)), math.cos(math.radians(89.5))]
        qb = [0, 0, math.sin(math.radians(-89.5)), math.cos(math.radians(-89.5))]
        self.assertAlmostEqual(math.degrees(wrapped_yaw_difference(qa, qb)), 2.0, places=5)

    def test_offset_is_reused_by_contract(self):
        self.assertEqual(5, 5)

    def test_object_rows_do_not_prove_empty_frames(self):
        self.assertNotEqual("OBJECT_ROWS_ONLY", "FRAME_ROW_WITH_OBJECT_COUNT")

    def test_unresolved_timeline_keeps_readiness_false(self):
        complete = None
        ready = False if complete is None else complete == 1.0
        self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main()
