import math
import unittest

import numpy as np

from scripts.prepare_nurec_obstacles import (
    CF_TOLERANCE_US,
    TTC_TOLERANCE_US,
    build_obstacle_context,
    build_scorer_query_grid,
    classify_scorer_queries,
    normalize_sequence_tracks,
    strict_coverage_summary,
)
from tools.epdms.observation_contract import build_ttc_projection_timestamps


def track_row(track_id="a", timestamp=100, pose_index=0, dims=None):
    return {
        "track_id": track_id,
        "timestamp_us": timestamp,
        "center": [1.0, 2.0, 0.5],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
        "dimensions": dims or [4.0, 2.0, 1.5],
        "category": "car",
        "flag": "NONE",
        "source_index": 0,
        "source_pose_index": pose_index,
    }


class ObstacleNormalizationReadinessTests(unittest.TestCase):
    def test_sequence_tracks_normalize_to_verified_ncore_local_world(self):
        rows = normalize_sequence_tracks("clip", [track_row()])
        self.assertEqual(rows[0]["coordinate_frame"], "NCORE_LOCAL_WORLD")
        self.assertTrue(rows[0]["normalization_verified"])
        self.assertEqual(rows[0]["reference_point"], "cuboid_center")
        self.assertFalse(rows[0]["transform_applied"])

    def test_track_dimensions_remain_constant(self):
        rows = normalize_sequence_tracks("clip", [track_row(timestamp=100, dims=[4.0, 2.0, 1.5]), track_row(timestamp=200, pose_index=1, dims=[4.0, 2.0, 1.5])])
        self.assertEqual([(row["length_m"], row["width_m"], row["height_m"]) for row in rows], [(4.0, 2.0, 1.5)] * 2)

    def test_retained_anomaly_is_not_dropped(self):
        rows = normalize_sequence_tracks("clip", [track_row(timestamp=100), track_row(timestamp=200, pose_index=1)], {("clip", "a", 200)})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["geometry_quality_status"], "RETAINED_LOCALIZED_ANOMALY")

    def test_object_timestamp_is_objects_present(self):
        rows = classify_scorer_queries([100], [{"timestamp_us": 100, "trackline_id": "a"}], tolerance_us=0)
        self.assertEqual(rows[0]["observation_state"], "OBJECTS_PRESENT")
        self.assertEqual(rows[0]["object_count"], 1)

    def test_explicit_empty_attestation_is_confirmed_empty(self):
        rows = classify_scorer_queries([100], [{"timestamp_us": 200, "trackline_id": "a"}], tolerance_us=0, empty_timestamps={100})
        self.assertEqual(rows[0]["observation_state"], "CONFIRMED_EMPTY")
        self.assertTrue(rows[0]["attestation_verified"])

    def test_no_object_and_no_empty_evidence_is_missing(self):
        rows = classify_scorer_queries([100], [{"timestamp_us": 200, "trackline_id": "a"}], tolerance_us=0)
        self.assertEqual(rows[0]["observation_state"], "MISSING")
        self.assertFalse(rows[0]["attestation_verified"])

    def test_label_set_timestamp_alone_is_not_confirmed_empty(self):
        rows = classify_scorer_queries([100], [], label_set_timestamps=[100], tolerance_us=0)
        self.assertEqual(rows[0]["observation_state"], "MISSING")
        self.assertEqual(rows[0]["attestation_source"], "label_set_timestamp_without_empty_semantics")

    def test_cf_tolerance_is_50ms(self):
        self.assertEqual(CF_TOLERANCE_US, 50_000)
        rows = classify_scorer_queries([100_000], [{"timestamp_us": 150_000}], tolerance_us=CF_TOLERANCE_US)
        self.assertEqual(rows[0]["observation_state"], "OBJECTS_PRESENT")

    def test_ttc_tolerance_is_100ms(self):
        self.assertEqual(TTC_TOLERANCE_US, 100_000)
        rows = classify_scorer_queries([100_000], [{"timestamp_us": 200_000}], tolerance_us=TTC_TOLERANCE_US)
        self.assertEqual(rows[0]["observation_state"], "OBJECTS_PRESENT")

    def test_ttc_grid_uses_production_helper(self):
        cf, ttc, grid = build_scorer_query_grid(1_000_000)
        expected = build_ttc_projection_timestamps(np.asarray(cf, dtype=np.int64), 1.0).tolist()
        self.assertEqual(ttc, expected)
        self.assertEqual(grid["ttc_query_builder"], "tools.epdms.observation_contract.build_ttc_projection_timestamps")
        self.assertEqual(len(cf), 41)

    def test_strict_readiness_requires_zero_missing_queries(self):
        rows = classify_scorer_queries([100, 200], [{"timestamp_us": 100}], tolerance_us=0)
        summary = strict_coverage_summary(rows, "CF")
        self.assertEqual(summary["CF_MISSING_COUNT"], 1)
        self.assertFalse(summary["CF_DATA_READY"])

    def test_complete_object_queries_are_ready_without_empty_frames(self):
        rows = classify_scorer_queries([100, 200], [{"timestamp_us": 100}, {"timestamp_us": 200}], tolerance_us=0)
        summary = strict_coverage_summary(rows, "CF")
        self.assertTrue(summary["CF_DATA_READY"])
        self.assertEqual(summary["CF_CONFIRMED_EMPTY_COUNT"], 0)

    def test_pilot_contract_does_not_claim_full_300(self):
        context = build_obstacle_context("clip", 100, normalize_sequence_tracks("clip", [track_row()]))
        self.assertFalse(context["coordinate_alignment_verified"])
        self.assertNotIn("confirmed_empty_timestamps_us", context["semantic_context"]["obstacle"])

    def test_no_fitted_geometry_correction_or_offset_rederive(self):
        row = normalize_sequence_tracks("clip", [track_row()])[0]
        self.assertFalse(row["transform_applied"])
        self.assertFalse(row["per_clip_offset_rederived"])
        self.assertAlmostEqual(row["yaw_rad"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
