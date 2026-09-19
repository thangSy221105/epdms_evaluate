import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools.data_prep import nurec


ENGINE = {"available": True, "engines": ["pyarrow"], "status": "READY"}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


class TestNuRecNestedContracts(unittest.TestCase):
    def _obstacle_frame(self, **overrides):
        row = {
            "key": {"timestamp_micros": 100},
            "obstacle": {
                "trackline_id": "track-1",
                "center": {"x": 1.0, "y": 2.0, "z": 0.5},
                "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "category": "automobile",
            },
        }
        row["key"].update(overrides.pop("key", {}))
        row["obstacle"].update(overrides.pop("obstacle", {}))
        return pd.DataFrame([row])

    def _inspect(self, frame, path_name="obstacle.parquet"):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / path_name
        path.write_bytes(b"fixture")
        patches = [
            mock.patch.object(nurec, "parquet_engine", return_value=ENGINE),
            mock.patch("pandas.read_parquet", return_value=frame),
        ]
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])
        self.addCleanup(td.cleanup)
        return path

    def test_01_canonical_key_timestamp_selected_over_frame_id(self):
        frame = self._obstacle_frame()
        frame["frame_id"] = [999]
        result = nurec.inspect_parquet(self._inspect(frame))
        self.assertEqual(result["selected_timestamp_field"], "key.timestamp_micros")

    def test_02_frame_id_never_controls_timestamp_range(self):
        frame = pd.DataFrame({"key": [{"timestamp_micros": 100}, {"timestamp_micros": 200}], "frame_id": [999, 1]})
        result = nurec._parquet_timestamp_summary(self._inspect(frame))
        self.assertEqual((result["min"], result["max"]), (100, 200))

    def test_03_multiple_physical_timestamp_fields_are_ambiguous(self):
        frame = pd.DataFrame({"timestamp_micros": [100], "timestamp_us": [200]})
        result = nurec.inspect_parquet(self._inspect(frame))
        self.assertEqual(result["timestamp_selection_status"], "AMBIGUOUS_TIMESTAMP_FIELDS")
        self.assertIsNone(result["selected_timestamp_field"])

    def test_04_timestamp_summary_uses_one_selected_field(self):
        frame = pd.DataFrame({"timestamp_micros": [100, 200], "frame_id": [10000, 1]})
        result = nurec._parquet_timestamp_summary(self._inspect(frame))
        self.assertEqual(result["selected_timestamp_field"], "timestamp_micros")
        self.assertEqual(result["unique_count"], 2)
        self.assertEqual(result["median_dt"], 100)

    def test_05_none_timestamp_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(key={"timestamp_micros": None})))
        self.assertEqual(details["missing_timestamp_count"], 1)
        self.assertEqual(details["invalid_row_count"], 1)

    def test_06_boolean_timestamp_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(key={"timestamp_micros": True})))
        self.assertEqual(details["missing_timestamp_count"], 1)

    def test_07_non_integral_timestamp_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(key={"timestamp_micros": 100.5})))
        self.assertEqual(details["missing_timestamp_count"], 1)

    def test_08_integral_float_timestamp_is_accepted(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(key={"timestamp_micros": 100.0})))
        self.assertEqual(details["missing_timestamp_count"], 0)
        self.assertEqual(details["invalid_row_count"], 0)

    def test_09_missing_track_id_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"trackline_id": None})))
        self.assertEqual(details["invalid_track_id_count"], 1)

    def test_10_empty_track_id_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"trackline_id": "  "})))
        self.assertEqual(details["invalid_track_id_count"], 1)

    def test_11_missing_category_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"category": None})))
        self.assertEqual(details["invalid_category_count"], 1)

    def test_12_empty_category_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"category": ""})))
        self.assertEqual(details["invalid_category_count"], 1)

    def test_13_invalid_center_is_counted(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"center": {"x": float("nan"), "y": 2.0, "z": 0.5}})))
        self.assertEqual(details["invalid_center_count"], 1)

    def test_14_zero_size_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"size": {"x": 0.0, "y": 2.0, "z": 1.5}})))
        self.assertEqual(details["invalid_size_count"], 1)

    def test_15_negative_size_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"size": {"x": -1.0, "y": 2.0, "z": 1.5}})))
        self.assertEqual(details["invalid_size_count"], 1)

    def test_16_zero_norm_quaternion_is_invalid(self):
        details = nurec._obstacle_inventory(self._inspect(self._obstacle_frame(obstacle={"orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}})))
        self.assertEqual(details["invalid_orientation_count"], 1)

    def test_17_invalid_row_count_is_not_double_counted(self):
        frame = pd.concat([
            self._obstacle_frame(obstacle={"trackline_id": None, "category": None, "size": {"x": 0.0, "y": 2.0, "z": 1.5}}),
            self._obstacle_frame(key={"timestamp_micros": 200}),
        ], ignore_index=True)
        details = nurec._obstacle_inventory(self._inspect(frame))
        self.assertEqual(details["invalid_row_count"], 1)
        self.assertEqual(details["valid_row_count"], 1)

    def test_18_nested_egomotion_fields_are_discovered(self):
        frame = pd.DataFrame({"key": [{"timestamp_micros": 1}], "egomotion_estimate": [{"location": {"x": 1, "y": 2, "z": 3}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}]})
        result = nurec.inspect_parquet(self._inspect(frame, "egomotion_estimate.parquet"))
        self.assertIn("egomotion_estimate.location.x", result["coordinate_field_candidates"])
        self.assertIn("egomotion_estimate.orientation.w", result["coordinate_field_candidates"])

    def test_19_nested_lane_rails_are_discovered(self):
        frame = pd.DataFrame({"key": [{"map_id": "a"}], "lane": [{"left_rail": [{"x": 0, "y": 0, "z": 0}], "right_rail": [{"x": 1, "y": 0, "z": 0}]}]})
        result = nurec.inspect_parquet(self._inspect(frame, "lane.parquet"))
        self.assertIn("lane.left_rail[].x", result["coordinate_field_candidates"])

    def test_20_nested_intersection_location_is_discovered(self):
        frame = pd.DataFrame({"key": [{"map_id": "a"}], "intersection_area": [{"location": [{"x": 0, "y": 0, "z": 0}]}]})
        result = nurec.inspect_parquet(self._inspect(frame, "intersection_area.parquet"))
        self.assertIn("intersection_area.location[].x", result["coordinate_field_candidates"])

    def test_21_nested_road_boundary_location_is_discovered(self):
        frame = pd.DataFrame({"key": [{"map_id": "a"}], "road_boundary": [{"location": [{"x": 0, "y": 0, "z": 0}]}]})
        result = nurec.inspect_parquet(self._inspect(frame, "road_boundary.parquet"))
        self.assertIn("road_boundary.location[].x", result["coordinate_field_candidates"])

    def test_22_list_of_struct_accessor_traverses_all_points(self):
        value = {"foo": {"bar": [{"x": 1}, {"x": 2}]}}
        self.assertEqual(nurec._path_value(value, "foo.bar[].x"), [1, 2])

    def _map_frames(self):
        points = [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 1.0, "y": 1.0, "z": 0.0}]
        return {
            "lane": pd.DataFrame({"lane": [{"left_rail": points, "right_rail": points}]}),
            "intersection_area": pd.DataFrame({"intersection_area": [{"location": points}]}),
            "road_boundary": pd.DataFrame({"road_boundary": [{"location": points}]}),
            "drivable_space": pd.DataFrame({"drivable_space": [{"location": points}]}),
        }

    def _map_inspector(self, frames):
        def fake_inspect(path, sample_rows=0):
            source = Path(path).stem
            frame = frames[source]
            return {"status": "OK", "row_count": len(frame), "nested_field_paths": nurec._nested_frame_paths(frame)}
        return fake_inspect

    def test_23_missing_drivable_space_is_only_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td); (clip / "clipgt").mkdir()
            for source in ("lane", "intersection_area", "road_boundary"):
                (clip / "clipgt" / f"{source}.parquet").write_bytes(b"fixture")
            frames = self._map_frames()
            with mock.patch.object(nurec, "parquet_engine", return_value=ENGINE), mock.patch.object(nurec, "inspect_parquet", side_effect=self._map_inspector(frames)), mock.patch("pandas.read_parquet", side_effect=lambda path, **_: frames[Path(path).stem]):
                result = nurec._map_status(clip, "READY")
            self.assertEqual(result["status"], "DAC_CANDIDATE_AVAILABLE")
            self.assertTrue(result["dac_candidate_available"])
            self.assertFalse(result["dac_geometry_verified"])
            self.assertFalse(result["ready"])

    def test_24_structural_drivable_geometry_is_not_semantically_verified(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td); (clip / "clipgt").mkdir()
            (clip / "clipgt" / "drivable_space.parquet").write_bytes(b"fixture")
            frames = self._map_frames()
            with mock.patch.object(nurec, "parquet_engine", return_value=ENGINE), mock.patch.object(nurec, "inspect_parquet", side_effect=self._map_inspector(frames)), mock.patch("pandas.read_parquet", side_effect=lambda path, **_: frames[Path(path).stem]):
                result = nurec._map_status(clip, "READY")
            self.assertEqual(result["status"], "UNVERIFIED_DAC_GEOMETRY")
            self.assertFalse(result["ready"])

    def test_25_transform_metadata_is_not_transform_verified(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td); (clip / "clipgt").mkdir()
            (clip / "rig_trajectories.json").write_text("{}", encoding="utf-8")
            (clip / "clipgt" / "calibration_estimate.parquet").write_bytes(b"fixture")
            pred = clip.parent / "pred.jsonl"; gt = clip.parent / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": clip.name, "t0_us": 1, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": clip.name, "t0_us": 1}) + "\n", encoding="utf-8")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "OK", "row_count": 1, "min": 1, "max": 1, "field": "key.timestamp_micros"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 1, "max": 2}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                result = nurec.audit_dataset(clip.parent, pred, gt, clip.parent / "audit")
            row = list(csv_rows(clip.parent / "audit" / "coordinate_contract.csv"))[0]
            self.assertEqual(row["status"], "TRANSFORM_METADATA_AVAILABLE")
            self.assertEqual(row["coordinate_alignment_verified"], "False")

    def test_26_obstacle_timestamps_do_not_count_as_observation_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100}) + "\n", encoding="utf-8")
            fake = {"status": "OK", "row_count": 5, "min": 100, "max": 200, "unique_count": 5, "field": "key.timestamp_micros"}
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value=fake), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 100, "max": 200}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "observation_coverage.csv"))[0]
            self.assertEqual(row["available_frame_count"], "0")
            self.assertEqual(row["obstacle_timestamp_count"], "5")

    def test_27_no_frame_evidence_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 100}) + "\n", encoding="utf-8")
            fake = {"status": "OK", "row_count": 0, "min": None, "max": None, "unique_count": 0, "field": "key.timestamp_micros"}
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value=fake), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 100, "max": 200}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "observation_coverage.csv"))[0]
            self.assertEqual(row["observation_contract_status"], "TIME_ALIGNMENT_UNRESOLVED")

    def test_28_numeric_overlap_does_not_verify_clock(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 150, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 150}) + "\n", encoding="utf-8")
            fake = {"status": "OK", "row_count": 1, "min": 100, "max": 200, "unique_count": 1, "field": "key.timestamp_micros"}
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value=fake), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 100, "max": 200}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "time_alignment.csv"))[0]
            self.assertEqual(row["time_alignment_status"], "UNRESOLVED")
            self.assertEqual(row["numeric_range_overlap"], "True")

    def test_29_verified_mapping_requires_nonempty_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"; ctx = root / "ctx.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1}) + "\n", encoding="utf-8")
            ctx.write_text(json.dumps({"clip_id": "clip-a", "time_alignment": {"verified": True}}) + "\n", encoding="utf-8")
            fake = {"status": "OK", "row_count": 1, "min": 1, "max": 2, "unique_count": 1, "field": "key.timestamp_micros"}
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value=fake), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "OK", "min": 1, "max": 2}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                nurec.audit_dataset(root, pred, gt, root / "audit", ctx)
            row = list(csv_rows(root / "audit" / "time_alignment.csv"))[0]
            self.assertEqual(row["time_alignment_status"], "UNRESOLVED")

    def test_30_malformed_prediction_json_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text("{bad json\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1}) + "\n", encoding="utf-8")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertEqual(result["errors"][0]["failure_type"], "MALFORMED_JSON")

    def test_31_malformed_ground_truth_json_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text("not-json\n", encoding="utf-8")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertEqual(next(error for error in result["errors"] if error["source"] == "ground_truth")["source"], "ground_truth")

    def test_32_duplicate_prediction_clip_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text("\n".join([json.dumps({"clip_id": "clip-a", "t0_us": 1, "mode": "cross_scene", "alpha": 0}), json.dumps({"clip_id": "clip-a", "t0_us": 2, "mode": "cross_scene", "alpha": 0})]) + "\n", encoding="utf-8")
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1}) + "\n", encoding="utf-8")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertIn("PREDICTION_CONDITION_DUPLICATE", result["contracts"]["clip-a"]["blockers"])

    def test_33_duplicate_ground_truth_clip_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = root / "clip-a"; (clip / "clipgt").mkdir(parents=True)
            pred = root / "pred.jsonl"; gt = root / "gt.jsonl"
            pred.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 1, "mode": "cross_scene", "alpha": 0}) + "\n", encoding="utf-8")
            gt.write_text("\n".join([json.dumps({"clip_id": "clip-a", "t0_us": 1}), json.dumps({"clip_id": "clip-a", "t0_us": 2})]) + "\n", encoding="utf-8")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_parquet_clip_interval_summary", return_value={"status": "FILE_NOT_FOUND"}), mock.patch.object(nurec, "_map_status", return_value={"status": "FILE_NOT_FOUND", "ready": False, "dac_candidate_available": False, "dac_geometry_verified": False}):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertIn("GT_DUPLICATE", result["contracts"]["clip-a"]["blockers"])


if __name__ == "__main__":
    unittest.main()
