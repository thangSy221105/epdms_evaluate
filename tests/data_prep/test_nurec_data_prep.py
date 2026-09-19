import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools.data_prep import nurec


class TestNuRecDataPreparation(unittest.TestCase):
    def _clip(self, root: Path, clip_id: str = "clip-a") -> Path:
        clip = root / clip_id
        (clip / "clipgt").mkdir(parents=True)
        return clip

    def _inputs(self, root: Path, clip_id: str = "clip-a", context=None):
        pred = root / "pred.jsonl"
        gt = root / "gt.jsonl"
        pred.write_text(json.dumps({"clip_id": clip_id, "t0_us": 100, "coordinate_frame": "ego", "reference_point": "rear", "alpha": 0.0}) + "\n", encoding="utf-8")
        gt.write_text(json.dumps({"clip_id": clip_id, "t0_us": 100, "future_frame": "ego"}) + "\n", encoding="utf-8")
        ctx = None
        if context is not None:
            ctx = root / "context.jsonl"
            ctx.write_text(json.dumps({"clip_id": clip_id, **context}) + "\n", encoding="utf-8")
        return pred, gt, ctx

    def _patch_parquet(self, obstacle=None, maps=None):
        obstacle = obstacle or {"status": "OK", "row_count": 0, "min": None, "max": None, "unique_count": 0, "field": "timestamp_micros"}
        maps = maps or {"status": "OK", "row_count": 1, "columns": ["geometry"], "valid_polygon_count": 1, "invalid_polygon_count": 0}
        def fake_summary(path):
            if "obstacle.parquet" in str(path):
                return dict(obstacle)
            if "egomotion" in str(path):
                return {"status": "OK", "row_count": 10, "min": 100, "max": 200, "unique_count": 10, "field": "timestamp_micros", "median_dt": 10}
            return {"status": "OK", "row_count": 10, "min": 100, "max": 200, "unique_count": 10, "field": "timestamp_micros"}
        return mock.patch.multiple(nurec, _parquet_timestamp_summary=mock.Mock(side_effect=fake_summary), _map_status=mock.Mock(return_value={"status": "OK", "ready": True, "drivable_space_available": True, "lane_available": True, "intersection_available": True}))

    def test_01_missing_parquet(self):
        with tempfile.TemporaryDirectory() as td:
            result = nurec.inspect_parquet(Path(td) / "missing.parquet")
            self.assertEqual(result["status"], "FILE_NOT_FOUND")

    def test_02_missing_parquet_engine(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(nurec, "parquet_engine", return_value={"available": False, "engines": [], "status": "PARQUET_ENGINE_UNAVAILABLE", "install_hint": "install"}):
            path = Path(td) / "x.parquet"
            path.write_bytes(b"not-a-parquet")
            self.assertEqual(nurec.inspect_parquet(path)["error_code"], "PARQUET_ENGINE_UNAVAILABLE")

    def test_03_valid_obstacle_schema(self):
        frame = pd.DataFrame({"timestamp_micros": [100, 200], "center_x": [1.0, 2.0], "center_y": [0.0, 0.0], "trackline_id": ["a", "a"]})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(nurec, "parquet_engine", return_value={"available": True, "engines": ["pyarrow"], "status": "READY"}), mock.patch("pandas.read_parquet", return_value=frame):
            path = Path(td) / "obstacle.parquet"; path.write_bytes(b"fixture")
            result = nurec.inspect_parquet(path)
            self.assertEqual(result["status"], "OK")
            self.assertIn("timestamp_micros", result["timestamp_field_candidates"])

    def test_03b_nurec_nested_struct_schema_is_inspected(self):
        frame = pd.DataFrame({
            "key": [{"timestamp_micros": 100}, {"timestamp_micros": 200}],
            "obstacle": [{
                "trackline_id": "a",
                "center": {"x": 1.0, "y": 0.0, "z": 0.5},
                "size": {"x": 4.0, "y": 2.0, "z": 1.5},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "category": "automobile",
            }] * 2,
        })
        with tempfile.TemporaryDirectory() as td, mock.patch.object(nurec, "parquet_engine", return_value={"available": True, "engines": ["pyarrow"], "status": "READY"}), mock.patch("pandas.read_parquet", return_value=frame):
            path = Path(td) / "obstacle.parquet"; path.write_bytes(b"fixture")
            result = nurec.inspect_parquet(path)
            self.assertIn("key.timestamp_micros", result["timestamp_field_candidates"])
            self.assertIn("obstacle.center.x", result["coordinate_field_candidates"])
            details = nurec._obstacle_inventory(path)
            self.assertEqual(details["status"], "OK")
            self.assertEqual(details["unique_timestamp_count"], 2)

    def test_04_obstacle_missing_timestamp_is_visible(self):
        frame = pd.DataFrame({"center_x": [1.0], "center_y": [0.0]})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(nurec, "parquet_engine", return_value={"available": True, "engines": ["pyarrow"], "status": "READY"}), mock.patch("pandas.read_parquet", return_value=frame):
            path = Path(td) / "obstacle.parquet"; path.write_bytes(b"fixture")
            result = nurec.inspect_parquet(path)
            self.assertEqual(result["timestamp_field_candidates"], [])

    def test_05_empty_obstacle_without_frame_evidence_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root)
            with self._patch_parquet():
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "observation_coverage.csv"))[0]
            self.assertEqual(row["observation_contract_status"], "UNKNOWN")

    def test_06_empty_obstacle_with_frame_evidence_is_observed_empty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root, context={"observation_frames": [100, 110]})
            with self._patch_parquet():
                nurec.audit_dataset(root, pred, gt, root / "audit", ctx)
            row = list(csv_rows(root / "audit" / "observation_coverage.csv"))[0]
            self.assertEqual(row["observation_contract_status"], "OBSERVED_EMPTY")

    def test_07_mismatched_clocks_are_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root)
            gt.write_text(json.dumps({"clip_id": "clip-a", "t0_us": 999}) + "\n", encoding="utf-8")
            with self._patch_parquet(): nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "time_alignment.csv"))[0]
            self.assertEqual(row["time_alignment_status"], "CONFLICTING_TIME_ORIGIN")

    def test_08_explicit_clock_mapping_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root, context={"time_alignment": {"status": "ALIGNED_BY_EXPLICIT_METADATA", "source": "clip.parquet.start_timestamp"}})
            with self._patch_parquet(): nurec.audit_dataset(root, pred, gt, root / "audit", ctx)
            row = list(csv_rows(root / "audit" / "time_alignment.csv"))[0]
            self.assertEqual(row["time_alignment_status"], "ALIGNED_BY_EXPLICIT_METADATA")

    def test_09_missing_coordinate_frame_is_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root)
            with self._patch_parquet(): nurec.audit_dataset(root, pred, gt, root / "audit")
            row = list(csv_rows(root / "audit" / "coordinate_contract.csv"))[0]
            self.assertIn(row["status"], {"MISSING_FRAME_METADATA", "UNRESOLVED"})

    def test_10_transform_metadata_is_candidate_not_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root, context={"transform_chain_available": True, "transform_source": "calibration_estimate.parquet"})
            with self._patch_parquet(): nurec.audit_dataset(root, pred, gt, root / "audit", ctx)
            row = list(csv_rows(root / "audit" / "coordinate_contract.csv"))[0]
            self.assertEqual(row["status"], "TRANSFORM_AVAILABLE")
            self.assertEqual(row["coordinate_alignment_verified"], "False")

    def test_11_invalid_polygon_is_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root)
            with mock.patch.object(nurec, "_parquet_timestamp_summary", side_effect=lambda path: {"status": "OK", "row_count": 1, "min": 100, "max": 200, "unique_count": 2}), mock.patch.object(nurec, "_map_status", return_value={"status": "NO_VALID_GEOMETRY"}):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertEqual(result["summary"]["map_ready"], 0)

    def test_12_valid_drivable_polygon_can_be_ready_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root); pred, gt, ctx = self._inputs(root)
            with self._patch_parquet():
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertEqual(result["summary"]["map_ready"], 1)

    def test_13_one_bad_clip_does_not_stop_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self._clip(root, "good"); self._clip(root, "bad"); pred, gt, ctx = self._inputs(root, "good")
            with mock.patch.object(nurec, "_parquet_timestamp_summary", side_effect=RuntimeError("corrupt")):
                result = nurec.audit_dataset(root, pred, gt, root / "audit")
            self.assertEqual(result["summary"]["total_clips"], 2)
            self.assertEqual(result["summary"]["audit_error_count"], 2)

    def test_14_eligible_clip_is_staged_only_from_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); audit = root / "audit"; audit.mkdir(); (audit / "contracts").mkdir()
            contract = {"clip_id": "ready", "ready_for_proxy": True, "blockers": [], "map": {"ready": True}, "coordinate": {"verified": True, "transform_required": False}}
            (audit / "contracts" / "ready.json").write_text(json.dumps(contract), encoding="utf-8")
            (audit / "dataset_readiness_summary.json").write_text(json.dumps({"proxy_ready_clip_count": 1, "blocked_clips": []}), encoding="utf-8")
            result = nurec.prepare_staging(audit, root / "prepared")
            self.assertEqual(result["clips_staged"], 1)
            self.assertTrue((root / "prepared" / "ready" / "contract.json").is_file())

    def test_15_raw_files_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); clip = self._clip(root); raw = clip / "raw.txt"; raw.write_text("immutable", encoding="utf-8")
            before = hashlib.sha256(raw.read_bytes()).hexdigest()
            nurec.write_schema_report(clip, root / "schema")
            self.assertEqual(before, hashlib.sha256(raw.read_bytes()).hexdigest())


def csv_rows(path):
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


if __name__ == "__main__":
    unittest.main()
