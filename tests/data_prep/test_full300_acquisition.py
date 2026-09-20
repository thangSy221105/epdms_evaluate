import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.acquire_full300_data import (
    acquire_pai_clip,
    find_member,
    load_existing_sequence_rows,
    local_central_directory,
    parse_central_directory,
    scan_pai_archives,
    validate_ego_bytes,
)


class Full300AcquisitionTests(unittest.TestCase):
    def _ego_parquet(self) -> bytes:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "timestamp": [-1, 0, 100_000],
                "qx": [0.0, 0.0, 0.0],
                "qy": [0.0, 0.0, 0.0],
                "qz": [0.0, 0.0, 0.0],
                "qw": [1.0, 1.0, 1.0],
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            }
        )
        sink = io.BytesIO()
        pq.write_table(table, sink)
        return sink.getvalue()

    def test_zip_central_directory_and_exact_member_selection(self):
        clip_id = "00000000-0000-0000-0000-000000000001"
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"nested/{clip_id}.egomotion.offline.parquet", b"payload")
        raw = archive_bytes.getvalue()
        central_start = raw.index(b"PK\x01\x02")
        members = parse_central_directory(raw[central_start : raw.rindex(b"PK\x05\x06")])
        self.assertIn(f"nested/{clip_id}.egomotion.offline.parquet", members)
        self.assertEqual(find_member(members, f"{clip_id}.egomotion.offline.parquet")[0], f"nested/{clip_id}.egomotion.offline.parquet")

    def test_duplicate_basename_is_ambiguous_and_not_guessed(self):
        members = {
            "a/sequence_tracks.json": {"filename": "a/sequence_tracks.json"},
            "b/sequence_tracks.json": {"filename": "b/sequence_tracks.json"},
        }
        self.assertIsNone(find_member(members, "sequence_tracks.json"))

    def test_pai_scan_uses_offline_suffix_and_exact_uuid(self):
        clip_id = "00000000-0000-0000-0000-000000000002"
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "egomotion.offline.chunk_0001.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"x/{clip_id}.egomotion.offline.parquet", b"x")
                archive.writestr(f"x/{clip_id}.egomotion.parquet", b"wrong-component")
            remote_name = "labels/egomotion.offline/egomotion.offline.chunk_0001.zip"
            hits, errors = scan_pai_archives(
                [{"path": remote_name, "size": archive_path.stat().st_size, "url": "unused"}],
                {remote_name: archive_path},
                workers=1,
                target_ids={clip_id},
            )
            self.assertEqual(errors, [])
            self.assertEqual(hits[clip_id]["member"], f"x/{clip_id}.egomotion.offline.parquet")

    def test_existing_valid_pai_file_is_reused_without_download(self):
        clip_id = "00000000-0000-0000-0000-000000000003"
        with tempfile.TemporaryDirectory() as td:
            raw = self._ego_parquet()
            destination = Path(td) / "physicalai_offline" / clip_id / "egomotion.offline.parquet"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(raw)
            result = acquire_pai_clip(
                clip_id,
                {
                    "archive": "unused.zip",
                    "member": "unused.parquet",
                    "member_metadata": {},
                    "compressed_size_bytes": 0,
                },
                Path(td),
            )
            self.assertEqual(result["status"], "ALREADY_PRESENT")
            self.assertEqual(result["downloaded_bytes"], 0)
            self.assertTrue(result["schema_valid"])

    def test_sequence_tracks_dummy_chunk_wrapper_is_validated(self):
        value = {
            "dummy_chunk_id": {
                "tracks_data": {
                    "tracks_id": ["track-1"],
                    "tracks_poses": [[[0, 0, 0, 0, 0, 0, 1]]],
                    "tracks_timestamps_us": [[123]],
                    "tracks_label_class": ["car"],
                },
                "cuboidtracks_data": {"cuboids_dims": [[4, 2, 1]]},
            }
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sequence_tracks.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            result = load_existing_sequence_rows(path)
            self.assertEqual(result["read_status"], "READ")
            self.assertEqual(result["schema_status"], "VALID")
            self.assertEqual(result["row_count"], 1)

    def test_invalid_ego_schema_is_fail_closed(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        sink = io.BytesIO()
        pq.write_table(pa.table({"timestamp": [0, 1]}), sink)
        result = validate_ego_bytes(sink.getvalue())
        self.assertFalse(result["schema_valid"])
        self.assertTrue(result["status"].startswith("MISSING_COLUMNS"))


if __name__ == "__main__":
    unittest.main()
