"""Audit official PhysicalAI obstacle.offline data against local NuRec rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:  # pragma: no cover - dependency error is reported by CLI
    pq = None

try:
    from scripts.prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _pose, _wrap, _csv, _dump
except ModuleNotFoundError:
    from prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _pose, _wrap, _csv, _dump

PILOT_CLIPS = [
    "028508ba-ef59-48d3-a95b-94eb92e3b063",
    "d078258b-9339-425d-a040-68346ef0d5bc",
    "689889c5-95b0-42ce-a1c9-f97a4388cb28",
    "37f45f87-dc3b-4425-a388-fa7bfa4a11a6",
    "bb1b395f-c51d-4a16-87ad-7310a7bbf086",
]
PUBLIC_NCORE_CONVERTER = "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
PUBLIC_NCORE_API = "https://nvidia.github.io/nurec/ncore/reference/apis/data.v3.html"


def _percentile(values, q=0.95):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def classify_representation(columns, rows):
    names = set(columns)
    if "object_count" in names:
        return "FRAME_ROW_WITH_OBJECT_COUNT"
    if any(name in names for name in ("objects", "cuboids", "obstacles")):
        return "FRAME_GROUPED_OBJECT_LIST"
    if rows and "timestamp_us" in names and "track_id" in names:
        return "OBJECT_ROWS_ONLY"
    return "UNKNOWN"


def wrapped_yaw_difference(quaternion_a, quaternion_b):
    return abs(_wrap(pose_yaw(_pose([0.0, 0.0, 0.0], quaternion_a)) - pose_yaw(_pose([0.0, 0.0, 0.0], quaternion_b))))


def exact_match_index(rows):
    return {(str(row["track_id"]), int(row["timestamp_us"])): row for row in rows}


def group_rows_by_mapped_timestamp(rows, offset_us):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["timestamp_us"]) + int(offset_us), []).append(row)
    return grouped


def nearest_group(query_us, grouped, tolerance_us):
    if not grouped:
        return None, None
    nearest = min(grouped, key=lambda timestamp: abs(timestamp - query_us))
    delta = abs(nearest - query_us)
    return (nearest, delta) if delta <= tolerance_us else (None, delta)


def classify_track_gap(gap_us, expected_step_us):
    if expected_step_us is None or gap_us <= expected_step_us * 1.5:
        return "NO_INTERIOR_GAP_IN_SOURCE_GRID"
    return "SOURCE_TRACK_GAP_PRESENT"


def _read_presence(path, clip_ids):
    table = pq.read_table(path)
    rows = {str(r["clip_id"]): r for r in table.to_pylist()}
    result = []
    for cid in clip_ids:
        row = rows.get(cid)
        result.append({
            "clip_id": cid,
            "obstacle_offline_feature_present": None if row is None else bool(row.get("obstacle.offline")),
            "egomotion_offline_feature_present": None if row is None else bool(row.get("egomotion.offline")),
            "feature_presence_status": "UNKNOWN_CLIP_ID" if row is None else "FOUND_IN_OFFICIAL_FEATURE_PRESENCE",
        })
    return result


def _find_member(pai_root, clip_id):
    for path in sorted((Path(pai_root) / "labels" / "obstacle.offline").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if clip_id in n and n.endswith(".parquet")]
            if members:
                return path, members[0], archive.read(members[0])
    direct = list(Path(pai_root).rglob(f"{clip_id}.obstacle.offline.parquet"))
    if direct:
        return direct[0], direct[0].name, direct[0].read_bytes()
    return None, None, None


def _schema_report(data, source_path, source_member):
    table = pq.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    timestamps = [int(r["timestamp_us"]) for r in rows if r.get("timestamp_us") is not None]
    gaps = sorted(b - a for a, b in zip(sorted(set(timestamps)), sorted(set(timestamps))[1:]) if b > a)
    columns = table.column_names
    null_counts = {name: sum(row.get(name) is None for row in rows) for name in columns}
    categories = sorted({str(r.get("label_class")) for r in rows if r.get("label_class") is not None})
    return {
        "source_path": str(source_path), "source_member": source_member, "row_count": len(rows),
        "column_names": columns, "arrow_schema": str(table.schema), "null_counts": null_counts,
        "sample_rows": rows[:5], "unique_timestamp_count": len(set(timestamps)),
        "timestamp_min_us": min(timestamps) if timestamps else None, "timestamp_max_us": max(timestamps) if timestamps else None,
        "timestamp_median_step_us": statistics.median(gaps) if gaps else None,
        "timestamp_p95_step_us": _percentile(gaps), "timestamp_min_step_us": min(gaps) if gaps else None,
        "timestamp_max_step_us": max(gaps) if gaps else None, "unique_track_count": len({str(r.get("track_id")) for r in rows}),
        "category_values": categories, "representation_type": classify_representation(columns, rows),
        "empty_semantics_fields": [name for name in columns if name in {"object_count", "objects", "cuboids", "obstacles", "annotation_status", "frame_present", "annotated"}],
    }, rows


def _crosscheck(clip_id, pai_rows, local_rows, offset_us):
    local = exact_match_index(local_rows)
    pairs = []
    for pai in pai_rows:
        key = (str(pai["track_id"]), int(pai["timestamp_us"]) + int(offset_us))
        nu = local.get(key)
        if not nu:
            continue
        center = [pai["center_x"], pai["center_y"], pai["center_z"]]
        center_residual = math.dist(center, nu["center"])
        yaw_residual = wrapped_yaw_difference([pai["orientation_x"], pai["orientation_y"], pai["orientation_z"], pai["orientation_w"]], nu["quaternion"])
        dimension_residual = max(abs(pai[k] - nu["dimensions"][i]) for i, k in enumerate(("size_x", "size_y", "size_z")))
        pairs.append({"clip_id": clip_id, "track_id": str(pai["track_id"]), "pai_timestamp_us": int(pai["timestamp_us"]), "nurec_timestamp_us": key[1], "center_residual_m": center_residual, "yaw_residual_deg": math.degrees(yaw_residual), "dimension_max_abs_m": dimension_residual, "identity_match": "EXACT_TRACK_ID_AND_MAPPED_TIMESTAMP"})
    centers = [r["center_residual_m"] for r in pairs]
    yaws = [r["yaw_residual_deg"] for r in pairs]
    dims = [r["dimension_max_abs_m"] for r in pairs]
    summary = {"clip_id": clip_id, "matched_observation_count": len(pairs), "center_rmse_m": math.sqrt(sum(v * v for v in centers) / len(centers)) if centers else None, "center_median_m": statistics.median(centers) if centers else None, "center_p95_m": _percentile(centers), "center_max_m": max(centers) if centers else None, "yaw_rmse_deg": math.sqrt(sum(v * v for v in yaws) / len(yaws)) if yaws else None, "yaw_p95_deg": _percentile(yaws), "yaw_max_deg": max(yaws) if yaws else None, "dimension_max_abs_m": max(dims) if dims else None, "per_clip_offset_rederived": False}
    return pairs, summary


def _timestamp_coverage(pai_rows, local_rows, offset_us):
    pai = {int(r["timestamp_us"]) + int(offset_us) for r in pai_rows}
    local = {int(r["timestamp_us"]) for r in local_rows}
    return {"pai_obstacle_timestamp_count": len(pai), "nurec_sequence_track_timestamp_count": len(local), "common_timestamp_count": len(pai & local), "pai_only_timestamp_count": len(pai - local), "nurec_only_timestamp_count": len(local - pai), "offset_reused": True}


def _gap_rows(clip_id, pai_rows, local_rows, offset_us):
    out = []
    by_pai = {}
    for row in pai_rows:
        by_pai.setdefault(str(row["track_id"]), []).append(int(row["timestamp_us"]))
    local_keys = exact_match_index(local_rows)
    for track_id, timestamps in by_pai.items():
        unique = sorted(set(timestamps)); gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
        step = min(gaps) if gaps else None
        for a, b in zip(unique, unique[1:]):
            if b <= a:
                continue
            status = classify_track_gap(b - a, step)
            out.append({"clip_id": clip_id, "track_id": track_id, "pai_first_timestamp_us": a, "pai_last_timestamp_us": b, "gap_us": b - a, "expected_step_us": step, "gap_status": status, "local_missing_observation_count": 0})
        for row in pai_rows:
            if str(row["track_id"]) == track_id and (track_id, int(row["timestamp_us"]) + int(offset_us)) not in local_keys:
                out.append({"clip_id": clip_id, "track_id": track_id, "pai_first_timestamp_us": int(row["timestamp_us"]), "pai_last_timestamp_us": int(row["timestamp_us"]), "gap_us": 0, "expected_step_us": step, "gap_status": "SOURCE_HAS_OBSERVATION_LOCAL_MISSING_CANDIDATE", "classification_confidence": "CANDIDATE_ONLY", "local_missing_observation_count": 1})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pai-root", required=True)
    parser.add_argument("--feature-presence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if pq is None:
        raise SystemExit("PARQUET_ENGINE_UNAVAILABLE: install pyarrow")
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = [item for item in manifest if item["clip_id"] in PILOT_CLIPS]
    clip_ids = [item["clip_id"] for item in manifest]
    presence = _read_presence(Path(args.feature_presence), clip_ids)
    _csv(root / "feature_presence_5clip.csv", presence)
    time_records = {json.loads(line)["clip_id"]: json.loads(line) for line in Path(args.time_alignment_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()}
    schema_rows, inventory_rows, cross_rows, coverage_rows, gap_rows, cf_recheck, ttc_recheck, cross_summaries = [], [], [], [], [], [], [], []
    pai_available = []
    for item in manifest:
        cid = item["clip_id"]; clip = Path(item["nurec_clip_dir"]); out = root / cid; out.mkdir(parents=True, exist_ok=True)
        source_path, source_member, data = _find_member(args.pai_root, cid)
        presence_row = next(row for row in presence if row["clip_id"] == cid)
        if data is None:
            inventory_rows.append({"clip_id": cid, "feature_present": presence_row["obstacle_offline_feature_present"], "file_found": False, "status": "MISSING_OFFICIAL_FILE"}); continue
        pai_available.append(cid); target = out / "obstacle.offline.parquet"; target.write_bytes(data)
        schema, pai_rows = _schema_report(data, source_path, source_member); schema["clip_id"] = cid; schema_rows.append(schema)
        inventory_rows.append({"clip_id": cid, "feature_present": True, "file_found": True, "row_count": len(pai_rows), "representation_type": schema["representation_type"], "reference_frames": sorted({str(r.get("reference_frame")) for r in pai_rows}), "sha256": hashlib.sha256(data).hexdigest(), "file_size_bytes": len(data), "status": "FOUND"})
        _dump(out / "provenance.json", {"dataset_repo": "nvidia/PhysicalAI-Autonomous-Vehicles", "revision": "main", "download_source": str(source_path), "archive_member": source_member, "file_size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "download_timestamp_utc": datetime.now(timezone.utc).isoformat()})
        _dump(out / "schema.json", schema)
        local_rows, _, _ = load_sequence_tracks(clip / "sequence_tracks.json")
        offset = int(time_records[cid]["offset_us"]); pairs, cross = _crosscheck(cid, pai_rows, local_rows, offset); cross_rows.extend(pairs); cross["pai_row_count"] = len(pai_rows); cross["nurec_row_count"] = len(local_rows); cross_summaries.append(cross); _dump(out / "crosscheck_summary.json", cross)
        coverage = {"clip_id": cid, **_timestamp_coverage(pai_rows, local_rows, offset)}; coverage_rows.append(coverage); gap_rows.extend(_gap_rows(cid, pai_rows, local_rows, offset))
        mapped_by_ts = group_rows_by_mapped_timestamp(pai_rows, offset)
        cf_queries, ttc_queries = build_queries(time_records[cid]["nurec_t0_us"])
        def recheck(queries, tolerance, target):
            for query in queries:
                nearest, delta = nearest_group(query, mapped_by_ts, tolerance)
                target.append({"clip_id": cid, "query_timestamp_us": query, "nearest_annotation_timestamp_us": nearest, "delta_us": delta, "object_count": len(mapped_by_ts[nearest]) if nearest is not None else 0, "frame_annotation_status": "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN" if nearest is not None else "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN", "authoritative_frame_coverage": False, "complete_observation": False, "query_ready": False})
        recheck(cf_queries, 50_000, cf_recheck); recheck(ttc_queries, 100_000, ttc_recheck)
        schema_rows[-1]["timestamp_semantics"] = "CLIP_RELATIVE_MICROSECONDS; reference_frame_timestamp_us_EQUALS_timestamp_us"; schema_rows[-1]["empty_semantics"] = "OBJECT_ROWS_ONLY_NO_EXPLICIT_EMPTY"
    _csv(root / "pai_obstacle_offline_inventory.csv", inventory_rows); _dump(root / "pai_obstacle_offline_schema.json", {"clips": schema_rows, "status": "INSPECTED_OFFICIAL_FILES" if schema_rows else "NO_FILES"}); _csv(root / "pai_timestamp_analysis.csv", schema_rows); _csv(root / "pai_to_nurec_crosscheck.csv", cross_rows); _csv(root / "pai_vs_nurec_timestamp_coverage.csv", coverage_rows); _csv(root / "pai_track_gap_classification.csv", gap_rows)
    _dump(root / "pai_empty_frame_semantics.json", {"status": "UNRESOLVED_EMPTY_ATTESTATION", "representation": "OBJECT_ROWS_ONLY", "explicit_zero_count": False, "explicit_empty_list": False, "missing_timestamp_is_not_empty": True})
    _dump(root / "pai_annotation_timeline_contract.json", {"status": "UNRESOLVED", "feature_presence_is_availability_only": True, "obstacle_timestamp_index_is_object_row_index_only": True, "authoritative_frame_processed_evidence": False})
    _dump(root / "pai_to_ncore_field_mapping.json", {"status": "PUBLIC_CONVERTER_FIELD_MAPPING", "mapping": {"track_id": "CuboidTrackObservation.track_id", "timestamp_us": "CuboidTrackObservation.timestamp_us", "label_class": "CuboidTrackObservation.class_id", "center_x/y/z+size_x/y/z+orientation_x/y/z/w": "CuboidTrackObservation.bbox3", "reference_frame": "CuboidTrackObservation.reference_frame_id", "reference_frame_timestamp_us": "CuboidTrackObservation.reference_frame_timestamp_us"}, "source": PUBLIC_NCORE_CONVERTER, "api": PUBLIC_NCORE_API})
    all_cross = [r for r in cross_rows]
    _dump(root / "pai_obstacle_offline_final_summary.json", {"pilot_clip_count": len(clip_ids), "feature_presence_source_status": "OFFICIAL_FEATURE_PRESENCE_INSPECTED", "obstacle_offline_available_clips": pai_available, "obstacle_offline_missing_clips": [cid for cid in clip_ids if cid not in pai_available], "pai_obstacle_schema_status": "INSPECTED_OFFICIAL_FILES", "pai_obstacle_representation_type": "OBJECT_ROWS_ONLY", "pai_obstacle_timestamp_semantics": "CLIP_RELATIVE_MICROSECONDS", "pai_empty_frame_semantics": "UNRESOLVED_EMPTY_ATTESTATION", "authoritative_annotation_timeline_status": "UNRESOLVED", "pai_to_ncore_field_mapping_status": "PUBLIC_CONVERTER_FIELD_MAPPING", "pai_ncore_empty_frame_preservation": "A_OBJECT_ROWS_ONLY_NO_EXPLICIT_EMPTY_STORAGE", "pai_nurec_matched_observation_count": len(all_cross), "max_center_rmse_m": max((r["center_rmse_m"] for r in cross_summaries if r["center_rmse_m"] is not None), default=None), "max_center_p95_m": max((r["center_p95_m"] for r in cross_summaries if r["center_p95_m"] is not None), default=None), "max_yaw_rmse_deg": max((r["yaw_rmse_deg"] for r in cross_summaries if r["yaw_rmse_deg"] is not None), default=None), "max_yaw_p95_deg": max((r["yaw_p95_deg"] for r in cross_summaries if r["yaw_p95_deg"] is not None), default=None), "raw_source_gap_count": sum(r["gap_status"] == "SOURCE_TRACK_GAP_PRESENT" for r in gap_rows), "local_extraction_missing_observation_count": None, "local_extraction_missing_observation_status": "UNRESOLVED_CANDIDATES_NOT_YET_FRAME_ALIGNED", "local_extraction_missing_observation_candidate_count": sum(r["gap_status"] == "SOURCE_HAS_OBSERVATION_LOCAL_MISSING_CANDIDATE" for r in gap_rows), "cf_required_query_count": len(cf_recheck), "cf_complete_query_count": 0, "cf_unknown_query_count": len(cf_recheck), "cf_complete_observation_rate": None, "cf_data_ready": False, "ttc_required_query_count": len(ttc_recheck), "ttc_complete_query_count": 0, "ttc_unknown_query_count": len(ttc_recheck), "ttc_complete_observation_rate": None, "ttc_data_ready": False, "per_clip_offset_rederived": False, "remaining_blockers": ["NO_EXPLICIT_EMPTY_FRAME_SEMANTICS", "NO_AUTHORITATIVE_ANNOTATION_TIMELINE", "LOCAL_OBSTACLE_FRAME_TRANSFORM_REMAINS_PARTIAL", "PAI_NUREC_GEOMETRY_CROSSCHECK_NOT_FRAME_VERIFIED"], "recommended_next_step": "Keep CF/TTC blocked; obtain an explicit frame-level annotation manifest or accepted complete-frame contract, then resolve the PAI/NuRec frame transform before consuming geometry."})
    _csv(root / "cf_completeness_recheck.csv", cf_recheck); _csv(root / "ttc_completeness_recheck.csv", ttc_recheck)
    _dump(root / "cf_completeness_recheck_summary.json", {"required_query_count": len(cf_recheck), "complete_query_count": sum(r["query_ready"] for r in cf_recheck), "unknown_query_count": sum(not r["query_ready"] for r in cf_recheck), "complete_observation_rate": None, "data_ready": False})
    _dump(root / "ttc_completeness_recheck_summary.json", {"required_query_count": len(ttc_recheck), "complete_query_count": sum(r["query_ready"] for r in ttc_recheck), "unknown_query_count": sum(not r["query_ready"] for r in ttc_recheck), "complete_observation_rate": None, "data_ready": False})


if __name__ == "__main__":
    main()
