"""Audit NuRec track and frame observation completeness without scoring.

The audit distinguishes object-row presence from complete annotation evidence.
It never treats an absent obstacle row as an empty frame unless an independent
frame-level attestation is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from scripts.prepare_nurec_obstacles import (
        CF_TOLERANCE_US, TTC_TOLERANCE_US, build_queries, load_obstacle_rows,
        load_sequence_tracks, load_time_record, _csv, _dump,
    )
except ModuleNotFoundError:
    from prepare_nurec_obstacles import (
        CF_TOLERANCE_US, TTC_TOLERANCE_US, build_queries, load_obstacle_rows,
        load_sequence_tracks, load_time_record, _csv, _dump,
    )


def _p95(values):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))]


def _finite_vec(values, size):
    return isinstance(values, (list, tuple)) and len(values) == size and all(
        isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values
    )


def _required_fields_valid(row):
    if not row.get("track_id") or not isinstance(row.get("timestamp_us"), int):
        return False
    if not _finite_vec(row.get("center"), 3):
        return False
    if not _finite_vec(row.get("quaternion"), 4) or sum(float(v) * float(v) for v in row["quaternion"]) <= 1e-12:
        return False
    if not _finite_vec(row.get("dimensions"), 3):
        return False
    if any(float(v) <= 0 for v in row["dimensions"]):
        return False
    return row.get("category") not in (None, "")


def _derive_expected_step(gaps):
    positive = [g for g in gaps if g > 0]
    if not positive:
        return None, "NO_POSITIVE_TIMESTAMP_GAPS"
    # The smallest stable positive delta is the safest source-derived base
    # cadence: a missing interior sample otherwise inflates the median and
    # hides the gap (e.g. 100 ms, 200 ms should imply a missing 100 ms slot).
    step = int(min(positive))
    distribution = {}
    for gap in positive:
        distribution[str(gap)] = distribution.get(str(gap), 0) + 1
    return step, "MIN_POSITIVE_SOURCE_TIMESTAMP_GAP"


def _group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["track_id"], []).append(row)
    return grouped


def audit_tracks(rows, errors=None):
    errors = errors or []
    by = _group_rows(rows)
    errors_by_track = {}
    for error in errors:
        if error.get("track_id") is not None:
            errors_by_track.setdefault(str(error["track_id"]), []).append(error)
    all_gaps = []
    for track_rows in by.values():
        ordered = sorted(track_rows, key=lambda r: r.get("source_pose_index", 0))
        all_gaps.extend(b["timestamp_us"] - a["timestamp_us"] for a, b in zip(ordered, ordered[1:]))
    global_step, cadence_source = _derive_expected_step(all_gaps)
    results = []
    for track_id, track_rows in sorted(by.items()):
        source_order = sorted(track_rows, key=lambda r: r.get("source_pose_index", 0))
        timestamps = [r["timestamp_us"] for r in source_order]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        positive = [g for g in gaps if g > 0]
        expected_step = int(min(positive)) if positive else global_step
        duplicate_count = sum(g == 0 for g in gaps)
        nonmonotonic_count = sum(g < 0 for g in gaps)
        missing_slots = 0
        if expected_step:
            missing_slots = sum(max(0, int(round(g / expected_step)) - 1) for g in positive if g > expected_step * 1.5)
        irregular = bool(positive and expected_step and any(abs(g - expected_step) > max(25_000, expected_step * 0.20) and g <= expected_step * 1.5 for g in positive))
        invalid = bool(errors_by_track.get(str(track_id))) or any(not _required_fields_valid(r) for r in track_rows) or duplicate_count > 0 or nonmonotonic_count > 0
        if invalid:
            status = "INVALID_TIMESTAMPS"
        elif len(timestamps) == 1:
            status = "SINGLE_OBSERVATION_ONLY"
        elif expected_step is None:
            status = "UNKNOWN_EXPECTED_CADENCE"
        elif missing_slots:
            status = "HAS_INTERIOR_GAPS"
        elif irregular:
            status = "IRREGULAR_CADENCE"
        else:
            status = "COMPLETE_ON_OBSERVED_GRID"
        results.append({
            "track_id": track_id,
            "first_timestamp_us": min(timestamps),
            "last_timestamp_us": max(timestamps),
            "observation_count": len(timestamps),
            "expected_step_us": expected_step,
            "expected_rate_hz": (1_000_000 / expected_step) if expected_step else None,
            "cadence_source": cadence_source,
            "median_gap_us": statistics.median(positive) if positive else None,
            "p95_gap_us": _p95(positive),
            "max_gap_us": max(positive) if positive else None,
            "interior_missing_slot_count": missing_slots,
            "duplicate_timestamp_count": duplicate_count,
            "nonmonotonic_count": nonmonotonic_count,
            "track_completeness_status": status,
        })
    return results, {"global_expected_step_us": global_step, "global_expected_rate_hz": (1_000_000 / global_step) if global_step else None, "cadence_source": cadence_source}


def discover_annotation_timeline(clip, sequence_rows, obstacle_rows):
    """Return evidence inventory; local sources are not completeness attestations."""
    sources = []
    for rel in (
        "data_info.json", "datasource_summary.json", "sequence_tracks.json",
        "clipgt/clip.parquet", "clipgt/association.parquet", "clipgt/obstacle.parquet",
    ):
        path = clip / rel
        sources.append({"source": rel, "exists": path.is_file(), "semantics": "TEMPORAL_REFERENCE_ONLY" if rel in ("data_info.json", "datasource_summary.json") else "OBJECT_ROWS_ONLY"})
    return {
        "sources": sources,
        "physicalai_obstacle_offline_status": "NOT_AVAILABLE_IN_LOCAL_CACHE",
        "ncore_cuboids_component_status": "PUBLIC_SEMANTICS_NOT_LOCALLY_INSPECTED",
        "authoritative_frame_timeline_found": False,
        "frame_zero_count_field_found": False,
        "empty_frame_attestation_found": False,
        "obstacle_representation": "OBJECT_ROWS_ONLY" if obstacle_rows or sequence_rows else "NO_OBJECT_ROWS",
        "status": "UNRESOLVED_EMPTY_ATTESTATION",
    }


def _nearest(query, timestamps, tolerance):
    if not timestamps:
        return None, None
    nearest = min(timestamps, key=lambda t: abs(t - query))
    delta = abs(nearest - query)
    return (nearest, delta) if delta <= tolerance else (None, delta)


def frame_completeness(required_timestamps, object_rows, tolerance):
    by_ts = {}
    for row in object_rows:
        by_ts.setdefault(row["timestamp_us"], 0)
        by_ts[row["timestamp_us"]] += 1
    source_ts = sorted(by_ts)
    rows = []
    for timestamp in sorted(set(required_timestamps)):
        nearest, delta = _nearest(timestamp, source_ts, tolerance)
        if nearest is None:
            status = "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN"
            count = 0
            present = False
        else:
            status = "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN"
            count = by_ts[nearest]
            present = True
        rows.append({
            "timestamp_us": timestamp,
            "object_count": count,
            "object_rows_present": present,
            "annotation_frame_present": present,
            "frame_annotation_status": status,
            "completeness_source": "NO_AUTHORITATIVE_FRAME_ATTESTATION",
            "completeness_verified": False,
            "nearest_source_timestamp_us": nearest,
            "delta_us": delta,
        })
    return rows


def query_completeness(queries, frame_rows, tolerance):
    indexed = {r["timestamp_us"]: r for r in frame_rows}
    available = sorted(indexed)
    out = []
    for query in queries:
        nearest, delta = _nearest(query, available, tolerance)
        frame = indexed.get(nearest) if nearest is not None else None
        out.append({
            "query_timestamp_us": query,
            "nearest_annotation_timestamp_us": nearest,
            "delta_us": delta,
            "object_count": frame["object_count"] if frame else 0,
            "frame_annotation_status": frame["frame_annotation_status"] if frame else "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN",
            "query_ready": bool(frame and frame["completeness_verified"] and frame["frame_annotation_status"] in ("COMPLETE_OBJECTS_PRESENT", "CONFIRMED_EMPTY")),
        })
    return out


def _summary(rows):
    total = len(rows)
    complete = sum(r["query_ready"] for r in rows)
    unknown = total - complete
    evidence = sum(r["object_count"] > 0 for r in rows)
    return {"required_query_count": total, "complete_query_count": complete, "unknown_query_count": unknown, "temporal_object_evidence_rate": evidence / total if total else 0.0, "complete_observation_rate": complete / total if total and unknown == 0 else None, "data_ready": complete == total and total > 0}


def process_clip(item, time_path):
    cid = item["clip_id"]
    clip = Path(item["nurec_clip_dir"])
    rec = load_time_record(time_path, cid, int(item.get("physicalai_t0_us", 5_100_000)))
    sequence_rows, sequence_errors, schema = load_sequence_tracks(clip / "sequence_tracks.json")
    obstacle_rows, obstacle_errors = load_obstacle_rows(clip / "clipgt" / "obstacle.parquet")
    tracks, cadence = audit_tracks(sequence_rows, sequence_errors)
    cf, ttc = build_queries(rec["nurec_t0_us"])
    frame_rows = frame_completeness(cf + ttc, sequence_rows, TTC_TOLERANCE_US)
    cf_rows = query_completeness(cf, frame_rows, CF_TOLERANCE_US)
    ttc_rows = query_completeness(ttc, frame_rows, TTC_TOLERANCE_US)
    timeline = discover_annotation_timeline(clip, sequence_rows, obstacle_rows)
    complete_tracks = sum(r["track_completeness_status"] == "COMPLETE_ON_OBSERVED_GRID" for r in tracks)
    gap_tracks = sum(r["track_completeness_status"] == "HAS_INTERIOR_GAPS" for r in tracks)
    single_tracks = sum(r["track_completeness_status"] == "SINGLE_OBSERVATION_ONLY" for r in tracks)
    invalid_tracks = sum(r["track_completeness_status"] == "INVALID_TIMESTAMPS" for r in tracks)
    track_summary = {
        "clip_id": cid, "track_count": len(tracks), "tracks_complete_on_observed_grid": complete_tracks,
        "tracks_with_interior_gaps": gap_tracks, "tracks_single_observation": single_tracks,
        "tracks_invalid": invalid_tracks, "total_track_observations": len(sequence_rows),
        "interior_missing_slot_count": sum(r["interior_missing_slot_count"] for r in tracks),
        "max_track_gap_us": max((r["max_gap_us"] for r in tracks if r["max_gap_us"] is not None), default=None),
        "p95_track_gap_us": _p95([b["timestamp_us"] - a["timestamp_us"] for track_rows in _group_rows(sequence_rows).values() for a, b in zip(sorted(track_rows, key=lambda x: x.get("source_pose_index", 0)), sorted(track_rows, key=lambda x: x.get("source_pose_index", 0))[1:]) if b["timestamp_us"] > a["timestamp_us"]]),
        **cadence,
    }
    return {"clip_id": cid, "clip": clip, "schema": schema, "sequence_errors": sequence_errors, "obstacle_errors": obstacle_errors, "tracks": tracks, "track_summary": track_summary, "timeline": timeline, "frames": frame_rows, "cf": cf_rows, "ttc": ttc_rows, "cf_summary": _summary(cf_rows), "ttc_summary": _summary(ttc_rows), "obstacle_row_count": len(obstacle_rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    time_path = Path(args.time_alignment_jsonl)
    results = []
    all_tracks, all_frames, all_cf, all_ttc = [], [], [], []
    for item in manifest:
        result = process_clip(item, time_path)
        cid = result["clip_id"]
        results.append(result)
        all_tracks.extend({"clip_id": cid, **row} for row in result["tracks"])
        all_frames.extend({"clip_id": cid, **row} for row in result["frames"])
        all_cf.extend({"clip_id": cid, **row} for row in result["cf"])
        all_ttc.extend({"clip_id": cid, **row} for row in result["ttc"])
        out = root / cid
        _dump(out / "track_completeness_summary.json", result["track_summary"])
        _dump(out / "annotation_timeline_contract.json", result["timeline"])
        _dump(out / "empty_frame_semantics.json", {"status": "UNRESOLVED_EMPTY_ATTESTATION", "explicit_zero_count": False, "object_rows_only": True})
        _dump(out / "frame_completeness_summary.json", {"frame_count": len(result["frames"]), "verified_complete_count": 0, "unknown_count": len(result["frames"]), "status": "UNRESOLVED"})
        _dump(out / "cf_completeness_summary.json", result["cf_summary"])
        _dump(out / "ttc_completeness_summary.json", result["ttc_summary"])
        _csv(out / "track_completeness.csv", [{"clip_id": cid, **r} for r in result["tracks"]])
        _csv(out / "frame_completeness.csv", [{"clip_id": cid, **r} for r in result["frames"]])
        _csv(out / "cf_completeness.csv", [{"clip_id": cid, **r} for r in result["cf"]])
        _csv(out / "ttc_completeness.csv", [{"clip_id": cid, **r} for r in result["ttc"]])
        print(f"CONTEXT_RECORD clip={cid} intents=obstacle sources=sequence_tracks,obstacle aligned=true transformed=false ready=false")
        print(f"CLIP_DONE clip={cid} status=complete")
    cf_complete = sum(r["cf_summary"]["complete_query_count"] for r in results)
    ttc_complete = sum(r["ttc_summary"]["complete_query_count"] for r in results)
    cf_required = sum(r["cf_summary"]["required_query_count"] for r in results)
    ttc_required = sum(r["ttc_summary"]["required_query_count"] for r in results)
    summary = {
        "pilot_clip_count": len(results),
        "obstacle_offline_available_clips": [],
        "annotation_timeline_source": "LOCAL_NUREC_OBJECT_ROWS_ONLY_NO_AUTHORITATIVE_FRAME_ATTESTATION",
        "track_count": sum(r["track_summary"]["track_count"] for r in results),
        "tracks_complete_on_observed_grid": sum(r["track_summary"]["tracks_complete_on_observed_grid"] for r in results),
        "tracks_with_interior_gaps": sum(r["track_summary"]["tracks_with_interior_gaps"] for r in results),
        "tracks_single_observation": sum(r["track_summary"]["tracks_single_observation"] for r in results),
        "tracks_invalid": sum(r["track_summary"]["tracks_invalid"] for r in results),
        "total_track_observations": sum(r["track_summary"]["total_track_observations"] for r in results),
        "interior_missing_slot_count": sum(r["track_summary"]["interior_missing_slot_count"] for r in results),
        "max_track_gap_us": max((r["track_summary"]["max_track_gap_us"] for r in results if r["track_summary"]["max_track_gap_us"] is not None), default=None),
        "p95_track_gap_us": _p95([r["track_summary"]["p95_track_gap_us"] for r in results if r["track_summary"]["p95_track_gap_us"] is not None]),
        "track_expected_step_us_distribution": sorted({r["track_summary"]["global_expected_step_us"] for r in results}),
        "track_expected_rate_hz_distribution": sorted({r["track_summary"]["global_expected_rate_hz"] for r in results}),
        "track_completeness_status": "PARTIALLY_VERIFIED",
        "complete_observation_timeline_status": "UNRESOLVED",
        "empty_scene_attestation_status": "UNRESOLVED_EMPTY_ATTESTATION",
        "cf_required_query_count": cf_required, "cf_complete_query_count": cf_complete, "cf_unknown_query_count": cf_required - cf_complete,
        "cf_temporal_object_evidence_rate": sum(r["cf_summary"]["temporal_object_evidence_rate"] * r["cf_summary"]["required_query_count"] for r in results) / cf_required if cf_required else 0.0,
        "cf_complete_observation_rate": cf_complete / cf_required if cf_required and cf_complete == cf_required else None, "cf_data_ready": False,
        "ttc_required_query_count": ttc_required, "ttc_complete_query_count": ttc_complete, "ttc_unknown_query_count": ttc_required - ttc_complete,
        "ttc_temporal_object_evidence_rate": sum(r["ttc_summary"]["temporal_object_evidence_rate"] * r["ttc_summary"]["required_query_count"] for r in results) / ttc_required if ttc_required else 0.0,
        "ttc_complete_observation_rate": ttc_complete / ttc_required if ttc_required and ttc_complete == ttc_required else None, "ttc_data_ready": False,
        "observation_completeness_block_status": "BLOCKED_UNRESOLVED_FRAME_COMPLETENESS",
        "remaining_blockers": ["NO_AUTHORITATIVE_OBSTACLE_ANNOTATION_TIMELINE", "NO_EXPLICIT_EMPTY_FRAME_ATTESTATION", "OBSTACLE_FRAME_TRANSFORM_REMAINS_PARTIAL"],
    }
    _csv(root / "track_completeness.csv", all_tracks)
    _csv(root / "frame_completeness.csv", all_frames)
    _csv(root / "cf_completeness.csv", all_cf)
    _csv(root / "ttc_completeness.csv", all_ttc)
    _dump(root / "track_completeness_summary.json", {"clips": [r["track_summary"] for r in results], "aggregate": summary})
    _dump(root / "annotation_timeline_contract.json", {"clips": [r["timeline"] for r in results], "status": "UNRESOLVED"})
    _dump(root / "empty_frame_semantics.json", {"status": "UNRESOLVED_EMPTY_ATTESTATION", "explicit_zero_count_found": False, "obstacle_offline_available_clips": []})
    _dump(root / "frame_completeness_summary.json", {"required_frame_count": len(all_frames), "verified_complete_count": 0, "unknown_count": len(all_frames), "status": "UNRESOLVED"})
    _dump(root / "cf_completeness_summary.json", {"required_query_count": cf_required, "complete_query_count": cf_complete, "unknown_query_count": cf_required - cf_complete, "temporal_object_evidence_rate": summary["cf_temporal_object_evidence_rate"], "complete_observation_rate": summary["cf_complete_observation_rate"], "data_ready": False})
    _dump(root / "ttc_completeness_summary.json", {"required_query_count": ttc_required, "complete_query_count": ttc_complete, "unknown_query_count": ttc_required - ttc_complete, "temporal_object_evidence_rate": summary["ttc_temporal_object_evidence_rate"], "complete_observation_rate": summary["ttc_complete_observation_rate"], "data_ready": False})
    _dump(root / "observation_completeness_final_summary.json", summary)


if __name__ == "__main__":
    main()
