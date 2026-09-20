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
        CF_TOLERANCE_US, TTC_TOLERANCE_US, build_scorer_query_grid,
        classify_scorer_queries, load_obstacle_rows, load_sequence_tracks,
        load_time_record, normalize_sequence_tracks, build_obstacle_context, strict_coverage_summary,
        load_geometry_anomaly_keys, _csv, _dump, _jsonl,
    )
except ModuleNotFoundError:
    from prepare_nurec_obstacles import (
        CF_TOLERANCE_US, TTC_TOLERANCE_US, build_scorer_query_grid,
        classify_scorer_queries, load_obstacle_rows, load_sequence_tracks,
        load_time_record, normalize_sequence_tracks, build_obstacle_context, strict_coverage_summary,
        load_geometry_anomaly_keys, _csv, _dump, _jsonl,
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


def _nearest(query, timestamps, tolerance):
    if not timestamps:
        return None, None
    nearest = min(timestamps, key=lambda value: abs(value - query))
    delta = abs(nearest - query)
    return (nearest, delta) if delta <= tolerance else (None, delta)


def frame_completeness(required_timestamps, object_rows, tolerance):
    """Legacy diagnostic API: object rows never prove complete annotation."""
    by_timestamp = {}
    for row in object_rows:
        by_timestamp.setdefault(row["timestamp_us"], 0)
        by_timestamp[row["timestamp_us"]] += 1
    source_timestamps = sorted(by_timestamp)
    rows = []
    for timestamp in sorted(set(required_timestamps)):
        nearest, delta = _nearest(timestamp, source_timestamps, tolerance)
        present = nearest is not None
        rows.append({
            "timestamp_us": timestamp,
            "object_count": by_timestamp[nearest] if present else 0,
            "object_rows_present": present,
            "annotation_frame_present": present,
            "frame_annotation_status": "OBJECT_ROWS_PRESENT_BUT_COMPLETENESS_UNKNOWN" if present else "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN",
            "completeness_source": "NO_AUTHORITATIVE_FRAME_ATTESTATION",
            "completeness_verified": False,
            "nearest_source_timestamp_us": nearest,
            "delta_us": delta,
        })
    return rows


def query_completeness(queries, frame_rows, tolerance):
    """Legacy diagnostic API retained for existing unit tests and reports."""
    indexed = {row["timestamp_us"]: row for row in frame_rows}
    output = []
    for query in queries:
        nearest, delta = _nearest(query, sorted(indexed), tolerance)
        frame = indexed.get(nearest) if nearest is not None else None
        output.append({
            "query_timestamp_us": query,
            "nearest_annotation_timestamp_us": nearest,
            "delta_us": delta,
            "object_count": frame["object_count"] if frame else 0,
            "frame_annotation_status": frame["frame_annotation_status"] if frame else "NO_OBJECT_ROWS_COMPLETENESS_UNKNOWN",
            "query_ready": bool(frame and frame["completeness_verified"] and frame["frame_annotation_status"] in ("COMPLETE_OBJECTS_PRESENT", "CONFIRMED_EMPTY")),
        })
    return output


def _metadata_value(path):
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__read_error__": f"{type(exc).__name__}: {exc}"}


_EXPLICIT_EMPTY_FIELDS = {
    "confirmed_empty_timestamps_us",
    "empty_frame_timestamps_us",
    "zero_object_frame_timestamps_us",
}


def _collect_explicit_empty(value, path=""):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in _EXPLICIT_EMPTY_FIELDS and isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        item = next((item.get(name) for name in ("timestamp_us", "timestamp_micros", "timestamp") if item.get(name) is not None), None)
                    try:
                        if item is not None:
                            result.append(int(item))
                    except (TypeError, ValueError):
                        continue
            result.extend(_collect_explicit_empty(child, f"{path}.{key}"))
    elif isinstance(value, list) and len(value) < 1000:
        for index, child in enumerate(value):
            result.extend(_collect_explicit_empty(child, f"{path}[{index}]"))
    return result


def discover_annotation_timeline(clip, sequence_rows, obstacle_rows):
    """Inventory independent timeline evidence without promoting sensor frames."""
    sources = []
    explicit_empty = []
    for relative in (
        "data_info.json", "datasource_summary.json", "metadata.yaml",
        "parsed_config.yaml", "pose_record.json", "rig_trajectories.json",
        "sequence_tracks.json", "clipgt/clip.parquet",
        "clipgt/association.parquet", "clipgt/obstacle.parquet",
    ):
        path = Path(clip) / relative
        item = {"source": relative, "exists": path.is_file()}
        if not path.is_file():
            item["read_status"] = "NOT_FOUND"
        elif path.suffix.lower() in {".json", ".yaml", ".yml"}:
            value = _metadata_value(path)
            if isinstance(value, dict) and "__read_error__" in value:
                item["read_status"] = "READ_ERROR"
                item["error"] = value["__read_error__"]
            else:
                item["read_status"] = "READ"
                item["semantics"] = "OBJECT_ROWS_ONLY" if relative == "sequence_tracks.json" else "SENSOR_OR_METADATA_REFERENCE_ONLY"
                explicit_empty.extend(_collect_explicit_empty(value))
        else:
            item["read_status"] = "PRESENT_NOT_PARSED"
            item["semantics"] = "OBJECT_ROWS_ONLY"
        sources.append(item)
    explicit_empty = sorted(set(explicit_empty))
    object_timestamps = sorted({int(row["timestamp_us"]) for row in sequence_rows})
    return {
        "sources": sources,
        "object_timestamp_count": len(object_timestamps),
        "object_timestamps_us": object_timestamps,
        "label_set_timestamp_source": "sequence_tracks.json.object_rows",
        "label_set_timestamp_status": "OBJECT_TIMESTAMP_ONLY",
        "authoritative_label_set_timestamp_grid_found": False,
        "sensor_frame_timestamps_are_label_attestation": False,
        "explicit_empty_timestamps_us": explicit_empty,
        "empty_observation_attestation_status": "AVAILABLE" if explicit_empty else "NOT_AVAILABLE",
        "label_set_empty_semantics_status": "PROVEN" if explicit_empty else "UNRESOLVED",
        "complete_observation_timeline_status": "FOUND_EXPLICIT_EMPTY_AND_OBJECTS" if explicit_empty else "NOT_FOUND",
        "obstacle_representation": "OBJECT_ROWS_ONLY",
        "physical_world_completeness": "NOT_CLAIMED",
    }


def _query_row_with_clip(clip_id, row, semantics):
    return {
        "clip_id": clip_id,
        **row,
        "label_set_timestamp_available": bool(
            row["nearest_label_set_timestamp_us"] is not None
            and row["label_set_delta_us"] is not None
            and row["label_set_delta_us"] <= semantics["tolerance_us"]
        ),
        "label_set_empty_semantics_status": semantics["label_set_empty_semantics_status"],
    }


def process_clip(item, time_path, anomaly_keys=None):
    clip_id = item["clip_id"]
    clip = Path(item["nurec_clip_dir"])
    record = load_time_record(time_path, clip_id, int(item.get("physicalai_t0_us", 5_100_000)))
    sequence_rows, sequence_errors, schema = load_sequence_tracks(clip / "sequence_tracks.json")
    obstacle_rows, obstacle_errors = load_obstacle_rows(clip / "clipgt" / "obstacle.parquet")
    normalized = normalize_sequence_tracks(clip_id, sequence_rows, anomaly_keys)
    timeline = discover_annotation_timeline(clip, sequence_rows, obstacle_rows)
    cf, ttc, grid = build_scorer_query_grid(record["nurec_t0_us"])
    empty = set(timeline["explicit_empty_timestamps_us"])
    cf_semantics = {"tolerance_us": CF_TOLERANCE_US, **timeline}
    ttc_semantics = {"tolerance_us": TTC_TOLERANCE_US, **timeline}
    cf_rows = classify_scorer_queries(cf, normalized, tolerance_us=CF_TOLERANCE_US, empty_timestamps=empty)
    ttc_rows = classify_scorer_queries(ttc, normalized, tolerance_us=TTC_TOLERANCE_US, empty_timestamps=empty)
    cf_rows = [_query_row_with_clip(clip_id, row, cf_semantics) for row in cf_rows]
    ttc_rows = [_query_row_with_clip(clip_id, row, ttc_semantics) for row in ttc_rows]
    cf_summary = strict_coverage_summary(cf_rows, "CF")
    ttc_summary = strict_coverage_summary(ttc_rows, "TTC")
    anomaly_count = sum(row["geometry_quality_status"] == "RETAINED_LOCALIZED_ANOMALY" for row in normalized)
    summary = {
        "clip_id": clip_id,
        "authoritative_obstacle_source": "sequence_tracks.json",
        "sequence_tracks_pose_frame": "NCORE_LOCAL_WORLD",
        "sequence_tracks_pose_frame_verified": True,
        "obstacle_transform_status": "VERIFIED",
        "normalized_obstacle_status": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "obstacle_row_quality_status": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "obstacle_geometry_block": "CLOSED",
        "coordinate_alignment_verified": False,
        "coordinate_alignment_status": "NOT_ASSERTED_BY_OBSTACLE_NORMALIZATION",
        "time_mapping_source": record.get("source"),
        "time_mapping_verified": bool(record.get("verified", False)),
        "per_clip_offset_rederived": False,
        "label_set_timestamp_status": timeline["label_set_timestamp_status"],
        "label_set_empty_semantics_status": timeline["label_set_empty_semantics_status"],
        "empty_observation_attestation_status": timeline["empty_observation_attestation_status"],
        "complete_observation_timeline_status": timeline["complete_observation_timeline_status"],
        "normalized_obstacle_count": len(normalized),
        "normalized_retained_anomaly_count": anomaly_count,
        "sequence_errors": len(sequence_errors),
        "obstacle_errors": len(obstacle_errors),
        **cf_summary,
        **ttc_summary,
        "query_grid": grid,
    }
    return {
        "summary": summary,
        "schema": schema,
        "normalized": normalized,
        "context": build_obstacle_context(clip_id, record["nurec_t0_us"], normalized, empty),
        "timeline": timeline,
        "cf": cf_rows,
        "ttc": ttc_rows,
    }


def _aggregate_coverage(results, prefix):
    required = sum(result["summary"][f"{prefix}_REQUIRED_QUERY_COUNT"] for result in results)
    objects = sum(result["summary"][f"{prefix}_OBJECT_PRESENT_COUNT"] for result in results)
    empty = sum(result["summary"][f"{prefix}_CONFIRMED_EMPTY_COUNT"] for result in results)
    missing = sum(result["summary"][f"{prefix}_MISSING_COUNT"] for result in results)
    complete = objects + empty
    return {
        f"TOTAL_{prefix}_REQUIRED_QUERY_COUNT": required,
        f"TOTAL_{prefix}_OBJECT_PRESENT_COUNT": objects,
        f"TOTAL_{prefix}_CONFIRMED_EMPTY_COUNT": empty,
        f"TOTAL_{prefix}_MISSING_COUNT": missing,
        f"{prefix}_COMPLETE_OBSERVATION_RATE": complete / required if required else 0.0,
        f"{prefix}_TEMPORAL_OBJECT_EVIDENCE_RATE": objects / required if required else 0.0,
        f"{prefix}_DATA_READY_PILOT": bool(required and missing == 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry-audit-dir", default=None)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    time_path = Path(args.time_alignment_jsonl)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    offsets = {}
    for item in manifest:
        record = load_time_record(time_path, item["clip_id"], int(item.get("physicalai_t0_us", 5_100_000)))
        offsets[item["clip_id"]] = int(record["offset_us"])
    anomaly_keys, anomaly_evidence = load_geometry_anomaly_keys(args.geometry_audit_dir, offsets)
    results, all_cf, all_ttc, all_normalized, timeline_rows = [], [], [], [], []
    for item in manifest:
        result = process_clip(item, time_path, anomaly_keys)
        clip_id = item["clip_id"]
        output = root / clip_id
        output.mkdir(parents=True, exist_ok=True)
        results.append(result)
        all_cf.extend(result["cf"])
        all_ttc.extend(result["ttc"])
        all_normalized.extend(result["normalized"])
        timeline_rows.append({"clip_id": clip_id, **result["timeline"]})
        _jsonl(output / "normalized_obstacle.jsonl", result["normalized"])
        _dump(output / "normalized_context.json", result["context"])
        _dump(output / "observation_timeline_provenance.json", result["timeline"])
        _dump(output / "label_set_timestamp_semantics.json", {
            "clip_id": clip_id,
            "label_set_timestamp_status": result["summary"]["label_set_timestamp_status"],
            "label_set_empty_semantics_status": result["summary"]["label_set_empty_semantics_status"],
            "empty_observation_attestation_status": result["summary"]["empty_observation_attestation_status"],
        })
        _csv(output / "cf_observation_queries.csv", result["cf"])
        _csv(output / "ttc_observation_queries.csv", result["ttc"])
        _dump(output / "cf_observation_coverage_summary.json", result["summary"])
        _dump(output / "ttc_observation_coverage_summary.json", result["summary"])
    cf_aggregate = _aggregate_coverage(results, "CF")
    ttc_aggregate = _aggregate_coverage(results, "TTC")
    cf_ready = cf_aggregate["CF_DATA_READY_PILOT"]
    ttc_ready = ttc_aggregate["TTC_DATA_READY_PILOT"]
    blockers = []
    if not cf_ready or not ttc_ready:
        blockers.append("EXPLICIT_EMPTY_OBSERVATION_ATTESTATION_UNAVAILABLE_OR_QUERY_MISSING")
    _jsonl(root / "normalized_obstacles.jsonl", all_normalized)
    _csv(root / "cf_observation_queries.csv", all_cf)
    _csv(root / "ttc_observation_queries.csv", all_ttc)
    _csv(root / "cf_ttc_readiness_per_clip.csv", [result["summary"] for result in results])
    _dump(root / "normalized_obstacle_contract.json", {
        "authoritative_obstacle_source": "sequence_tracks.json",
        "sequence_tracks_pose_frame": "NCORE_LOCAL_WORLD",
        "sequence_tracks_pose_frame_verified": True,
        "obstacle_transform_status": "VERIFIED",
        "normalized_obstacle_status": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "obstacle_row_quality_status": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "obstacle_geometry_block": "CLOSED",
        "retained_anomaly_count": sum(result["summary"]["normalized_retained_anomaly_count"] for result in results),
        "geometry_anomaly_evidence": anomaly_evidence,
        "no_transform_applied": True,
        "coordinate_alignment_verified": False,
        "per_clip_offset_rederived": False,
    })
    _dump(root / "observation_timeline_provenance.json", {
        "clips": timeline_rows,
        "source_policy": "LOCAL_AUTOMATED_INSPECTION",
        "authoritative_label_set_timestamp_grid_found": False,
        "empty_frames_inferred_from_absence": False,
    })
    _dump(root / "label_set_timestamp_semantics.json", {
        "status": "UNRESOLVED",
        "label_set_timestamp_status": "OBJECT_TIMESTAMP_ONLY",
        "label_set_empty_semantics_status": "UNRESOLVED",
        "empty_observation_attestation_status": "NOT_AVAILABLE",
        "evidence_policy": "camera/pose timestamps and cadence are not label-set attestation",
        "clips": [{"clip_id": result["summary"]["clip_id"], "object_timestamp_count": result["timeline"]["object_timestamp_count"]} for result in results],
    })
    _dump(root / "cf_observation_coverage_summary.json", {
        "pilot_clip_count": len(results),
        **cf_aggregate,
        "CF_DATA_READY": cf_ready,
        "CF_REQUIRED_QUERY_COUNT": cf_aggregate["TOTAL_CF_REQUIRED_QUERY_COUNT"],
        "CF_OBJECT_PRESENT_COUNT": cf_aggregate["TOTAL_CF_OBJECT_PRESENT_COUNT"],
        "CF_CONFIRMED_EMPTY_COUNT": cf_aggregate["TOTAL_CF_CONFIRMED_EMPTY_COUNT"],
        "CF_MISSING_COUNT": cf_aggregate["TOTAL_CF_MISSING_COUNT"],
    })
    _dump(root / "ttc_observation_coverage_summary.json", {
        "pilot_clip_count": len(results),
        **ttc_aggregate,
        "TTC_DATA_READY": ttc_ready,
        "TTC_REQUIRED_QUERY_COUNT": ttc_aggregate["TOTAL_TTC_REQUIRED_QUERY_COUNT"],
        "TTC_OBJECT_PRESENT_COUNT": ttc_aggregate["TOTAL_TTC_OBJECT_PRESENT_COUNT"],
        "TTC_CONFIRMED_EMPTY_COUNT": ttc_aggregate["TOTAL_TTC_CONFIRMED_EMPTY_COUNT"],
        "TTC_MISSING_COUNT": ttc_aggregate["TOTAL_TTC_MISSING_COUNT"],
    })
    _dump(root / "cf_ttc_observation_contract.json", {
        "cf": {
            "future_pose_count": 40,
            "includes_t0": True,
            "frequency_hz": 10.0,
            "horizon_s": 4.0,
            "tolerance_us": CF_TOLERANCE_US,
        },
        "ttc": {
            "horizon_s": 1.0,
            "step_s": 0.2,
            "tolerance_us": TTC_TOLERANCE_US,
            "query_builder": "tools.epdms.observation_contract.build_ttc_projection_timestamps",
        },
        "strict_states": ["OBJECTS_PRESENT", "CONFIRMED_EMPTY", "MISSING"],
        "strict_ready_rule": "all required queries must be OBJECTS_PRESENT or CONFIRMED_EMPTY",
        "no_inferred_empty": True,
        "cf_data_ready_full_300": "NOT_EVALUATED",
        "ttc_data_ready_full_300": "NOT_EVALUATED",
    })
    final = {
        "BRANCH_SCOPE": "feat/obstacle-normalization-cf-ttc-readiness",
        "OBSTACLE_GEOMETRY_BLOCK": "CLOSED",
        "AUTHORITATIVE_OBSTACLE_SOURCE": "sequence_tracks.json",
        "SEQUENCE_TRACKS_POSE_FRAME": "NCORE_LOCAL_WORLD",
        "SEQUENCE_TRACKS_POSE_FRAME_VERIFIED": True,
        "OBSTACLE_TRANSFORM_STATUS": "VERIFIED",
        "NORMALIZED_OBSTACLE_STATUS": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "OBSTACLE_ROW_QUALITY_STATUS": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "OBSERVATION_TIMELINE_SOURCE": "LOCAL_OBJECT_ROWS_ONLY_NO_AUTHORITATIVE_FRAME_ATTESTATION",
        "LABEL_SET_TIMESTAMP_STATUS": "OBJECT_TIMESTAMP_ONLY",
        "LABEL_SET_EMPTY_SEMANTICS_STATUS": "UNRESOLVED",
        "EMPTY_OBSERVATION_ATTESTATION_STATUS": "NOT_AVAILABLE",
        "PILOT_CLIP_COUNT": len(results),
        **cf_aggregate,
        **ttc_aggregate,
        "CF_DATA_READY_PILOT": cf_ready,
        "TTC_DATA_READY_PILOT": ttc_ready,
        "CF_DATA_READY_FULL_300": "NOT_EVALUATED",
        "TTC_DATA_READY_FULL_300": "NOT_EVALUATED",
        "CF_PROXY_LABEL_SET_READY": cf_ready,
        "TTC_PROXY_LABEL_SET_READY": ttc_ready,
        "PHYSICAL_WORLD_OBSTACLE_COMPLETENESS": "NOT_CLAIMED",
        "PER_CLIP_OFFSET_REDERIVED": False,
        "RETAINED_GEOMETRY_ANOMALY_COUNT": sum(result["summary"]["normalized_retained_anomaly_count"] for result in results),
        "REMAINING_BLOCKERS": blockers,
        "RECOMMENDED_NEXT_STEP": "expand the exact same readiness audit to all 300 clips before 4800-condition scoring" if cf_ready and ttc_ready else "obtain explicit empty-observation attestation or keep strict CF/TTC blocked",
    }
    _dump(root / "cf_ttc_readiness_final_summary.json", final)


if __name__ == "__main__":
    main()
