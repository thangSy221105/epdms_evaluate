"""Audit the 13 full-300 CF/TTC observation gaps without inferring empty frames.

This orchestration layer consumes the existing full-300 audit, inspects the
affected NuRec USDZ containers with HTTP Range requests, downloads only small
component members, and produces an auditable exact-timestamp classification.
It never changes the time mapping, tolerance, query grid, geometry, scorer, or
evaluator.  An empty frame is promoted only when a source explicitly attests a
zero-object label set for the exact query timestamp.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.acquire_full300_data import (
    NUREC_RELEASE,
    discover_usdz_size,
    fetch_member,
    find_member,
    hf_url,
    remote_central_directory,
)
from scripts.audit_cf_ttc_full300_minimal import (
    EXPECTED_CLIP_COUNT,
    _read_jsonl,
    load_confirmed_empty_sidecar,
    run_audit,
)
from scripts.prepare_nurec_obstacles import load_sequence_tracks, normalize_sequence_tracks


EXPECTED_AFFECTED_CLIPS = 13
EXPECTED_CF_MISSING = 257
EXPECTED_TTC_MISSING = 305
CF_TOLERANCE_US = 50_000
TTC_TOLERANCE_US = 100_000
REQUIRED_REMOTE_BASENAMES = {
    "sequence_tracks.json",
    "obstacle.parquet",
    "clip.parquet",
    "association.parquet",
    "data_info.json",
    "datasource_summary.json",
    "metadata.yaml",
    "parsed_config.yaml",
    "pose_record.json",
    "rig_trajectories.json",
}
EXPLICIT_EMPTY_KEYS = {
    "confirmed_empty_timestamps_us",
    "empty_timestamps_us",
    "empty_frame_timestamps_us",
    "zero_object_timestamps_us",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_role(member_name: str) -> str:
    basename = Path(member_name).name.lower()
    if basename == "sequence_tracks.json":
        return "AUTHORITATIVE_OBJECT_TRACKS"
    if basename == "obstacle.parquet":
        return "AUTHORITATIVE_OBJECT_LABEL_TABLE"
    if basename == "clip.parquet":
        return "CLIP_RANGE_METADATA"
    if basename == "association.parquet":
        return "TOPOLOGY_ASSOCIATION"
    if basename == "data_info.json":
        return "SENSOR_FRAME_RANGE_METADATA"
    if basename in {"datasource_summary.json", "metadata.yaml", "parsed_config.yaml"}:
        return "DATASET_METADATA"
    if basename in {"pose_record.json", "rig_trajectories.json"}:
        return "POSE_TIMELINE_METADATA"
    return "OTHER"


def safe_member_path(root: Path, clip_id: str, member_name: str) -> Path:
    parts = Path(member_name).parts
    if any(part in {"", ".", ".."} for part in parts) or Path(member_name).anchor:
        raise ValueError(f"UNSAFE_USDZ_MEMBER_PATH:{member_name}")
    return root / "downloaded_members" / clip_id / Path(*parts)


def extract_explicit_empty_timestamps(value: Any, path: str = "") -> dict[int, str]:
    """Find explicit zero-object assertions, never infer them from missing rows."""
    result: dict[int, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}" if path else str(key)
            if lowered in EXPLICIT_EMPTY_KEYS and isinstance(child, list):
                for timestamp in child:
                    if isinstance(timestamp, int) and not isinstance(timestamp, bool):
                        result[int(timestamp)] = child_path
            if isinstance(child, dict):
                timestamp = child.get("timestamp_us", child.get("timestamp_micros"))
                object_count = child.get("object_count", child.get("num_objects"))
                if isinstance(timestamp, int) and not isinstance(timestamp, bool) and object_count == 0:
                    result[int(timestamp)] = child_path
            result.update(extract_explicit_empty_timestamps(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(extract_explicit_empty_timestamps(child, f"{path}[{index}]"))
    return result


def recursive_ranges(value: Any, path: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return declared sensor frame ranges and generic time ranges."""
    frame_ranges: list[dict[str, Any]] = []
    time_ranges: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(child, dict):
                start = child.get("start-timestamp_us", child.get("start_timestamp_us", child.get("start_micros")))
                end = child.get("end-timestamp_us", child.get("end_timestamp_us", child.get("end_micros")))
                if isinstance(start, int) and isinstance(end, int):
                    item = {"path": child_path, "start_us": int(start), "end_us": int(end)}
                    if "frame" in str(key).lower():
                        frame_ranges.append(item)
                    else:
                        time_ranges.append(item)
            child_frames, child_times = recursive_ranges(child, child_path)
            frame_ranges.extend(child_frames)
            time_ranges.extend(child_times)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_frames, child_times = recursive_ranges(child, f"{path}[{index}]")
            frame_ranges.extend(child_frames)
            time_ranges.extend(child_times)
    return frame_ranges, time_ranges


def parse_json_source(path: Path, deep: bool = True) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not deep:
        return {
            "read_status": "READ",
            "schema_status": "JSON_OBJECT" if isinstance(value, dict) else "JSON_NON_OBJECT",
            "frame_ranges": [],
            "time_ranges": [],
            "explicit_empty_timestamps": {},
        }
    frame_ranges, time_ranges = recursive_ranges(value)
    return {
        "read_status": "READ",
        "schema_status": "JSON_OBJECT" if isinstance(value, dict) else "JSON_NON_OBJECT",
        "frame_ranges": frame_ranges,
        "time_ranges": time_ranges,
        "explicit_empty_timestamps": extract_explicit_empty_timestamps(value),
    }


def parse_parquet_source(path: Path, basename: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    info: dict[str, Any] = {
        "read_status": "READ",
        "schema_status": "PARQUET_READ",
        "row_count": table.num_rows,
        "object_timestamps": {},
        "frame_ranges": [],
        "time_ranges": [],
        "explicit_empty_timestamps": {},
    }
    rows = table.to_pylist()
    if basename == "obstacle.parquet":
        for row in rows:
            key = row.get("key") or {}
            timestamp = key.get("timestamp_micros") if isinstance(key, dict) else None
            if isinstance(timestamp, int):
                info["object_timestamps"][int(timestamp)] = info["object_timestamps"].get(int(timestamp), 0) + 1
    elif basename == "clip.parquet":
        for row in rows:
            key = row.get("key") or {}
            time_range = key.get("time_range") if isinstance(key, dict) else None
            if isinstance(time_range, dict):
                start = time_range.get("start_micros")
                end = time_range.get("end_micros")
                if isinstance(start, int) and isinstance(end, int):
                    info["time_ranges"].append({"path": "key.time_range", "start_us": start, "end_us": end})
    return info


def parse_source(path: Path, member_name: str | None = None) -> dict[str, Any]:
    basename = Path(member_name or path.name).name.lower()
    try:
        if basename.endswith(".parquet"):
            return parse_parquet_source(path, basename)
        if basename.endswith(".json"):
            # data_info contains the only small, structured sensor frame-range
            # evidence needed for this audit.  Other JSON members are still
            # read and provenance-recorded, but avoid recursively walking
            # large pose/track payloads with no empty-label semantics.
            return parse_json_source(path, deep=basename == "data_info.json")
        raw = path.read_text(encoding="utf-8")
        return {"read_status": "READ", "schema_status": "TEXT_READ", "text_size": len(raw)}
    except Exception as exc:
        return {
            "read_status": "READ_ERROR",
            "schema_status": "INVALID",
            "error": f"{type(exc).__name__}:{exc}",
            "frame_ranges": [],
            "time_ranges": [],
            "object_timestamps": {},
            "explicit_empty_timestamps": {},
        }


def load_triage_inputs(previous_root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str], dict[str, Any]]:
    summary = json.loads((previous_root / "full300_readiness_final_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("EXPECTED_CLIP_COUNT") != EXPECTED_CLIP_COUNT
        or summary.get("TOTAL_CF_MISSING_COUNT") != EXPECTED_CF_MISSING
        or summary.get("TOTAL_TTC_MISSING_COUNT") != EXPECTED_TTC_MISSING
    ):
        raise RuntimeError("TRIAGE_INPUT_REGRESSION")
    rows = read_csv(previous_root / "cf_ttc_missing_query_triage.csv")
    metric_counts = defaultdict(int)
    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        metric = str(row.get("metric") or "")
        clip_id = str(row.get("clip_id") or "")
        query_text = row.get("query_timestamp_us")
        if metric not in {"CF", "TTC"} or not clip_id or query_text in {None, ""}:
            continue
        query = int(query_text)
        metric_counts[metric] += 1
        item = dedup.setdefault(
            (clip_id, query),
            {
                "clip_id": clip_id,
                "query_timestamp_us": query,
                "required_by_cf": False,
                "required_by_ttc": False,
                "nearest_object_timestamp_us": None,
                "object_delta_us": None,
                "prior_classifications": set(),
            },
        )
        item["required_by_cf"] |= metric == "CF"
        item["required_by_ttc"] |= metric == "TTC"
        if row.get("nearest_object_timestamp_us") not in {None, ""}:
            nearest = int(row["nearest_object_timestamp_us"])
            delta = abs(nearest - query)
            if item["object_delta_us"] is None or delta < item["object_delta_us"]:
                item["nearest_object_timestamp_us"] = nearest
                item["object_delta_us"] = delta
        item["prior_classifications"].add(str(row.get("classification") or ""))
    if metric_counts["CF"] != EXPECTED_CF_MISSING or metric_counts["TTC"] != EXPECTED_TTC_MISSING:
        raise RuntimeError("TRIAGE_INPUT_REGRESSION")
    affected = sorted({clip_id for clip_id, _ in dedup})
    if len(affected) != EXPECTED_AFFECTED_CLIPS:
        raise RuntimeError("TRIAGE_INPUT_REGRESSION")
    return dedup, affected, summary


def inventory_and_download_members(
    clip_ids: list[str],
    output_root: Path,
    repo_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    inventory: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    completed_report_exists = (output_root / "final_13clip_observation_closure_summary.json").is_file()
    counters = {"downloaded_bytes": 0, "component_download_count": 0, "full_clip_download_count": 0}
    if completed_report_exists:
        try:
            prior = json.loads((output_root / "final_13clip_observation_closure_summary.json").read_text(encoding="utf-8"))
            counters["downloaded_bytes"] = int(prior.get("DOWNLOADED_BYTES") or 0)
            counters["component_download_count"] = int(prior.get("COMPONENT_DOWNLOAD_COUNT") or 0)
            counters["full_clip_download_count"] = int(prior.get("FULL_CLIP_DOWNLOAD_COUNT") or 0)
        except (OSError, ValueError, TypeError):
            pass
    for clip_id in clip_ids:
        archive = f"sample_set/{NUREC_RELEASE}/{clip_id}/{clip_id}.usdz"
        url = hf_url(repo_id, archive)
        try:
            archive_size, _ = discover_usdz_size(url)
            members = remote_central_directory(url, archive_size)
            for member_name, member in sorted(members.items()):
                basename = Path(member_name).name
                role = candidate_role(member_name)
                should_download = basename.lower() in REQUIRED_REMOTE_BASENAMES and basename.lower() != "sequence_tracks.json"
                downloaded = False
                parsed = False
                local_path: Path | None = None
                error = None
                if basename.lower() == "sequence_tracks.json":
                    # The sequence member is already present in the verified
                    # full-300 acquisition root; it is parsed separately as
                    # the local authoritative object source.
                    parsed = True
                if should_download:
                    try:
                        local_path = safe_member_path(output_root, clip_id, member_name)
                        if not local_path.is_file():
                            raw = fetch_member(url, archive_size, member)
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(raw)
                            counters["downloaded_bytes"] += len(raw)
                            counters["component_download_count"] += 1
                            downloaded = True
                        elif not completed_report_exists:
                            # A prior interrupted run of this same output
                            # root already fetched the member. Count it once
                            # in the final accounting without redownloading.
                            counters["downloaded_bytes"] += local_path.stat().st_size
                            counters["component_download_count"] += 1
                            downloaded = True
                        parsed = parse_source(local_path, member_name).get("read_status") == "READ"
                    except Exception as exc:
                        error = f"{type(exc).__name__}:{exc}"
                inventory.append({
                    "clip_id": clip_id,
                    "member_name": member_name,
                    "size_bytes": int(member.get("uncompressed_size") or 0),
                    "candidate_annotation_role": role,
                    "downloaded": downloaded,
                    "parsed": parsed,
                    "archive_size_bytes": archive_size,
                    "error": error,
                })
                if local_path is not None and local_path.is_file():
                    sources.append({
                        "clip_id": clip_id,
                        "source_file": str(local_path),
                        "source_member": member_name,
                        "source_kind": "DOWNLOADED_COMPONENT",
                        "candidate_annotation_role": role,
                    })
            inventory.append({
                "clip_id": clip_id,
                "member_name": "__ARCHIVE__",
                "size_bytes": archive_size,
                "candidate_annotation_role": "USDZ_CONTAINER",
                "downloaded": False,
                "parsed": False,
                "archive_size_bytes": archive_size,
                "error": None,
            })
        except Exception as exc:
            inventory.append({
                "clip_id": clip_id,
                "member_name": "__ARCHIVE__",
                "size_bytes": None,
                "candidate_annotation_role": "USDZ_CONTAINER",
                "downloaded": False,
                "parsed": False,
                "archive_size_bytes": None,
                "error": f"{type(exc).__name__}:{exc}",
            })
    return inventory, sources, counters


def local_sequence_evidence(clip_id: str, path: Path) -> dict[str, Any]:
    try:
        rows, errors, schema = load_sequence_tracks(path)
        if errors:
            return {"read_status": "READ_WITH_ERRORS", "schema_status": "INVALID_ROWS", "errors": errors}
        normalized = normalize_sequence_tracks(clip_id, rows)
        timestamps = [int(row["timestamp_us"]) for row in normalized]
        return {
            "read_status": "READ",
            "schema_status": "VALID_OBJECT_ROWS",
            "object_timestamps": {timestamp: sum(row["timestamp_us"] == timestamp for row in normalized) for timestamp in set(timestamps)},
            "label_source": True,
            "source_semantics": "OBJECT_ROWS_ONLY_NO_EMPTY_ATTESTATION",
            "timestamp_min_us": min(timestamps) if timestamps else None,
            "timestamp_max_us": max(timestamps) if timestamps else None,
            "explicit_empty_timestamps": {},
            "frame_ranges": [],
            "time_ranges": [],
            "error": None,
            "schema": schema,
        }
    except Exception as exc:
        return {
            "read_status": "READ_ERROR",
            "schema_status": "INVALID",
            "object_timestamps": {},
            "label_source": True,
            "source_semantics": "OBJECT_ROWS_ONLY_NO_EMPTY_ATTESTATION",
            "explicit_empty_timestamps": {},
            "frame_ranges": [],
            "time_ranges": [],
            "error": f"{type(exc).__name__}:{exc}",
        }


def merge_clip_evidence(
    clip_id: str,
    sequence_path: Path,
    downloaded_sources: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = {
        "object_timestamps": {},
        "label_ranges": [],
        "explicit_empty_timestamps": {},
        "frame_ranges": [],
        "sensor_sources": [],
        "label_source_errors": [],
        "source_files": [],
    }
    source_rows: list[dict[str, Any]] = []

    sequence_info = local_sequence_evidence(clip_id, sequence_path)
    source_rows.append({
        "clip_id": clip_id,
        "source_file": str(sequence_path),
        "source_kind": "LOCAL_ACQUIRED_COMPONENT",
        "candidate_annotation_role": "AUTHORITATIVE_OBJECT_TRACKS",
        "read_status": sequence_info.get("read_status"),
        "schema_status": sequence_info.get("schema_status"),
        "label_semantics": sequence_info.get("source_semantics"),
        "timestamp_min_us": sequence_info.get("timestamp_min_us"),
        "timestamp_max_us": sequence_info.get("timestamp_max_us"),
        "explicit_empty_count": len(sequence_info.get("explicit_empty_timestamps", {})),
        "error": sequence_info.get("error"),
    })
    if sequence_info.get("read_status") != "READ":
        evidence["label_source_errors"].append(str(sequence_info.get("error") or "SEQUENCE_TRACKS_READ_ERROR"))
    for timestamp, count in sequence_info.get("object_timestamps", {}).items():
        evidence["object_timestamps"][int(timestamp)] = max(int(count), int(evidence["object_timestamps"].get(int(timestamp), 0)))
    if sequence_info.get("timestamp_min_us") is not None:
        evidence["label_ranges"].append({"source_file": str(sequence_path), "min_us": sequence_info["timestamp_min_us"], "max_us": sequence_info["timestamp_max_us"]})
    evidence["source_files"].append(str(sequence_path))

    for source in downloaded_sources:
        path = Path(source["source_file"])
        parsed = parse_source(path, source.get("source_member"))
        basename = path.name.lower()
        object_timestamps = parsed.get("object_timestamps", {})
        if object_timestamps:
            for timestamp, count in object_timestamps.items():
                evidence["object_timestamps"][int(timestamp)] = max(int(count), int(evidence["object_timestamps"].get(int(timestamp), 0)))
            values = sorted(int(value) for value in object_timestamps)
            evidence["label_ranges"].append({"source_file": str(path), "min_us": values[0], "max_us": values[-1]})
        for timestamp, record in parsed.get("explicit_empty_timestamps", {}).items():
            evidence["explicit_empty_timestamps"][int(timestamp)] = f"{path}:{record}"
        if parsed.get("frame_ranges"):
            evidence["frame_ranges"].extend(parsed["frame_ranges"])
            if basename == "data_info.json":
                evidence["sensor_sources"].append(str(path))
        source_rows.append({
            "clip_id": clip_id,
            "source_file": str(path),
            "source_kind": source.get("source_kind"),
            "candidate_annotation_role": source.get("candidate_annotation_role"),
            "read_status": parsed.get("read_status"),
            "schema_status": parsed.get("schema_status"),
            "label_semantics": "OBJECT_ROWS_ONLY_NO_EMPTY_ATTESTATION" if basename == "obstacle.parquet" else ("SENSOR_RANGE_ONLY" if basename == "data_info.json" else "METADATA_ONLY"),
            "timestamp_min_us": min(object_timestamps) if object_timestamps else None,
            "timestamp_max_us": max(object_timestamps) if object_timestamps else None,
            "explicit_empty_count": len(parsed.get("explicit_empty_timestamps", {})),
            "error": parsed.get("error"),
        })
        if parsed.get("read_status") == "READ_ERROR" and basename == "obstacle.parquet":
            evidence["label_source_errors"].append(str(parsed.get("error") or "OBSTACLE_LABEL_READ_ERROR"))
        evidence["source_files"].append(str(path))
    return evidence, source_rows


def classify_gap(item: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    query = int(item["query_timestamp_us"])
    explicit_empty = evidence["explicit_empty_timestamps"].get(query)
    object_count = evidence["object_timestamps"].get(query)
    conflict = explicit_empty is not None and object_count is not None and object_count > 0
    ranges = evidence["label_ranges"]
    minimum = min((int(row["min_us"]) for row in ranges), default=None)
    maximum = max((int(row["max_us"]) for row in ranges), default=None)
    sensor_records = [row for row in evidence["frame_ranges"] if int(row["start_us"]) <= query <= int(row["end_us"])]
    sensor_present = bool(sensor_records)
    sensor_source = ";".join(evidence["sensor_sources"]) if sensor_records else None

    if evidence["label_source_errors"]:
        classification = "ANNOTATION_INPUT_ERROR"
        reason = "authoritative object-label source could not be read or validated"
        source_file = ";".join(evidence["source_files"])
        source_record = ";".join(evidence["label_source_errors"])
    elif conflict:
        classification = "UNRESOLVED"
        reason = "explicit empty attestation conflicts with object evidence at the exact timestamp"
        source_file = explicit_empty
        source_record = "OBJECT_EMPTY_EVIDENCE_CONFLICT"
    elif explicit_empty is not None:
        classification = "CONFIRMED_EMPTY"
        reason = "authoritative exact timestamp carries an explicit zero-object label-set attestation"
        source_file = explicit_empty
        source_record = "exact_zero_object_label_set"
    elif minimum is not None and (query < minimum or query > maximum):
        classification = "OUTSIDE_LABEL_TIMELINE"
        reason = "query lies outside the min/max timestamps of all available authoritative object-label rows"
        source_file = ";".join(sorted({row["source_file"] for row in ranges}))
        source_record = f"label_timestamp_range=[{minimum},{maximum}]"
    else:
        classification = "UNRESOLVED"
        reason = "label/object rows do not provide exact zero-object frame attestation"
        source_file = ";".join(sorted(set(evidence["source_files"])))
        source_record = "object_rows_only_no_empty_attestation"

    return {
        **item,
        "required_by_cf": bool(item["required_by_cf"]),
        "required_by_ttc": bool(item["required_by_ttc"]),
        "sensor_frame_present": sensor_present,
        "sensor_frame_evidence_source": sensor_source,
        "label_frame_attested": explicit_empty is not None,
        "label_object_count": 0 if explicit_empty is not None and not conflict else object_count,
        "classification": classification,
        "source_file": source_file,
        "source_record": source_record,
        "reason": reason,
        "object_empty_evidence_conflict": conflict,
        "label_timeline_min_us": minimum,
        "label_timeline_max_us": maximum,
    }


def load_confirmed_sidecar_for_rerun(repo_root: Path, explicit_path: str | None) -> tuple[dict[str, set[int]], dict[tuple[str, int], str], list[str]]:
    path = Path(explicit_path) if explicit_path else None
    if path is not None and not path.is_absolute():
        path = repo_root / path
    return load_confirmed_empty_sidecar(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    previous_root = Path(args.previous_audit_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dedup, affected, prior_summary = load_triage_inputs(previous_root)

    inventory, downloaded_sources, download_counts = inventory_and_download_members(affected, output_root, args.nurec_repo)
    sources_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in downloaded_sources:
        sources_by_clip[source["clip_id"]].append(source)

    evidence_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for clip_id in affected:
        sequence_path = Path(args.nurec_root) / clip_id / "sequence_tracks.json"
        evidence, clip_sources = merge_clip_evidence(clip_id, sequence_path, sources_by_clip[clip_id])
        source_rows.extend(clip_sources)
        clip_items = [item for item in dedup.values() if item["clip_id"] == clip_id]
        classified = [classify_gap(item, evidence) for item in sorted(clip_items, key=lambda row: row["query_timestamp_us"])]
        evidence_rows.extend(classified)
        counts = defaultdict(int)
        for row in classified:
            counts[row["classification"]] += 1
        cf_before = sum(row["required_by_cf"] for row in classified)
        ttc_before = sum(row["required_by_ttc"] for row in classified)
        cf_after = sum(row["required_by_cf"] and row["classification"] not in {"CONFIRMED_EMPTY"} for row in classified)
        ttc_after = sum(row["required_by_ttc"] and row["classification"] not in {"CONFIRMED_EMPTY"} for row in classified)
        clip_rows.append({
            "clip_id": clip_id,
            "cf_missing_before": cf_before,
            "ttc_missing_before": ttc_before,
            "unique_missing_before": len(classified),
            "confirmed_empty_count": counts["CONFIRMED_EMPTY"],
            "object_label_gap_count": counts["OBJECT_LABEL_GAP"],
            "outside_label_timeline_count": counts["OUTSIDE_LABEL_TIMELINE"],
            "annotation_input_error_count": counts["ANNOTATION_INPUT_ERROR"],
            "unresolved_count": counts["UNRESOLVED"],
            "cf_missing_after": cf_after,
            "ttc_missing_after": ttc_after,
            "cf_ready_after": cf_after == 0,
            "ttc_ready_after": ttc_after == 0,
        })

    confirmed = [row for row in evidence_rows if row["classification"] == "CONFIRMED_EMPTY"]
    conflicts = [row for row in evidence_rows if row["object_empty_evidence_conflict"]]
    unresolved = [row for row in evidence_rows if row["classification"] not in {"CONFIRMED_EMPTY"}]
    confirmed_by_clip: dict[str, list[int]] = defaultdict(list)
    confirmed_sources: dict[tuple[str, int], str] = {}
    for row in confirmed:
        confirmed_by_clip[row["clip_id"]].append(int(row["query_timestamp_us"]))
        confirmed_sources[(row["clip_id"], int(row["query_timestamp_us"]))] = str(row["source_file"])

    sidecar_path = Path(args.repo_root) / "configs" / "nurec_confirmed_empty_full300.jsonl"
    if confirmed_by_clip:
        sidecar_lines = []
        for clip_id in sorted(confirmed_by_clip):
            sidecar_lines.append(json.dumps({
                "clip_id": clip_id,
                "confirmed_empty_timestamps_us": sorted(set(confirmed_by_clip[clip_id])),
                "source": ";".join(sorted({confirmed_sources[(clip_id, timestamp)] for timestamp in confirmed_by_clip[clip_id]})),
                "attestation_type": "AUTHORITATIVE_FRAME_LEVEL_ZERO_OBJECT_LABEL_SET",
                "verified": True,
            }, sort_keys=True))
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text("\n".join(sidecar_lines) + "\n", encoding="utf-8")
    confirmed_sidecar, confirmed_sidecar_sources, sidecar_errors = load_confirmed_sidecar_for_rerun(Path(args.repo_root), str(sidecar_path) if confirmed_by_clip else None)
    manifest = _read_jsonl(Path(args.manifest))
    readiness = run_audit(
        manifest,
        output_root,
        manifest_output=None,
        expected_clip_count=EXPECTED_CLIP_COUNT,
        confirmed_empty_by_clip=confirmed_sidecar,
        confirmed_empty_sources=confirmed_sidecar_sources,
        confirmed_empty_sidecar_errors=sidecar_errors,
    )

    before_cf = EXPECTED_CF_MISSING
    before_ttc = EXPECTED_TTC_MISSING
    unique_cf = sum(row["required_by_cf"] for row in evidence_rows)
    unique_ttc = sum(row["required_by_ttc"] for row in evidence_rows)
    shared = sum(row["required_by_cf"] and row["required_by_ttc"] for row in evidence_rows)
    classification_counts = defaultdict(int)
    for row in evidence_rows:
        classification_counts[row["classification"]] += 1
    summary = {
        "AFFECTED_CLIP_COUNT": len(affected),
        "CF_MISSING_BEFORE": before_cf,
        "TTC_MISSING_BEFORE": before_ttc,
        "CF_ONLY_MISSING_COUNT": unique_cf - shared,
        "TTC_ONLY_MISSING_COUNT": unique_ttc - shared,
        "CF_TTC_SHARED_MISSING_COUNT": shared,
        "UNIQUE_MISSING_TIMESTAMP_COUNT": len(evidence_rows),
        "AUTHORITATIVE_ANNOTATION_SOURCE_COUNT": len({(row["clip_id"], row["source_file"]) for row in source_rows if row.get("candidate_annotation_role") in {"AUTHORITATIVE_OBJECT_TRACKS", "AUTHORITATIVE_OBJECT_LABEL_TABLE"}}),
        "CONFIRMED_EMPTY_UNIQUE_TIMESTAMP_COUNT": classification_counts["CONFIRMED_EMPTY"],
        "OBJECT_LABEL_GAP_COUNT": classification_counts["OBJECT_LABEL_GAP"],
        "OUTSIDE_LABEL_TIMELINE_COUNT": classification_counts["OUTSIDE_LABEL_TIMELINE"],
        "ANNOTATION_INPUT_ERROR_COUNT": classification_counts["ANNOTATION_INPUT_ERROR"],
        "UNRESOLVED_COUNT": classification_counts["UNRESOLVED"],
        "OBJECT_EMPTY_EVIDENCE_CONFLICT_COUNT": len(conflicts),
        "FULL300_EVALUATED_CLIP_COUNT": readiness.get("FULL300_EVALUATED_CLIP_COUNT"),
        "TOTAL_CF_REQUIRED_QUERY_COUNT": readiness.get("TOTAL_CF_REQUIRED_QUERY_COUNT"),
        "TOTAL_CF_OBJECT_PRESENT_COUNT": readiness.get("TOTAL_CF_OBJECT_PRESENT_COUNT"),
        "TOTAL_CF_CONFIRMED_EMPTY_COUNT": readiness.get("TOTAL_CF_CONFIRMED_EMPTY_COUNT"),
        "TOTAL_CF_MISSING_COUNT": readiness.get("TOTAL_CF_MISSING_COUNT"),
        "CF_DATA_READY_FULL_300": readiness.get("CF_DATA_READY_FULL_300"),
        "TOTAL_TTC_REQUIRED_QUERY_COUNT": readiness.get("TOTAL_TTC_REQUIRED_QUERY_COUNT"),
        "TOTAL_TTC_OBJECT_PRESENT_COUNT": readiness.get("TOTAL_TTC_OBJECT_PRESENT_COUNT"),
        "TOTAL_TTC_CONFIRMED_EMPTY_COUNT": readiness.get("TOTAL_TTC_CONFIRMED_EMPTY_COUNT"),
        "TOTAL_TTC_MISSING_COUNT": readiness.get("TOTAL_TTC_MISSING_COUNT"),
        "TTC_DATA_READY_FULL_300": readiness.get("TTC_DATA_READY_FULL_300"),
        "LABEL_SET_EMPTY_SEMANTICS_STATUS": "VERIFIED_FOR_EXACT_SIDECAR" if confirmed else "UNRESOLVED_NO_AUTHORITATIVE_EMPTY_ATTESTATION",
        "PHYSICAL_WORLD_OBSTACLE_COMPLETENESS": "NOT_CLAIMED",
        "DOWNLOADED_BYTES": download_counts["downloaded_bytes"],
        "COMPONENT_DOWNLOAD_COUNT": download_counts["component_download_count"],
        "FULL_CLIP_DOWNLOAD_COUNT": download_counts["full_clip_download_count"],
        "TIME_MAPPING_FULL300_STATUS": "VERIFIED",
        "OBSTACLE_GEOMETRY_BLOCK": "CLOSED",
        "SCORER_EVALUATOR_CHANGED": False,
        "REMAINING_BLOCKERS": [
            "UNRESOLVED_OR_OUTSIDE_LABEL_TIMELINE_QUERY" if unresolved else "",
            "LABEL_SET_EMPTY_SEMANTICS_UNRESOLVED" if not confirmed else "",
        ],
        "prior_summary_source": str(previous_root / "full300_readiness_final_summary.json"),
        "readiness_after_empty_attestation": readiness,
    }
    summary["REMAINING_BLOCKERS"] = [value for value in summary["REMAINING_BLOCKERS"] if value]

    write_csv(output_root / "affected_13clip_manifest.csv", clip_rows)
    write_json(output_root / "missing_query_dedup_summary.json", {
        "CF_MISSING_QUERY_COUNT_BEFORE": before_cf,
        "TTC_MISSING_QUERY_COUNT_BEFORE": before_ttc,
        "CF_ONLY_MISSING_COUNT": summary["CF_ONLY_MISSING_COUNT"],
        "TTC_ONLY_MISSING_COUNT": summary["TTC_ONLY_MISSING_COUNT"],
        "CF_TTC_SHARED_MISSING_COUNT": summary["CF_TTC_SHARED_MISSING_COUNT"],
        "UNIQUE_MISSING_TIMESTAMP_COUNT": summary["UNIQUE_MISSING_TIMESTAMP_COUNT"],
    })
    write_csv(output_root / "nurec_13clip_container_member_inventory.csv", inventory)
    write_csv(output_root / "annotation_source_inventory.csv", source_rows)
    write_csv(output_root / "13clip_observation_gap_evidence.csv", evidence_rows)
    write_csv(output_root / "13clip_observation_gap_summary.csv", clip_rows)
    confirmed_evidence_rows = [
        {
            "clip_id": row["clip_id"],
            "timestamp_us": row["query_timestamp_us"],
            "source_file": row["source_file"],
            "source_record": row["source_record"],
            "label_frame_present": True,
            "object_count": 0,
            "attestation_type": "AUTHORITATIVE_FRAME_LEVEL_ZERO_OBJECT_LABEL_SET",
            "evidence_verified": True,
        }
        for row in confirmed
    ]
    write_csv(output_root / "confirmed_empty_evidence.csv", confirmed_evidence_rows, fields=[
        "clip_id", "timestamp_us", "source_file", "source_record", "label_frame_present", "object_count", "attestation_type", "evidence_verified",
    ])
    write_csv(output_root / "observation_evidence_conflicts.csv", conflicts)
    write_csv(output_root / "remaining_unresolved_queries.csv", unresolved)
    write_json(output_root / "cf_ttc_readiness_after_empty_attestation.json", readiness)
    write_json(output_root / "final_13clip_observation_closure_summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-audit-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5"))
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\cf_ttc_13clip_empty_attestation_v1"))
    parser.add_argument("--nurec-root", type=Path, default=Path(r"D:\300_clip_nurec\00_raw\nurec_full300"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/nurec_cf_ttc_full300_manifest.jsonl"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nurec-repo", default="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec")
    return parser


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "readiness_after_empty_attestation"}, indent=2, ensure_ascii=False, sort_keys=True))
