"""Evidence-only NuRec time-domain forensics.

This module inventories timestamp representations and evaluates only explicit
metadata or semantically identified frame correspondences. Numeric proximity,
range overlap, and fitted offsets are deliberately never verification proofs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tools.data_prep.nurec import (
    _frame_path_values,
    _load_json,
    _nested_paths,
    _path_value,
    _timestamp_summary,
    _timestamp_value_valid,
    inspect_json_like,
    inspect_parquet,
    parquet_engine,
)

UNIT_EXPLICIT = "UNIT_EXPLICIT"
UNIT_AMBIGUOUS = "UNIT_AMBIGUOUS"
ALLOWED_STATUSES = {
    "ALIGNED_DIRECT",
    "ALIGNED_BY_EXPLICIT_METADATA",
    "UNRESOLVED",
    "CONFLICTING_TIME_ORIGIN",
    "MISSING_TIME_METADATA",
    "UNIT_AMBIGUOUS",
}


def _walk_values(value: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_values(child, child_path)
    elif isinstance(value, list):
        # Keep numeric timestamp arrays as one field. Recurse only into
        # mapping elements so forensic output stays one row per source field,
        # rather than one row per timestamp value.
        for child in value[:5000]:
            if isinstance(child, Mapping):
                yield from _walk_values(child, f"{path}[]")


def _looks_time(path: str) -> bool:
    normalized = path.lower()
    leaf = normalized.rsplit(".", 1)[-1].split("[")[0]
    return bool(
        re.search(r"(?:timestamp|t0(?:_|$)|epoch|origin|offset|frame_(?:timestamp|id|index)|sequence_id)", normalized, re.I)
        or re.search(r"(?:^|_)(?:start|end)_(?:timestamp|time|micros|us|millis|ms|nanos|ns)$", leaf, re.I)
    )


def _numeric_values(value: Any) -> List[Any]:
    if isinstance(value, list):
        result: List[Any] = []
        for item in value:
            result.extend(_numeric_values(item))
        return result
    return [value]


def unit_evidence(field: str, declared_unit: Optional[str] = None) -> Dict[str, Any]:
    """Classify units only from declarations or explicit field names."""
    if declared_unit:
        return {"unit": str(declared_unit), "status": UNIT_EXPLICIT, "basis": "schema_or_metadata_declaration"}
    lower = field.lower()
    if re.search(r"(?:micros|microseconds|_us)(?:$|[.\[\]])", lower):
        return {"unit": "microseconds", "status": UNIT_EXPLICIT, "basis": "field_name"}
    if re.search(r"(?:millis|milliseconds|_ms)(?:$|[.\[\]])", lower):
        return {"unit": "milliseconds", "status": UNIT_EXPLICIT, "basis": "field_name"}
    if re.search(r"(?:nanos|nanoseconds|_ns)(?:$|[.\[\]])", lower):
        return {"unit": "nanoseconds", "status": UNIT_EXPLICIT, "basis": "field_name"}
    if re.search(r"(?:seconds|_s)(?:$|[.\[\]])", lower):
        return {"unit": "seconds", "status": UNIT_EXPLICIT, "basis": "field_name"}
    return {"unit": None, "status": UNIT_AMBIGUOUS, "basis": "no_declared_unit_or_explicit_suffix"}


def summarize_values(values: Sequence[Any]) -> Dict[str, Any]:
    summary = _timestamp_summary(values)
    valid = sorted({int(float(value)) for value in values if _timestamp_value_valid(value)})
    return {
        "count": len(values),
        "valid_count": summary["timestamp_valid_count"],
        "invalid_count": summary["timestamp_invalid_count"],
        "min": summary["min"],
        "max": summary["max"],
        "unique_count": summary["unique_count"],
        "median_step": summary["median_dt"],
        "first_5": valid[:5],
        "last_5": valid[-5:],
    }


def _summarize_timeline_values(values: Sequence[Any]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))]
    unique = sorted(set(numeric))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return {"count": len(values), "valid_count": len(numeric), "invalid_count": len(values) - len(numeric), "min": min(numeric) if numeric else None, "max": max(numeric) if numeric else None, "unique_count": len(unique), "median_step": median(diffs) if diffs else None, "first_5": numeric[:5], "last_5": numeric[-5:]}


def correspondence_status(pairs: Sequence[Mapping[str, Any]], explicit_identity: bool, min_pairs: int = 2, identity_source: Optional[str] = None) -> Dict[str, Any]:
    """Verify pairs only when an independent shared event/frame identity exists."""
    offsets = [int(pair["timestamp_b"]) - int(pair["timestamp_a"]) for pair in pairs]
    unique = sorted(set(offsets))
    residual = max((abs(value - median(offsets)) for value in offsets), default=None)
    conflict = bool(pairs and len(unique) > 1)
    verified = bool(explicit_identity and len(pairs) >= min_pairs and len(unique) == 1)
    return {
        "mapping_type": "semantic_frame_correspondence" if verified else None,
        "scale": 1.0 if verified else None,
        "offset": offsets[0] if verified else None,
        "pair_count": len(pairs),
        "offset_min": min(offsets) if offsets else None,
        "offset_max": max(offsets) if offsets else None,
        "offset_unique_count": len(unique),
        "residual": residual,
        "residual_us": residual,
        "identity_source": identity_source,
        "verified": verified,
        "status": "CONFLICTING_TIME_ORIGIN" if conflict else "MAPPING_EXPLICIT" if verified else "UNRESOLVED",
        "failure_reason": "CONFLICTING_OFFSETS" if conflict else None if verified else "INSUFFICIENT_SEMANTIC_CORRESPONDENCE",
    }


def resolve_time_alignment_evidence(
    prediction_timestamp_evidence: Mapping[str, Any],
    gt_timestamp_evidence: Mapping[str, Any],
    nurec_clock_evidence: Mapping[str, Any],
    explicit_origin_metadata: Optional[Mapping[str, Any]] = None,
    semantic_pairs: Optional[Sequence[Mapping[str, Any]]] = None,
    unit_evidence_records: Optional[Sequence[Mapping[str, Any]]] = None,
    input_integrity: Optional[Mapping[str, Any]] = None,
    ranges: Optional[Mapping[str, Tuple[Optional[int], Optional[int]]]] = None,
) -> Dict[str, Any]:
    """Resolve only explicit clock/origin evidence; never infer from ranges."""
    metadata = dict(explicit_origin_metadata or {})
    integrity = dict(input_integrity or {})
    if any(int(value or 0) > 0 for value in integrity.values() if isinstance(value, (int, float))):
        return {"status": "UNRESOLVED", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": [], "evidence": [], "blockers": ["INPUT_INTEGRITY_ERRORS"]}
    required_units = [prediction_timestamp_evidence.get("unit_status"), gt_timestamp_evidence.get("unit_status"), nurec_clock_evidence.get("unit_status")]
    if any(value == UNIT_AMBIGUOUS for value in required_units):
        return {"status": "UNIT_AMBIGUOUS", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": [], "evidence": [], "blockers": ["REQUIRED_TIMESTAMP_UNIT_AMBIGUOUS"]}
    if metadata.get("conflicting") is True:
        return {"status": "CONFLICTING_TIME_ORIGIN", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": metadata.get("source", []), "evidence": metadata.get("evidence", []), "blockers": ["CONFLICTING_EXPLICIT_ORIGINS"]}
    if metadata.get("shared_clock_domain") and metadata.get("prediction_clock_domain") == metadata.get("nurec_clock_domain") and metadata.get("prediction_origin_id") == metadata.get("nurec_origin_id") and metadata.get("same_unit") is True:
        return {"status": "ALIGNED_DIRECT", "verified": True, "mapping_type": "shared_clock_domain", "offset_us": None, "scale": 1.0, "source": metadata.get("source", []), "evidence": metadata.get("evidence", []), "blockers": []}
    if metadata.get("mapping_formula") and metadata.get("source") and metadata.get("unit_explicit") is True and metadata.get("origin_explicit") is True:
        offset = metadata.get("offset_us")
        if ranges and offset is not None and metadata.get("mapped_range"):
            mapped = metadata["mapped_range"]
            nurec_range = ranges.get("nurec")
            if nurec_range and (mapped[1] < nurec_range[0] or mapped[0] > nurec_range[1]):
                return {"status": "CONFLICTING_TIME_ORIGIN", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": metadata.get("source", []), "evidence": metadata.get("evidence", []), "blockers": ["EXPLICIT_MAPPING_OUTSIDE_NUREC_RANGE"]}
        return {"status": "ALIGNED_BY_EXPLICIT_METADATA", "verified": True, "mapping_type": metadata.get("mapping_type", "explicit_origin_mapping"), "offset_us": offset, "scale": metadata.get("scale", 1.0), "source": metadata.get("source", []), "evidence": metadata.get("evidence", []), "blockers": []}
    pair_result = correspondence_status(semantic_pairs or [], bool(metadata.get("semantic_identity", False)), int(metadata.get("min_pairs", 2)), metadata.get("identity_source"))
    if pair_result["status"] == "CONFLICTING_TIME_ORIGIN":
        return {**pair_result, "offset_us": None, "source": metadata.get("source", []), "evidence": metadata.get("evidence", []), "blockers": ["CONFLICTING_CORRESPONDENCE_OFFSETS"]}
    if pair_result["verified"]:
        return {"status": "ALIGNED_BY_EXPLICIT_METADATA", "verified": True, "mapping_type": pair_result["mapping_type"], "offset_us": pair_result["offset"], "scale": 1.0, "source": [pair_result["identity_source"]], "evidence": [pair_result], "blockers": []}
    if not prediction_timestamp_evidence.get("available") or not gt_timestamp_evidence.get("available") or not nurec_clock_evidence.get("available"):
        return {"status": "MISSING_TIME_METADATA", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": [], "evidence": [], "blockers": ["MISSING_REQUIRED_TIME_METADATA"]}
    return {"status": "UNRESOLVED", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": [], "evidence": [], "blockers": ["MISSING_EXPLICIT_RELATIVE_TO_GLOBAL_ORIGIN", "MISSING_FRAME_CORRESPONDENCE", "CLOCK_DOMAIN_NOT_PROVEN"]}


def _json_timestamp_records(value: Any, source: str, source_path: str, declared_unit: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Any]] = {}
    schema_by_path: Dict[str, Dict[str, Any]] = {}
    for path, child in _walk_values(value):
        if not path or not _looks_time(path):
            continue
        numbers = _numeric_values(child)
        if not numbers or not any(_timestamp_value_valid(item) for item in numbers):
            continue
        grouped.setdefault(path, []).extend(numbers)
        schema_by_path.setdefault(path, {"path": path, "value_type": type(child).__name__, "sample": numbers[:5], "unit_evidence": unit_evidence(path, declared_unit)})
    records = []
    for path, numbers in grouped.items():
        unit = unit_evidence(path, declared_unit)
        records.append({"source": source, "field": path, "unit_declared": unit["unit"], "unit_status": unit["status"], "unit_basis": unit["basis"], "clock_domain_declared": None, **summarize_values(numbers), "source_path": source_path, "evidence_type": "raw_json_field"})
        schema_by_path[path]["sample"] = numbers[:5]
    return records, list(schema_by_path.values())


def _load_jsonl_clip(path: Path, clip_id: str, source: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if not path.is_file():
        errors.append({"source": source, "line_number": None, "failure_type": "MISSING_SOURCE_FILE", "failure_reason": str(path)})
        return rows, errors
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"source": source, "line_number": line_number, "failure_type": "MALFORMED_JSON", "failure_reason": str(exc)})
                continue
            if not isinstance(row, Mapping):
                errors.append({"source": source, "line_number": line_number, "failure_type": "INVALID_JSON_ROW", "failure_reason": "JSON row is not an object"})
                continue
            if not row.get("clip_id"):
                errors.append({"source": source, "line_number": line_number, "failure_type": "MISSING_CLIP_ID", "failure_reason": "clip_id is absent or empty"})
                continue
            if str(row.get("clip_id")) == clip_id:
                rows.append(row)
    return rows, errors


def _parquet_records(path: Path, source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info = inspect_parquet(path, sample_rows=0)
    records: List[Dict[str, Any]] = []
    if info.get("status") != "OK":
        return records, {"path": str(path), "status": info.get("status"), "fields": []}
    import pandas as pd
    frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
    fields = [str(field) for field in info.get("timestamp_field_candidates", [])]
    for field in fields:
        values = _frame_path_values(frame, field)
        summary = summarize_values(values)
        unit = unit_evidence(field)
        records.append({"source": source, "field": field, "unit_declared": unit["unit"], "unit_status": unit["status"], "unit_basis": unit["basis"], "clock_domain_declared": None, **summary, "source_path": str(path), "evidence_type": "parquet_schema_and_values"})
    return records, {"path": str(path), "status": info.get("status"), "fields": fields, "nested_field_paths": info.get("nested_field_paths", [])}


def _camera_records(clip_dir: Path, declared_unit: Optional[str] = None, unit_basis: Optional[str] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    frames_root = clip_dir / "frames"
    if not frames_root.is_dir():
        return records
    for sensor_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        values: List[int] = []
        for path in sensor_dir.iterdir():
            match = re.search(r"(?:^|/)(\d+)(?:\.[^.]+)?$", path.name)
            if match:
                values.append(int(match.group(1)))
        if values:
            records.append({"source": "camera filenames", "field": sensor_dir.name, "unit_declared": declared_unit, "unit_status": UNIT_EXPLICIT if declared_unit else UNIT_AMBIGUOUS, "unit_basis": unit_basis or "filename_only_no_metadata", "clock_domain_declared": None, **summarize_values(values), "source_path": str(sensor_dir), "evidence_type": "filename_only"})
    return records


TIMELINE_KEYS = {"clean_waypoints", "guided_waypoints", "ego_future_xyz", "expert_future", "future_waypoints", "trajectories", "clean", "guided"}
TIMESTAMP_KEYS = {"t_s", "timestamp_s", "time_s", "timestamp_micros", "t_us", "timestamp_us"}


def _timeline_reports(value: Any, path: str = "") -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in TIMELINE_KEYS and isinstance(child, list):
                timestamps: List[Any] = []
                for waypoint in child:
                    if isinstance(waypoint, Mapping):
                        for timestamp_key in TIMESTAMP_KEYS:
                            if timestamp_key in waypoint:
                                timestamps.append(waypoint[timestamp_key])
                                break
                if timestamps:
                    field = next((key for key in TIMESTAMP_KEYS if any(isinstance(item, Mapping) and key in item for item in child)), None)
                    reports.append({"trajectory_path": child_path, "waypoint_count": len(child), "timestamp_field": field, "timestamp_kind": "absolute" if field in {"timestamp_micros", "t_us", "timestamp_us"} else "relative", "timestamp_count": len(timestamps), **_summarize_timeline_values(timestamps), "includes_t0_candidate": any(float(item) == 0 for item in timestamps if isinstance(item, (int, float)) and not isinstance(item, bool)), "unit_status": unit_evidence(field or "")["status"]})
                else:
                    reports.append({"trajectory_path": child_path, "waypoint_count": len(child), "timestamp_field": None, "timestamp_kind": None, "timestamp_count": 0, "first": None, "last": None, "median_step": None, "includes_t0_candidate": None, "unit_status": "TIMESTAMP_IMPLICIT_BY_PIPELINE"})
            reports.extend(_timeline_reports(child, child_path))
    elif isinstance(value, list):
        for child in value[:5000]:
            reports.extend(_timeline_reports(child, f"{path}[]"))
    return reports


def _explicit_metadata_candidates(value: Any, path: str = "") -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if re.search(r"origin|offset|global|relative|clock|alignment|mapping", str(key), re.I):
                candidates.append({"path": child_path, "raw_value": child, "semantic_interpretation": "CANDIDATE_METADATA_ONLY", "verification_usability": "requires_explicit_relation_to_prediction_or_GT"})
            candidates.extend(_explicit_metadata_candidates(child, child_path))
    elif isinstance(value, list):
        for child in value[:5000]:
            candidates.extend(_explicit_metadata_candidates(child, f"{path}[]"))
    return candidates


def _unique_timeline_reports(reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for report in reports:
        signature = tuple((key, str(report.get(key))) for key in ("trajectory_path", "waypoint_count", "timestamp_field", "timestamp_kind", "timestamp_count", "first_5", "last_5", "median_step", "unit_status"))
        if signature not in seen:
            seen.add(signature)
            result.append(dict(report))
    return result


def _extract_explicit_resolution(value: Any, source_path: str) -> Dict[str, Any]:
    """Accept only deliberately structured metadata, never generic offsets."""
    if not isinstance(value, Mapping):
        return {"candidates": [], "metadata": {}}
    candidates = _explicit_metadata_candidates(value)
    metadata: Dict[str, Any] = {}
    alignment = value.get("time_alignment") or value.get("clock_alignment")
    if isinstance(alignment, Mapping):
        metadata.update({"source": [f"{source_path}:time_alignment"], "evidence": [dict(alignment)]})
        metadata.update({key: alignment.get(key) for key in ("shared_clock_domain", "prediction_clock_domain", "nurec_clock_domain", "prediction_origin_id", "nurec_origin_id", "same_unit", "mapping_formula", "offset_us", "scale", "mapping_type", "unit_explicit", "origin_explicit", "semantic_identity", "identity_source", "min_pairs", "conflicting") if key in alignment})
    mapping = value.get("explicit_time_mapping") or value.get("time_mapping")
    if isinstance(mapping, Mapping) and mapping.get("mapping_formula") and mapping.get("source_field"):
        metadata.update({"source": [f"{source_path}:time_mapping"], "evidence": [dict(mapping)], "mapping_formula": mapping.get("mapping_formula"), "source_field": mapping.get("source_field"), "offset_us": mapping.get("offset_us"), "scale": mapping.get("scale", 1.0), "unit_explicit": mapping.get("unit_explicit") is True, "origin_explicit": mapping.get("origin_explicit") is True, "mapping_type": "explicit_origin_mapping"})
    return {"candidates": candidates, "metadata": metadata}


def _declared_sensor_unit(value: Any, path: str = "") -> Tuple[Optional[str], Optional[str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if re.search(r"(?:timestamp|time).*unit|unit.*(?:timestamp|time)", str(key), re.I) and isinstance(child, str):
                normalized = child.lower()
                if "micro" in normalized or normalized in {"us", "µs"}:
                    return "microseconds", child_path
                if "milli" in normalized or normalized == "ms":
                    return "milliseconds", child_path
                if "nano" in normalized or normalized == "ns":
                    return "nanoseconds", child_path
            result = _declared_sensor_unit(child, child_path)
            if result[0]:
                return result
    elif isinstance(value, list):
        for child in value[:5000]:
            result = _declared_sensor_unit(child, f"{path}[]")
            if result[0]:
                return result
    return None, None


def audit_time_alignment(clip_dir: Path, clip_id: str, prediction_jsonl: Path, ground_truth_jsonl: Path, output_dir: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory: List[Dict[str, Any]] = []
    schema: Dict[str, Any] = {"clip_id": clip_id, "sources": {}}

    prediction_rows, prediction_errors = _load_jsonl_clip(prediction_jsonl, clip_id, "prediction")
    gt_rows, gt_errors = _load_jsonl_clip(ground_truth_jsonl, clip_id, "GT")
    prediction_timeline_all = [report for row in prediction_rows for report in _timeline_reports(row)]
    gt_timeline_all = [report for row in gt_rows for report in _timeline_reports(row)]
    prediction_timeline = _unique_timeline_reports(prediction_timeline_all)
    gt_timeline = _unique_timeline_reports(gt_timeline_all)
    schema["prediction_timeline"] = prediction_timeline
    schema["gt_timeline"] = gt_timeline
    for source, rows, source_path in (("prediction", prediction_rows, str(prediction_jsonl)), ("GT", gt_rows, str(ground_truth_jsonl))):
        values: List[Any] = []
        for row in rows:
            if "t0_us" in row:
                values.append(row["t0_us"])
            fields, field_schema = _json_timestamp_records(row, source, source_path)
            fields = [field for field in fields if field.get("field") != "t0_us"]
            inventory.extend(fields)
            schema["sources"].setdefault(source, []).extend(field_schema)
        if not values:
            inventory.append({"source": source, "field": "t0_us", "unit_declared": "microseconds", "unit_status": UNIT_EXPLICIT, "unit_basis": "field_name", "clock_domain_declared": None, **summarize_values([]), "source_path": source_path, "evidence_type": "NOT_FOUND"})
        else:
            inventory.append({"source": source, "field": "t0_us", "unit_declared": "microseconds", "unit_status": UNIT_EXPLICIT, "unit_basis": "field_name", "clock_domain_declared": None, **summarize_values(values), "source_path": source_path, "evidence_type": "prediction_or_gt_record"})
        if rows:
            schema["sources"].setdefault(source, []).append({"path": "t0_us", "value_type": type(values[0]).__name__ if values else "NOT_FOUND", "sample": values[:5], "note": "trajectory timestamp fields are not assumed from waypoint array positions"})

    json_files = {"data_info": "data_info.json", "datasource_summary": "datasource_summary.json", "pose_record": "pose_record.json", "rig_trajectories": "rig_trajectories.json", "sequence_tracks": "sequence_tracks.json"}
    raw_json_values: Dict[str, Any] = {}
    explicit_candidates: List[Dict[str, Any]] = []
    explicit_metadata: Dict[str, Any] = {}
    for source, relative in json_files.items():
        path = clip_dir / relative
        if not path.is_file():
            schema["sources"][source] = {"status": "NOT_FOUND", "path": relative}
            continue
        value = _load_json(path)
        raw_json_values[source] = value
        fields, field_schema = _json_timestamp_records(value, source, str(path))
        inventory.extend(fields)
        schema["sources"][source] = {"status": "OK", "path": str(path), "top_level_keys": list(value.keys()) if isinstance(value, Mapping) else [], "nested_key_paths": _nested_paths(value), "timestamp_fields": field_schema}
        extracted = _extract_explicit_resolution(value, str(path))
        explicit_candidates.extend(extracted["candidates"])
        if extracted["metadata"]:
            explicit_metadata.update(extracted["metadata"])

    parquet_files = {"clip": "clipgt/clip.parquet", "egomotion": "clipgt/egomotion_estimate.parquet", "obstacle": "clipgt/obstacle.parquet", "association": "clipgt/association.parquet", "calibration": "clipgt/calibration_estimate.parquet"}
    for source, relative in parquet_files.items():
        records, report = _parquet_records(clip_dir / relative, source)
        inventory.extend(records)
        schema["sources"][source] = report
    camera_unit, camera_unit_path = _declared_sensor_unit(raw_json_values.get("data_info"))
    inventory.extend(_camera_records(clip_dir, camera_unit, f"data_info.json:{camera_unit_path}" if camera_unit_path else None))

    # Numeric relationships are diagnostic only. No range overlap is promoted
    # to a clock mapping and no offset is inferred from minima.
    prediction_t0 = {"available": bool(prediction_rows and any("t0_us" in row for row in prediction_rows)), "unit": "microseconds", "unit_status": UNIT_EXPLICIT, "condition_count": len(prediction_rows), "timeline_reports": prediction_timeline}
    gt_t0 = {"available": bool(gt_rows and any("t0_us" in row for row in gt_rows)), "unit": "microseconds", "unit_status": UNIT_EXPLICIT, "record_count": len(gt_rows), "timeline_reports": gt_timeline}
    nurec_rows = [row for row in inventory if row["source"] in {"clip", "egomotion", "obstacle"} and row.get("unit_status") == UNIT_EXPLICIT]
    nurec_clock = {"available": bool(nurec_rows), "unit": "microseconds", "unit_status": UNIT_EXPLICIT, "timestamp_fields": sorted({row["field"] for row in nurec_rows})}
    nurec_min = min((row["min"] for row in nurec_rows if row.get("min") is not None), default=None)
    nurec_max = max((row["max"] for row in nurec_rows if row.get("max") is not None), default=None)
    integrity = {"prediction_jsonl_error_count": len(prediction_errors), "gt_jsonl_error_count": len(gt_errors), "raw_metadata_errors": sum(1 for value in schema["sources"].values() if isinstance(value, Mapping) and value.get("status") == "JSON_READ_ERROR"), "parquet_read_errors": sum(1 for value in schema["sources"].values() if isinstance(value, Mapping) and value.get("status") in {"PARQUET_READ_ERROR", "PARQUET_ENGINE_UNAVAILABLE"}), "missing_source_files": [value.get("path") for value in schema["sources"].values() if isinstance(value, Mapping) and value.get("status") == "NOT_FOUND"]}
    explicit_metadata.setdefault("source", [])
    explicit_metadata.setdefault("evidence", [])
    if len({row.get("t0_us") for row in prediction_rows}) > 1 or len({report.get("timestamp_field") for report in prediction_timeline_all}) > 2:
        explicit_metadata["conflicting"] = True
        explicit_metadata.setdefault("evidence", []).append({"reason": "PREDICTION_TIMELINE_CONFLICT", "t0_values": sorted({row.get("t0_us") for row in prediction_rows})})
    mapping = resolve_time_alignment_evidence(prediction_t0, gt_t0, nurec_clock, explicit_metadata, [], None, {"prediction_jsonl_error_count": len(prediction_errors), "gt_jsonl_error_count": len(gt_errors), "raw_metadata_errors": integrity["raw_metadata_errors"], "parquet_read_errors": integrity["parquet_read_errors"]}, {"nurec": (nurec_min, nurec_max)})
    matrix: List[Dict[str, Any]] = []
    for row in inventory:
        status = "UNIT_AMBIGUOUS" if row.get("unit_status") == UNIT_AMBIGUOUS else "NUMERICALLY_SIMILAR_ONLY"
        matrix.append({"source": row["source"], "timestamp_field": row["field"], "unit": row.get("unit_declared"), "unit_status": row.get("unit_status"), "origin_identifier": row.get("clock_domain_declared"), "origin_status": "ORIGIN_AMBIGUOUS", "range_min": row["min"], "range_max": row["max"], "median_step": row["median_step"], "exact_value_overlap_sources": "", "explicit_mapping_source": "", "semantic_identity_source": "", "status": status})
    correspondences = [{"source_a": "prediction", "source_b": "NuRec", "mapping_type": None, "pair_count": 0, "offset_min": None, "offset_max": None, "offset_unique_count": 0, "residual": None, "identity_source": None, "status": "UNRESOLVED", "reason": "MISSING_FRAME_CORRESPONDENCE"}]
    evidence = {
        "clip_id": clip_id,
        "prediction_clock": {"t0_field": "t0_us", "unit": "microseconds", "conditions": len(prediction_rows), "t0_values": sorted({row.get("t0_us") for row in prediction_rows}), "timeline_reports": prediction_timeline, "timestamp_contract_variants": len(prediction_timeline), "timestamp_contract": "FOUND" if any(report.get("timestamp_field") for report in prediction_timeline) else "WAYPOINT_TIMESTAMPS_NOT_PRESENT", "timeline_consistent_across_conditions": len({(report.get("trajectory_path"), report.get("timestamp_field"), report.get("timestamp_kind"), report.get("waypoint_count"), report.get("timestamp_count")) for report in prediction_timeline_all}) <= 2},
        "gt_clock": {"t0_field": "t0_us", "unit": "microseconds", "records": len(gt_rows), "timeline_reports": gt_timeline, "timestamp_contract": "FOUND" if any(report.get("timestamp_field") for report in gt_timeline) else "WAYPOINT_TIMESTAMPS_NOT_PRESENT"},
        "nurec_clock": {"timestamp_fields": nurec_clock["timestamp_fields"], "unit": "microseconds", "sources": ["clip", "egomotion", "obstacle"]},
        "mapping": mapping,
        "blockers": mapping.get("blockers", []),
        "input_integrity": integrity,
        "explicit_origin_candidates": explicit_candidates,
        "numeric_relationships_are_not_verification": True,
    }
    _write_csv(output_dir / "timestamp_inventory.csv", inventory)
    _write_csv(output_dir / "clock_domain_matrix.csv", matrix)
    _write_csv(output_dir / "timestamp_correspondences.csv", correspondences)
    (output_dir / "time_schema_report.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "time_alignment_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "input_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = _t0_provenance_candidates(prediction_rows, gt_rows, explicit_candidates)
    (output_dir / "t0_provenance_candidates.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if repo_root is not None:
        (output_dir / "upstream_time_provenance.md").write_text(_upstream_provenance_report(repo_root), encoding="utf-8")
    report = _markdown_report(evidence, inventory, schema)
    (output_dir / "time_schema_report.md").write_text(report[0], encoding="utf-8")
    (output_dir / "time_alignment_report.md").write_text(report[1], encoding="utf-8")
    return {"evidence": evidence, "inventory": inventory, "schema": schema, "output_dir": str(output_dir)}


def _t0_provenance_candidates(prediction_rows: Sequence[Mapping[str, Any]], gt_rows: Sequence[Mapping[str, Any]], metadata_candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for source, rows in (("prediction", prediction_rows), ("GT", gt_rows)):
        for row in rows:
            for path, value in _walk_values(row):
                if re.search(r"source_frame_id|source_timestamp|sample_token|clip_offset|sequence_offset|frame_index|anchor_frame|history_end|current_frame", path, re.I):
                    candidates.append({"source": source, "path": path, "sample": value if not isinstance(value, (dict, list)) else str(value)[:500], "verification_usability": "candidate_only"})
    candidates.extend({"source": "raw_metadata", **dict(item)} for item in metadata_candidates if re.search(r"origin|offset|global|relative|clock|alignment|mapping", str(item.get("path", "")), re.I))
    return {"prediction_t0_field": "t0_us", "gt_t0_field": "t0_us", "candidates": candidates, "status": "NO_VERIFIED_T0_BRIDGE" if not candidates else "CANDIDATES_REQUIRE_SEMANTIC_LINK"}


def _upstream_provenance_report(repo_root: Path) -> str:
    patterns = ("t0_us", "ego_future_gt", "reasoning_intervention_nurec_selected", "ground_truth", "NuRec sample")
    matches: List[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in {".py", ".md", ".json", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(pattern.lower() in text.lower() for pattern in patterns):
            matches.append(str(path.relative_to(repo_root)))
    matches = sorted(set(matches))
    lines = ["# Upstream t0 provenance search", "", f"Repository: `{repo_root}`", ""]
    if matches:
        lines += ["Candidate files containing t0/GT/prediction-construction references:", ""] + [f"- `{item}`" for item in matches]
        lines += ["", "These are candidates only; this audit did not modify or execute upstream generation code."]
    else:
        lines += ["UPSTREAM_T0_PROVENANCE_NOT_FOUND_IN_REPOSITORY", "", "No repository file matched the upstream t0/prediction/GT construction search terms."]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(evidence: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> Tuple[str, str]:
    schema_lines = ["# NuRec time schema report", "", f"- Clip: `{evidence['clip_id']}`", "", "## Exact timestamp fields"]
    for source, value in schema.get("sources", {}).items():
        fields = value.get("fields", []) if isinstance(value, Mapping) else []
        if isinstance(value, Mapping) and "timestamp_fields" in value:
            fields = [item.get("path") for item in value["timestamp_fields"]]
        schema_lines.append(f"- `{source}`: {', '.join(fields) if fields else 'NOT_FOUND'}")
    schema_lines += ["", "No timestamp unit or origin is inferred from magnitude."]
    mapping = evidence.get("mapping", {})
    report_lines = ["# NuRec time-alignment evidence", "", f"TIME_ALIGNMENT_STATUS = `{mapping.get('status')}`", f"TIME_ALIGNMENT_VERIFIED = `{str(bool(mapping.get('verified'))).lower()}`", "", "## Findings", "", f"1. Prediction t0: `t0_us`, microseconds by explicit field name; timeline contract: `{evidence['prediction_clock'].get('timestamp_contract')}`.", f"2. GT t0: `t0_us`, microseconds by explicit field name; timeline contract: `{evidence['gt_clock'].get('timestamp_contract')}`.", "3. NuRec obstacle: `key.timestamp_micros`, microseconds by explicit field name.", "4. Egomotion: `key.timestamp_micros`, microseconds by explicit field name.", "5. Sensor filename clock is never assigned a unit without metadata.", "6. Explicit clip time origin: not found unless listed in the structured metadata candidates.", f"7. Prediction-to-NuRec mapping: `{mapping.get('mapping_type') or 'not verified'}`.", f"8. Mapping source: `{', '.join(mapping.get('source') or []) or 'none'}`.", "9. Unit: explicit for suffixed microsecond fields; unsuffixed required fields remain ambiguous.", f"10. Verified: `{str(bool(mapping.get('verified'))).lower()}`.", "", "## Exact blockers", ""] + [f"- `{item}`" for item in evidence["blockers"]] + ["", "Numeric range overlap and candidate differences are diagnostic only; no guessed offset or zero-origin correction was created."]
    return "\n".join(schema_lines) + "\n", "\n".join(report_lines) + "\n"
