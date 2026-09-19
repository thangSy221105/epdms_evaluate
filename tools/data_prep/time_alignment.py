"""Evidence-only NuRec time-domain forensics.

This module inventories timestamp representations and evaluates only explicit
metadata or semantically identified frame correspondences. Numeric proximity,
range overlap, and fitted offsets are deliberately never verification proofs.
"""

from __future__ import annotations

import json
import hashlib
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


def _parquet_identity_candidates(path: Path, source: str) -> List[Dict[str, Any]]:
    """Read row-local identities and timestamps for forensic joins only."""
    info = inspect_parquet(path, sample_rows=0)
    if info.get("status") != "OK":
        return []
    import pandas as pd
    try:
        frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
    except Exception:
        return []
    result: List[Dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        record = row.to_dict()
        source_path = f"{path}#row={row_index}"
        for candidate in extract_identity_candidates(record, source, source_path):
            result.append(candidate)
    return result


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
IDENTITY_KEYS = {
    "frame_id", "frame_index", "sample_id", "sample_token", "token", "sequence_id",
    "pose_id", "source_frame_id", "anchor_frame", "current_frame", "current_idx",
    "anchor_idx", "future_idx", "sensor_frame_id", "clip_frame_id",
}


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _identity_type(path: str) -> Optional[str]:
    leaf = path.rsplit(".", 1)[-1].split("[")[0].lower()
    if leaf in IDENTITY_KEYS:
        return leaf
    return None


def extract_identity_candidates(value: Any, source: str, source_path: str, path: str = "") -> List[Dict[str, Any]]:
    """Extract identity candidates without treating equal numbers as proof.

    ``semantic_explicit`` is true only for a frame/sample identity that is
    colocated with a timestamp in the same record or is explicitly marked by
    the caller.  A repeated value alone is never sufficient.
    """
    candidates: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        local_timestamp = None
        local_timestamp_path = None
        for key, child in value.items():
            unit = unit_evidence(str(key))
            if unit["status"] == UNIT_EXPLICIT and unit["unit"] == "microseconds" and _scalar(child) and re.search(r"timestamp|frame_timestamp|sensor_timestamp|pose_timestamp", str(key), re.I):
                local_timestamp = child
                local_timestamp_path = f"{path}.{key}" if path else str(key)
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            identity_type = _identity_type(child_path)
            if identity_type and _scalar(child):
                candidates.append({
                    "source": source,
                    "source_path": source_path,
                    "path": child_path,
                    "identity_type": identity_type,
                    "identity_value": str(child),
                    "timestamp_us": int(float(local_timestamp)) if local_timestamp is not None and _timestamp_value_valid(local_timestamp) else None,
                    "timestamp_path": local_timestamp_path,
                    "semantic_explicit": bool(local_timestamp_path),
                    "verification_usability": "candidate_with_same_record_timestamp" if local_timestamp_path else "candidate_only",
                })
            candidates.extend(extract_identity_candidates(child, source, source_path, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value[:5000]):
            if isinstance(child, Mapping):
                candidates.extend(extract_identity_candidates(child, source, source_path, f"{path}[{index}]"))
    return candidates


def extract_semantic_correspondences(
    prediction_candidates: Sequence[Mapping[str, Any]],
    gt_candidates: Sequence[Mapping[str, Any]],
    nurec_candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Join prediction/GT/NuRec identities only when semantics are explicit.

    The caller supplies extracted candidates so this function is also usable
    by deterministic unit tests.  Matching identity text without an explicit
    same-record timestamp or mapping declaration is deliberately rejected.
    """
    by_identity: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for candidate in nurec_candidates:
        key = (str(candidate.get("identity_type")), str(candidate.get("identity_value")))
        if candidate.get("timestamp_us") is not None:
            by_identity.setdefault(key, []).append(candidate)
    pairs: List[Dict[str, Any]] = []
    for pred in prediction_candidates:
        key = (str(pred.get("identity_type")), str(pred.get("identity_value")))
        matches = by_identity.get(key, [])
        if not matches or not pred.get("semantic_explicit"):
            continue
        for gt in gt_candidates:
            gt_key = (str(gt.get("identity_type")), str(gt.get("identity_value")))
            if gt_key != key or not gt.get("semantic_explicit"):
                continue
            for raw in matches:
                if raw.get("semantic_explicit") is not True:
                    continue
                if pred.get("timestamp_us") is None or gt.get("timestamp_us") is None:
                    continue
                prediction_relative_us = int(pred.get("relative_us", pred["timestamp_us"]))
                gt_relative_us = int(gt.get("relative_us", gt["timestamp_us"]))
                pairs.append({
                    "identity_type": key[0],
                    "identity_value": key[1],
                    "timestamp_a": prediction_relative_us,
                    "timestamp_b": int(raw["timestamp_us"]),
                    "prediction_relative_us": prediction_relative_us,
                    "gt_relative_us": gt_relative_us,
                    "nurec_timestamp_us": int(raw["timestamp_us"]),
                    "source_prediction": pred.get("source_path"),
                    "source_gt": gt.get("source_path"),
                    "source_nurec": raw.get("source_path"),
                    "unit_verified": True,
                    "identity_verified": True,
                    "identity_source": raw.get("source_path"),
                    "offset_us": int(raw["timestamp_us"]) - prediction_relative_us,
                    "status": "VERIFIED_SEMANTIC_PAIR",
                })
    return pairs


def _timeline_consistency(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    reports: List[Dict[str, Any]] = []
    for row in rows:
        mode = row.get("mode")
        alpha = row.get("alpha")
        for item in _timeline_reports(row):
            path = str(item.get("trajectory_path", ""))
            role = "clean" if "clean" in path.lower() else "guided" if "guided" in path.lower() else "other"
            payload = {"role": role, "field": item.get("timestamp_field"), "values": item.get("timestamp_values", [])}
            reports.append({"mode": mode, "alpha": alpha, "role": role, "waypoint_path": path, "timestamp_field": item.get("timestamp_field"), "timestamp_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest(), "count": item.get("timestamp_count", 0), "first": item.get("first"), "last": item.get("last"), "median_step": item.get("median_step"), "status": "OBSERVED"})
    conflicts: List[str] = []
    for role in {str(item["role"]) for item in reports}:
        hashes = {str(item["timestamp_hash"]) for item in reports if item["role"] == role}
        if role != "other" and len(hashes) > 1:
            conflicts.append("PREDICTION_TIMELINE_CONFLICT")
    return reports, sorted(set(conflicts))


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
                    numeric_timestamps = [float(item) for item in timestamps if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))]
                    reports.append({"trajectory_path": child_path, "waypoint_count": len(child), "timestamp_field": field, "timestamp_kind": "absolute" if field in {"timestamp_micros", "t_us", "timestamp_us"} else "relative", "timestamp_count": len(timestamps), "timestamp_values": numeric_timestamps, "first": numeric_timestamps[0] if numeric_timestamps else None, "last": numeric_timestamps[-1] if numeric_timestamps else None, **_summarize_timeline_values(timestamps), "includes_t0_candidate": any(float(item) == 0 for item in timestamps if isinstance(item, (int, float)) and not isinstance(item, bool)), "unit_status": unit_evidence(field or "")["status"]})
                else:
                    reports.append({"trajectory_path": child_path, "waypoint_count": len(child), "timestamp_field": None, "timestamp_kind": None, "timestamp_count": 0, "timestamp_values": [], "first": None, "last": None, "median_step": None, "includes_t0_candidate": None, "unit_status": "TIMESTAMP_IMPLICIT_BY_PIPELINE"})
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
    timeline_consistency, timeline_conflicts = _timeline_consistency(prediction_rows)
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
    offset_inventory: List[Dict[str, Any]] = []
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
        offset_inventory.extend(_offset_inventory(value, str(path)))
        if extracted["metadata"]:
            explicit_metadata.update(extracted["metadata"])

    parquet_files = {"clip": "clipgt/clip.parquet", "egomotion": "clipgt/egomotion_estimate.parquet", "obstacle": "clipgt/obstacle.parquet", "association": "clipgt/association.parquet", "calibration": "clipgt/calibration_estimate.parquet"}
    nurec_identity_candidates: List[Dict[str, Any]] = []
    for source, relative in parquet_files.items():
        records, report = _parquet_records(clip_dir / relative, source)
        inventory.extend(records)
        schema["sources"][source] = report
        nurec_identity_candidates.extend(_parquet_identity_candidates(clip_dir / relative, source))
    for source, value in raw_json_values.items():
        nurec_identity_candidates.extend(extract_identity_candidates(value, source, str(clip_dir / json_files[source])))
    prediction_identity_candidates = [item for row in prediction_rows for item in extract_identity_candidates(row, "prediction", str(prediction_jsonl))]
    gt_identity_candidates = [item for row in gt_rows for item in extract_identity_candidates(row, "GT", str(ground_truth_jsonl))]
    semantic_pairs = extract_semantic_correspondences(prediction_identity_candidates, gt_identity_candidates, nurec_identity_candidates)
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
    if len({row.get("t0_us") for row in prediction_rows}) > 1 or timeline_conflicts:
        explicit_metadata["conflicting"] = True
        explicit_metadata.setdefault("evidence", []).append({"reason": "PREDICTION_TIMELINE_CONFLICT", "t0_values": sorted({row.get("t0_us") for row in prediction_rows}), "timeline_conflicts": timeline_conflicts})
    explicit_metadata["semantic_identity"] = bool(semantic_pairs)
    explicit_metadata["min_pairs"] = 2
    mapping = resolve_time_alignment_evidence(prediction_t0, gt_t0, nurec_clock, explicit_metadata, semantic_pairs, None, {"prediction_jsonl_error_count": len(prediction_errors), "gt_jsonl_error_count": len(gt_errors), "raw_metadata_errors": integrity["raw_metadata_errors"], "parquet_read_errors": integrity["parquet_read_errors"]}, {"nurec": (nurec_min, nurec_max)})
    matrix: List[Dict[str, Any]] = []
    for row in inventory:
        status = "UNIT_AMBIGUOUS" if row.get("unit_status") == UNIT_AMBIGUOUS else "NUMERICALLY_SIMILAR_ONLY"
        matrix.append({"source": row["source"], "timestamp_field": row["field"], "unit": row.get("unit_declared"), "unit_status": row.get("unit_status"), "origin_identifier": row.get("clock_domain_declared"), "origin_status": "ORIGIN_AMBIGUOUS", "range_min": row["min"], "range_max": row["max"], "median_step": row["median_step"], "exact_value_overlap_sources": "", "explicit_mapping_source": "", "semantic_identity_source": "", "status": status})
    correspondences = [{"source_a": "prediction", "source_b": "NuRec", "mapping_type": mapping.get("mapping_type"), "pair_count": len(semantic_pairs), "offset_min": mapping.get("offset_us"), "offset_max": mapping.get("offset_us"), "offset_unique_count": 1 if semantic_pairs and mapping.get("offset_us") is not None else 0, "residual": (mapping.get("evidence") or [{}])[0].get("residual") if mapping.get("evidence") else None, "identity_source": (mapping.get("source") or [None])[0], "status": mapping.get("status"), "reason": (mapping.get("blockers") or [None])[0]}]
    evidence = {
        "clip_id": clip_id,
        "prediction_clock": {"t0_field": "t0_us", "unit": "microseconds", "conditions": len(prediction_rows), "t0_values": sorted({row.get("t0_us") for row in prediction_rows}), "timeline_reports": prediction_timeline, "timestamp_contract_variants": len(prediction_timeline), "timestamp_contract": "FOUND" if any(report.get("timestamp_field") for report in prediction_timeline) else "WAYPOINT_TIMESTAMPS_NOT_PRESENT", "timeline_consistent_across_conditions": len({(report.get("trajectory_path"), report.get("timestamp_field"), report.get("timestamp_kind"), report.get("waypoint_count"), report.get("timestamp_count")) for report in prediction_timeline_all}) <= 2},
        "gt_clock": {"t0_field": "t0_us", "unit": "microseconds", "records": len(gt_rows), "timeline_reports": gt_timeline, "timestamp_contract": "FOUND" if any(report.get("timestamp_field") for report in gt_timeline) else "WAYPOINT_TIMESTAMPS_NOT_PRESENT"},
        "nurec_clock": {"timestamp_fields": nurec_clock["timestamp_fields"], "unit": "microseconds", "sources": ["clip", "egomotion", "obstacle"]},
        "mapping": mapping,
        "semantic_pairs": semantic_pairs,
        "timeline_consistency": timeline_consistency,
        "timeline_conflicts": timeline_conflicts,
        "blockers": mapping.get("blockers", []),
        "input_integrity": integrity,
        "explicit_origin_candidates": explicit_candidates,
        "numeric_relationships_are_not_verification": True,
        "data_info_offsets": [item for item in offset_inventory if "data_info.json" in str(item.get("source"))],
    }
    _write_csv(output_dir / "timestamp_inventory.csv", inventory)
    _write_csv(output_dir / "clock_domain_matrix.csv", matrix)
    _write_csv(output_dir / "timestamp_correspondences.csv", correspondences)
    _write_csv(output_dir / "semantic_correspondences.csv", semantic_pairs or [{"identity_type": None, "identity_value": None, "prediction_condition": None, "prediction_relative_us": None, "gt_relative_us": None, "nurec_timestamp_us": None, "source_prediction": None, "source_gt": None, "source_nurec": None, "unit_verified": False, "identity_verified": False, "offset_us": None, "status": "MISSING_FRAME_CORRESPONDENCE"}])
    _write_csv(output_dir / "timeline_consistency.csv", timeline_consistency)
    (output_dir / "time_schema_report.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "time_alignment_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "input_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = _t0_provenance_candidates(prediction_rows, gt_rows, explicit_candidates)
    (output_dir / "t0_provenance_candidates.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "prediction_gt_identity_candidates.json").write_text(json.dumps({"prediction": prediction_identity_candidates, "GT": gt_identity_candidates, "status": "NO_EXPLICIT_SHARED_IDENTITY" if not prediction_identity_candidates or not gt_identity_candidates else "CANDIDATES_REQUIRE_SEMANTIC_LINK"}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "nurec_identity_candidates.json").write_text(json.dumps({"candidates": nurec_identity_candidates, "status": "CANDIDATES_REQUIRE_SEMANTIC_LINK" if nurec_identity_candidates else "NO_IDENTITY_CANDIDATES"}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    t0_report = _t0_origin_report(prediction_rows, gt_rows, provenance, semantic_pairs)
    t0_report["data_info_offsets"] = [item for item in offset_inventory if "data_info.json" in str(item.get("source"))]
    (output_dir / "t0_origin_report.json").write_text(json.dumps(t0_report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "data_info_offsets.json").write_text(json.dumps(offset_inventory, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if repo_root is not None:
        trace = _upstream_assignment_trace(repo_root)
        (output_dir / "upstream_time_provenance.md").write_text(_upstream_provenance_report(repo_root, trace), encoding="utf-8")
        (output_dir / "upstream_assignment_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = _markdown_report(evidence, inventory, schema)
    (output_dir / "time_schema_report.md").write_text(report[0], encoding="utf-8")
    (output_dir / "time_alignment_report.md").write_text(report[1], encoding="utf-8")
    (output_dir / "time_alignment_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "upstream_time_provenance.md").write_text(_upstream_provenance_report(repo_root, _upstream_assignment_trace(repo_root)) if repo_root is not None else "UPSTREAM_T0_PROVENANCE_NOT_REQUESTED\n", encoding="utf-8")
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


def _upstream_assignment_trace(repo_root: Path) -> Dict[str, Any]:
    patterns = ["t0_us", "5100000", "5_100_000", "ego_future_xyz", "ego_future_gt", "ground_truth", "reasoning_intervention", "nurec_selected", "clean_waypoints", "guided_waypoints", "source_includes_t0", "trajectory_origin_policy", "current_frame", "frame_index", "sample_token", "timestamp_micros", "pd.read_parquet", "egomotion_estimate", "pose_record", "range(64)", "[:64]", "[1:65]", "current_idx", "anchor_idx", "future_idx", "fps", "10.0", "0.1"]
    allowed = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
    hits: List[Dict[str, Any]] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in allowed:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        function = None
        for number, line in enumerate(lines, 1):
            matched = [pattern for pattern in patterns if pattern.lower() in line.lower()]
            if re.match(r"\s*(def|async def|class)\s+", line):
                function = line.strip()
            if not matched:
                continue
            relative = str(path.relative_to(repo_root))
            context = lines[max(0, number - 3):min(len(lines), number + 2)]
            code_source = path.suffix.lower() == ".py"
            hits.append({
                "file": relative,
                "line": number,
                "function": function,
                "match": line.strip(),
                "matched_patterns": matched,
                "context": context,
                "upstream_input": "repository source/text match; input assignment not established",
                "derived_value": "t0_us or trajectory/frame candidate only" if any(p in {"t0_us", "5100000", "5_100_000"} for p in matched) else None,
                "downstream_output": "prediction/GT/NuRec alignment evidence candidate only",
                "provenance_status": "CANDIDATE_ONLY" if not code_source else "CODE_MATCH_REQUIRES_ASSIGNMENT_TRACE",
            })
    code_matches = [item for item in hits if str(item["file"]).endswith(".py")]
    t0_code_matches = [item for item in code_matches if any(p in {"t0_us", "5100000", "5_100_000"} for p in item["matched_patterns"])]
    status = "VERIFIED" if False else "DERIVED_BUT_NO_NUREC_BRIDGE" if t0_code_matches else "NOT_FOUND"
    return {"repository": str(repo_root), "patterns": patterns, "matches": hits, "code_match_count": len(code_matches), "t0_code_match_count": len(t0_code_matches), "status": status, "raw_nurec_bridge_found": False}


def _upstream_provenance_report(repo_root: Path, trace: Optional[Mapping[str, Any]] = None) -> str:
    trace = trace or _upstream_assignment_trace(repo_root)
    lines = ["# Upstream time provenance", "", f"Repository: `{repo_root}`", "", f"UPSTREAM_STATUS = `{trace.get('status')}`", f"WHY_IS_T0_5100000 = `{'DERIVED_BUT_NO_NUREC_BRIDGE' if trace.get('status') == 'DERIVED_BUT_NO_NUREC_BRIDGE' else trace.get('status')}`", ""]
    if trace.get("matches"):
        lines += ["## Line-level trace", "", "| FILE | LINE | FUNCTION | MATCH | PROVENANCE STATUS |", "|---|---:|---|---|---|"]
        for item in trace["matches"]:
            lines.append(f"| `{item['file']}` | {item['line']} | `{item.get('function') or ''}` | `{str(item['match']).replace('|', '\\|')}` | `{item['provenance_status']}` |")
    else:
        lines += ["UPSTREAM_T0_PROVENANCE_NOT_FOUND_IN_REPOSITORY", "", "No requested upstream construction term was found."]
    lines += ["", "Matches in tests/docs/config are not treated as generation provenance. A raw NuRec frame/pose bridge must still be explicit before alignment can be verified."]
    return "\n".join(lines) + "\n"


def _t0_origin_report(prediction_rows: Sequence[Mapping[str, Any]], gt_rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any], semantic_pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = sorted({row.get("t0_us") for row in list(prediction_rows) + list(gt_rows) if row.get("t0_us") is not None})
    if semantic_pairs:
        status = "VERIFIED"
    elif provenance.get("status") == "DERIVED_BUT_NO_NUREC_BRIDGE":
        status = "DERIVED_BUT_NO_NUREC_BRIDGE"
    else:
        status = "NOT_FOUND"
    return {"WHY_IS_T0_5100000": status, "t0_values_observed": values, "prediction_field": "t0_us", "gt_field": "t0_us", "semantic_pair_count": len(semantic_pairs), "raw_nurec_bridge_found": bool(semantic_pairs), "status": status, "explanation": "t0_us is present in prediction/GT records, but no verified assignment to a NuRec frame/pose timestamp was found."}


def _offset_inventory(value: Any, source_path: str, path: str = "") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if re.search(r"offset", str(key), re.I) and _scalar(child):
                unit = unit_evidence(child_path)
                records.append({"path": child_path, "value": child, "unit": unit.get("unit"), "unit_status": unit.get("status"), "what_it_offsets": "declared by field path only; semantic target not established", "source": source_path, "target": None, "verified": False})
            records.extend(_offset_inventory(child, source_path, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value[:5000]):
            records.extend(_offset_inventory(child, source_path, f"{path}[{index}]"))
    return records


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
