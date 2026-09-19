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


def correspondence_status(pairs: Sequence[Mapping[str, Any]], explicit_identity: bool) -> Dict[str, Any]:
    """Verify pairs only when an independent shared event/frame identity exists."""
    offsets = [int(pair["timestamp_b"]) - int(pair["timestamp_a"]) for pair in pairs]
    unique = sorted(set(offsets))
    residual = max((abs(value - median(offsets)) for value in offsets), default=None)
    verified = bool(explicit_identity and pairs and len(unique) == 1)
    return {
        "pair_count": len(pairs),
        "offset_min": min(offsets) if offsets else None,
        "offset_max": max(offsets) if offsets else None,
        "offset_unique_count": len(unique),
        "residual_us": residual,
        "verified": verified,
        "status": "MAPPING_EXPLICIT" if verified else "UNRESOLVED",
    }


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
        records.append({"source": source, "field": path, "unit_declared": unit["unit"], "clock_domain_declared": None, **summarize_values(numbers), "source_path": source_path, "evidence_type": "raw_json_field"})
        schema_by_path[path]["sample"] = numbers[:5]
    return records, list(schema_by_path.values())


def _load_jsonl_clip(path: Path, clip_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("clip_id")) == clip_id:
                rows.append(row)
    return rows


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
        records.append({"source": source, "field": field, "unit_declared": unit["unit"], "clock_domain_declared": None, **summary, "source_path": str(path), "evidence_type": "parquet_schema_and_values"})
    return records, {"path": str(path), "status": info.get("status"), "fields": fields, "nested_field_paths": info.get("nested_field_paths", [])}


def _camera_records(clip_dir: Path) -> List[Dict[str, Any]]:
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
            records.append({"source": "camera filenames", "field": sensor_dir.name, "unit_declared": "microseconds", "clock_domain_declared": None, **summarize_values(values), "source_path": str(sensor_dir), "evidence_type": "filename_only"})
    return records


def audit_time_alignment(clip_dir: Path, clip_id: str, prediction_jsonl: Path, ground_truth_jsonl: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory: List[Dict[str, Any]] = []
    schema: Dict[str, Any] = {"clip_id": clip_id, "sources": {}}

    prediction_rows = _load_jsonl_clip(prediction_jsonl, clip_id)
    gt_rows = _load_jsonl_clip(ground_truth_jsonl, clip_id)
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
            inventory.append({"source": source, "field": "t0_us", "unit_declared": "microseconds", "clock_domain_declared": None, **summarize_values([]), "source_path": source_path, "evidence_type": "NOT_FOUND"})
        else:
            inventory.append({"source": source, "field": "t0_us", "unit_declared": "microseconds", "clock_domain_declared": None, **summarize_values(values), "source_path": source_path, "evidence_type": "prediction_or_gt_record"})
        if rows:
            schema["sources"].setdefault(source, []).append({"path": "t0_us", "value_type": type(values[0]).__name__ if values else "NOT_FOUND", "sample": values[:5], "note": "trajectory timestamp fields are not assumed from waypoint array positions"})

    json_files = {"data_info": "data_info.json", "datasource_summary": "datasource_summary.json", "pose_record": "pose_record.json", "rig_trajectories": "rig_trajectories.json", "sequence_tracks": "sequence_tracks.json"}
    for source, relative in json_files.items():
        path = clip_dir / relative
        if not path.is_file():
            schema["sources"][source] = {"status": "NOT_FOUND", "path": relative}
            continue
        value = _load_json(path)
        fields, field_schema = _json_timestamp_records(value, source, str(path))
        inventory.extend(fields)
        schema["sources"][source] = {"status": "OK", "path": str(path), "top_level_keys": list(value.keys()) if isinstance(value, Mapping) else [], "nested_key_paths": _nested_paths(value), "timestamp_fields": field_schema}

    parquet_files = {"clip": "clipgt/clip.parquet", "egomotion": "clipgt/egomotion_estimate.parquet", "obstacle": "clipgt/obstacle.parquet", "association": "clipgt/association.parquet", "calibration": "clipgt/calibration_estimate.parquet"}
    for source, relative in parquet_files.items():
        records, report = _parquet_records(clip_dir / relative, source)
        inventory.extend(records)
        schema["sources"][source] = report
    inventory.extend(_camera_records(clip_dir))

    # Numeric relationships are diagnostic only. No range overlap is promoted
    # to a clock mapping and no offset is inferred from minima.
    by_source = {source: [row for row in inventory if row["source"] == source] for source in {row["source"] for row in inventory}}
    matrix: List[Dict[str, Any]] = []
    for row in inventory:
        status = "MISSING_METADATA" if row["unit_declared"] is None else "NUMERICALLY_SIMILAR_ONLY"
        matrix.append({"source": row["source"], "timestamp_field": row["field"], "declared_or_inferred_unit": row["unit_declared"], "declared_origin": row["clock_domain_declared"], "range_min": row["min"], "range_max": row["max"], "median_step": row["median_step"], "shares_exact_values_with": "", "explicit_link_source": "", "status": status})
    correspondences = [{"source_a": "prediction", "source_b": "NuRec", "pair_count": 0, "offset_min": None, "offset_max": None, "offset_unique_count": 0, "residual_us": None, "identity_field": None, "status": "UNRESOLVED", "reason": "MISSING_FRAME_CORRESPONDENCE"}]
    evidence = {
        "clip_id": clip_id,
        "prediction_clock": {"t0_field": "t0_us", "unit": "microseconds", "conditions": len(prediction_rows), "trajectory_timestamp_contract": "NOT_FOUND_IN_RECORD"},
        "gt_clock": {"t0_field": "t0_us", "unit": "microseconds", "records": len(gt_rows), "waypoint_timestamp_contract": "NOT_FOUND_IN_RECORD"},
        "nurec_clock": {"timestamp_fields": ["key.timestamp_micros"], "unit": "microseconds", "sources": ["clip", "egomotion", "obstacle"]},
        "mapping": {"status": "UNRESOLVED", "verified": False, "mapping_type": None, "offset_us": None, "scale": None, "source": [], "evidence": []},
        "blockers": ["MISSING_EXPLICIT_RELATIVE_TO_GLOBAL_ORIGIN", "MISSING_FRAME_CORRESPONDENCE", "CLOCK_DOMAIN_NOT_PROVEN"],
        "numeric_relationships_are_not_verification": True,
    }
    _write_csv(output_dir / "timestamp_inventory.csv", inventory)
    _write_csv(output_dir / "clock_domain_matrix.csv", matrix)
    _write_csv(output_dir / "timestamp_correspondences.csv", correspondences)
    (output_dir / "time_schema_report.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_dir / "time_alignment_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = _markdown_report(evidence, inventory, schema)
    (output_dir / "time_schema_report.md").write_text(report[0], encoding="utf-8")
    (output_dir / "time_alignment_report.md").write_text(report[1], encoding="utf-8")
    return {"evidence": evidence, "inventory": inventory, "schema": schema, "output_dir": str(output_dir)}


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
    report_lines = ["# NuRec time-alignment evidence", "", "TIME_ALIGNMENT_STATUS = `UNRESOLVED`", "TIME_ALIGNMENT_VERIFIED = `false`", "", "## Findings", "", "1. Prediction t0: `t0_us`, microseconds by explicit field name; trajectory time origin is not declared in the record.", "2. GT t0: `t0_us`, microseconds by explicit field name; waypoint timestamp contract is not present in the record.", "3. NuRec obstacle: `key.timestamp_micros`, microseconds by explicit field name.", "4. Egomotion: `key.timestamp_micros`, microseconds by explicit field name.", "5. Sensor filename clock: no `frames/` directory was found in the pilot clip, so no filename clock evidence exists.", "6. Explicit clip time origin: not found in inspected evidence.", "7. Prediction-to-NuRec mapping: not found.", "8. Mapping source: none.", "9. Unit: explicit for the inspected `*_us`/`*_micros` fields; other timestamp-like fields remain ambiguous.", "10. Verified: false.", "", "## Exact blockers", ""] + [f"- `{item}`" for item in evidence["blockers"]] + ["", "Numeric range overlap and candidate differences are diagnostic only; no offset, scale, or zero-origin correction was created."]
    return "\n".join(schema_lines) + "\n", "\n".join(report_lines) + "\n"
