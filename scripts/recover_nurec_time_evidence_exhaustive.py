"""Exhaustive lightweight local evidence audit for the NuRec full-300 clock.

The audit has deliberately separate discovery and verification stages.  It
scans source contents and known per-clip metadata regardless of filename, but
only promotes an offset when provenance satisfies the frozen time contract.
No offset is estimated from ranges, minima, nearest timestamps, or aggregate
statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.recover_nurec_time_mapping_full300 import (
        EXPECTED_CLIP_COUNT,
        HISTORICAL_IDS,
        HISTORICAL_MIN_PAIR_COUNT,
        compute_nurec_t0,
        load_t0_index,
        read_jsonl,
        verify_physicalai_t0,
        verify_semantic_pairs,
    )
except ModuleNotFoundError:
    from recover_nurec_time_mapping_full300 import (
        EXPECTED_CLIP_COUNT,
        HISTORICAL_IDS,
        HISTORICAL_MIN_PAIR_COUNT,
        compute_nurec_t0,
        load_t0_index,
        read_jsonl,
        verify_physicalai_t0,
        verify_semantic_pairs,
    )


FINAL_STATUSES = {
    "VERIFIED_REUSED_EXISTING",
    "VERIFIED_REGENERATED_EXPLICIT_METADATA",
    "VERIFIED_REGENERATED_ESTABLISHED_METHOD",
    "CONFLICTING_EVIDENCE",
    "MISSING_REQUIRED_EVIDENCE",
    "INPUT_ERROR",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml"}
KNOWN_METADATA_FILES = (
    "rig_trajectories.json",
    "pose_record.json",
    "data_info.json",
    "datasource_summary.json",
    "metadata.yaml",
    "metadata.yml",
    "parsed_config.yaml",
    "parsed_config.yml",
    "sequence_tracks.json",
)
EXACT_TIME_FIELDS = {
    "t0_us",
    "physicalai_t0_us",
    "nurec_t0_us",
    "offset_us",
    "per_clip_offset_us",
    "offset_start_us",
    "offset_end_us",
}
MAPPING_KEYS = {
    "offset_us",
    "per_clip_offset_us",
    "offset_start_us",
    "offset_end_us",
    "nurec_t0_us",
    "physicalai_t0_us",
    "mapping_type",
    "verification_status",
    "semantic_identity_source",
    "verification_method",
    "pair_count",
    "offset_residual_max_us",
}
EXCLUDED_DIR_NAMES = {
    "images",
    "camera",
    "cameras",
    "mesh",
    "meshes",
    "volume",
    "videos",
    "checkpoints",
}
MAX_GENERIC_SCAN_BYTES = 64 * 1024 * 1024


def _int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_numbers(child)


def _is_time_field(field_name: str) -> bool:
    lowered = field_name.lower()
    return (
        "timestamp" in lowered
        or lowered.endswith("_t0")
        or lowered.endswith("_t0_us")
        or lowered in EXACT_TIME_FIELDS
        or "offset" in lowered
        or lowered in {"start_time", "end_time", "start_us", "end_us", "duration_us"}
    )


def _unit_evidence(field_name: str) -> str:
    lowered = field_name.lower()
    if lowered.endswith("_us") or "micros" in lowered:
        return "EXPLICIT_MICROSECONDS_FIELD_NAME"
    if "timestamp" in lowered:
        return "TIMESTAMP_UNIT_NOT_DECLARED"
    return "NOT_APPLICABLE"


def _clock_evidence(path: str) -> str:
    lowered = path.lower()
    domains = []
    for token, label in (
        ("physicalai", "PHYSICALAI"),
        ("ncore", "NCORE"),
        ("nurec", "NUREC"),
        ("sensor", "SENSOR"),
        ("ego", "EGO"),
        ("pose", "POSE"),
    ):
        if token in lowered:
            domains.append(label)
    return "+".join(domains) if domains else "CLOCK_DOMAIN_NOT_DECLARED"


def _candidate_role(path: str, field_name: str) -> str:
    lowered = f"{path}.{field_name}".lower()
    field = field_name.lower()
    if field in {"nurec_t0_us", "physicalai_t0_us", "offset_us", "per_clip_offset_us", "offset_start_us", "offset_end_us"}:
        return "EXPLICIT_TIME_ORIGIN"
    if "reference_frame_timestamp" in field or "frame_timestamp" in field or "same_event" in lowered:
        return "SEMANTIC_FRAME_CORRESPONDENCE"
    if "sequence_tracks" in lowered or "obstacle" in lowered or "rig_trajector" in lowered or "egomotion" in lowered:
        return "NUREC_TIMELINE_ONLY"
    if field == "t0_us" or "physicalai" in lowered or "pai" in lowered:
        return "PAI_TIMELINE_ONLY"
    if "timestamp" in field:
        return "DIAGNOSTIC_ONLY"
    return "UNRESOLVED"


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.first_values: list[float] = []

    def add(self, value: Any) -> None:
        for number in _flatten_numbers(value):
            self.count += 1
            self.minimum = number if self.minimum is None else min(self.minimum, number)
            self.maximum = number if self.maximum is None else max(self.maximum, number)
            if len(self.first_values) < 5:
                self.first_values.append(number)


def collect_timestamp_fields(value: Any, path: str = "") -> dict[tuple[str, str], _Accumulator]:
    """Summarize time-like fields without retaining the giant source object."""
    accumulators: dict[tuple[str, str], _Accumulator] = {}

    def visit(child: Any, child_path: str) -> None:
        if isinstance(child, dict):
            for key, nested in child.items():
                field_name = str(key)
                next_path = f"{child_path}.{field_name}" if child_path else field_name
                if _is_time_field(field_name):
                    accumulator = accumulators.setdefault((next_path, field_name), _Accumulator())
                    accumulator.add(nested)
                if isinstance(nested, (dict, list)):
                    visit(nested, next_path)
        elif isinstance(child, list):
            for nested in child:
                if isinstance(nested, (dict, list)):
                    visit(nested, child_path + "[]")

    visit(value, path)
    return accumulators


def parse_context_full(
    path: Path,
    output_csv: Path,
    current_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], int, list[dict[str, Any]]]:
    """Dedicated streaming parser for nurec_context_full_300.jsonl."""
    fields = [
        "clip_id", "json_path", "field_name", "value_type", "count", "min", "max",
        "first_values", "unit_evidence", "clock_domain_evidence", "candidate_role",
    ]
    context_index: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    record_count = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8") as source, output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            clip_id = str(record.get("clip_id", ""))
            if not clip_id:
                continue
            record_count += 1
            context_index[clip_id] = {
                "record_available": True,
                "line_number": line_no,
                "files_present": record.get("files_present", []),
                "files_missing": record.get("files_missing", []),
                "context_status": record.get("context_status"),
            }
            summaries = collect_timestamp_fields(record)
            for (json_path, field_name), accumulator in summaries.items():
                role = _candidate_role(json_path, field_name)
                writer.writerow({
                    "clip_id": clip_id,
                    "json_path": json_path,
                    "field_name": field_name,
                    "value_type": "numeric_or_numeric_list",
                    "count": accumulator.count,
                    "min": accumulator.minimum,
                    "max": accumulator.maximum,
                    "first_values": json.dumps(accumulator.first_values),
                    "unit_evidence": _unit_evidence(field_name),
                    "clock_domain_evidence": _clock_evidence(json_path),
                    "candidate_role": role,
                })
            # Context summary is a discovery candidate, never a mapping proof.
            candidates.append({
                "clip_id": clip_id,
                "physicalai_t0_us": _int(record.get("t0_us")),
                "nurec_t0_us": _int(record.get("nurec_t0_us")),
                "offset_us": _int(record.get("offset_us")),
                "scale": _float(record.get("scale", 1.0)),
                "source_file": str(path),
                "source_record": line_no,
                "json_path": "t0_us|semantic_context|pose_context",
                "mapping_type": record.get("mapping_type") or "CONTEXT_SUMMARY_ONLY",
                "candidate_type": "CONTEXT_FULL_TIME_EVIDENCE",
                "candidate_role": "PAI_TIMELINE_ONLY",
                "evidence_description": "Full context summary timestamp/provenance inventory; no cross-domain proof",
                "current_300_member": clip_id in current_ids,
                "candidate_status": "EXISTING_UNVERIFIED",
                "verification_level": "LEVEL_1_DECLARED_TIMESTAMP" if record.get("t0_us") is not None else "LEVEL_0_NUMERIC_ONLY",
                "verification_status": "UNVERIFIED",
                "semantic_identity": False,
                "offset_residual_max_us": None,
                "pair_count": 0,
            })
    return context_index, record_count, candidates


def _iter_text_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part.lower() in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _clip_id_from_row(row: dict[str, Any], current_ids: set[str]) -> str | None:
    for key in ("clip_id", "scene_id", "sequence_id", "source_clip_id", "target_clip_id"):
        value = row.get(key)
        if value is not None and str(value) in current_ids:
            return str(value)
    return None


def _pair_info(row: dict[str, Any]) -> dict[str, Any]:
    pair_values = row.get("semantic_pairs") or row.get("pairs") or []
    if not isinstance(pair_values, list):
        pair_values = []
    offsets: list[int] = []
    semantic = True
    for pair in pair_values:
        if not isinstance(pair, dict):
            semantic = False
            continue
        offset = _int(pair.get("offset_us"))
        if offset is not None:
            offsets.append(offset)
        semantic = semantic and _bool(pair.get("semantic_identity"))
    semantic_marker = any(
        row.get(key) not in (None, "")
        for key in (
            "semantic_pairs",
            "pairs",
            "semantic_identity",
            "semantic_identity_source",
            "verification_method",
            "exact_relative_timestamp_matches",
        )
    )
    declared_pair_count = _int(row.get("pair_count")) if semantic_marker else None
    if declared_pair_count is None:
        declared_pair_count = _int(row.get("exact_relative_timestamp_matches"))
    pair_count = declared_pair_count if declared_pair_count is not None else len(pair_values)
    if not pair_values and declared_pair_count:
        # The accepted five-clip match CSV records the count of explicit
        # relative timestamp correspondences rather than materializing each
        # pair.  Its historical acceptance is handled separately, but retain
        # the count for the audit and canonical contract.
        semantic = _bool(row.get("semantic_identity")) or row.get("exact_relative_timestamp_matches") not in (None, "")
        offset = _int(row.get("offset_us"))
        if offset is None:
            offset = _int(row.get("offset_start_us"))
        if offset is not None:
            offsets = [offset] * declared_pair_count
    unique = sorted(set(offsets))
    residual = _int(row.get("offset_residual_max_us"))
    if residual is None and row.get("offset_start_us") not in (None, "") and row.get("offset_end_us") not in (None, ""):
        start = _int(row.get("offset_start_us"))
        end = _int(row.get("offset_end_us"))
        if start is not None and end is not None:
            residual = abs(end - start)
    if residual is None and offsets:
        residual = max(offsets) - min(offsets)
    return {
        "pair_count": pair_count,
        "semantic_identity": semantic and pair_count > 0,
        "offsets": offsets,
        "unique_offsets": unique,
        "offset_residual_max_us": residual,
    }


def _candidate_from_row(
    path: Path,
    row: dict[str, Any],
    record_number: int | str,
    current_ids: set[str],
    historical_contract: Path,
    historical_match: Path,
) -> dict[str, Any] | None:
    clip_id = _clip_id_from_row(row, current_ids)
    if clip_id is None:
        return None
    field_names = {str(key).lower() for key in row}
    mappingish = bool(field_names & {key.lower() for key in MAPPING_KEYS})
    if not mappingish:
        return None
    offset = _int(row.get("offset_us"))
    if offset is None:
        offset = _int(row.get("per_clip_offset_us"))
    if offset is None:
        offset = _int(row.get("offset_start_us"))
    nurec_t0 = _int(row.get("nurec_t0_us"))
    physicalai_t0 = _int(row.get("physicalai_t0_us"))
    scale = _float(row.get("scale", 1.0))
    pair_info = _pair_info(row)
    historical = path.resolve() in {historical_contract.resolve(), historical_match.resolve()}
    historical = historical and offset is not None and (
        "offset_equal" not in row or _bool(row.get("offset_equal"))
    )
    has_explicit_provenance = any(
        row.get(key) not in (None, "")
        for key in ("verification_method", "semantic_identity_source", "mapping_provenance", "explicit_offset_source", "source_clock_domain")
    )
    declared_verified = _bool(row.get("verified")) or _bool(row.get("mapping_verified")) or str(row.get("verification_status", "")).upper() == "VERIFIED"
    complete_equation = (
        offset is not None
        and physicalai_t0 is not None
        and nurec_t0 is not None
        and nurec_t0 == physicalai_t0 + offset
        and scale == 1.0
    )
    if historical:
        level = "HISTORICAL_ACCEPTED"
        verification_status = "VERIFIED"
        candidate_status = "EXISTING_VERIFIED"
        verification_method = "ACCEPTED_HISTORICAL_FIVE_CLIP_CONTRACT"
        semantic_source = "accepted PAI/NuRec pose timeline artifact"
    elif pair_info["pair_count"] >= HISTORICAL_MIN_PAIR_COUNT and pair_info["semantic_identity"] and pair_info["unique_offsets"] and len(pair_info["unique_offsets"]) == 1 and scale == 1.0 and declared_verified:
        level = "LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET"
        verification_status = "VERIFIED"
        candidate_status = "EXISTING_VERIFIED"
        verification_method = "EQUIVALENT_MULTI_PAIR_CONSTANT_OFFSET"
        semantic_source = row.get("semantic_identity_source") or row.get("verification_method") or "declared semantic pair evidence"
    elif complete_equation and declared_verified and has_explicit_provenance:
        level = "LEVEL_2_EXPLICIT_ORIGIN_METADATA"
        verification_status = "VERIFIED"
        candidate_status = "EXISTING_VERIFIED"
        verification_method = row.get("verification_method") or "EXPLICIT_ORIGIN_METADATA"
        semantic_source = row.get("semantic_identity_source") or row.get("source_clock_domain") or "explicit local origin metadata"
    elif pair_info["pair_count"] >= 1 and pair_info["semantic_identity"]:
        level = "LEVEL_3_SEMANTIC_CROSS_DOMAIN_PAIR"
        verification_status = "UNVERIFIED"
        candidate_status = "EXISTING_UNVERIFIED"
        verification_method = None
        semantic_source = None
    elif any(
        token in key
        for key in field_names
        for token in ("origin", "explicit_offset", "clock_domain", "mapping_provenance")
    ):
        level = "LEVEL_2_EXPLICIT_ORIGIN_METADATA"
        verification_status = "UNVERIFIED"
        candidate_status = "EXISTING_UNVERIFIED"
        verification_method = None
        semantic_source = None
    elif offset is not None or any("timestamp" in key or key.endswith("_us") for key in field_names):
        level = "LEVEL_1_DECLARED_TIMESTAMP" if offset is None else "LEVEL_0_NUMERIC_ONLY"
        verification_status = "UNVERIFIED"
        candidate_status = "EXISTING_UNVERIFIED"
        verification_method = None
        semantic_source = None
    else:
        level = "LEVEL_0_NUMERIC_ONLY"
        verification_status = "UNVERIFIED"
        candidate_status = "EXISTING_UNVERIFIED"
        verification_method = None
        semantic_source = None
    candidate_type = "MAPPING_CANDIDATE" if offset is not None else "TIME_EVIDENCE_CANDIDATE"
    return {
        "clip_id": clip_id,
        "physicalai_t0_us": physicalai_t0,
        "nurec_t0_us": nurec_t0,
        "offset_us": offset,
        "scale": scale,
        "source_file": str(path),
        "source_record": record_number,
        "json_path": row.get("json_path") or "record",
        "mapping_type": row.get("mapping_type") or "UNKNOWN",
        "candidate_type": candidate_type,
        "candidate_role": _candidate_role(str(path), "offset_us" if offset is not None else "timestamp"),
        "verification_flag": verification_status == "VERIFIED",
        "verification_level": level,
        "verification_method": verification_method,
        "semantic_identity_source": semantic_source,
        "semantic_identity": pair_info["semantic_identity"],
        "pair_count": pair_info["pair_count"],
        "offset_residual_max_us": pair_info["offset_residual_max_us"],
        "evidence_type": "HISTORICAL_ACCEPTED" if historical else "LOCAL_CONTENT_DISCOVERY",
        "evidence_description": "Structured content candidate; verification is provenance-based, not filename-based",
        "current_300_member": True,
        "candidate_status": candidate_status,
        "verification_status": verification_status,
        "multiple_source_agreement": None,
    }


def _structured_rows(path: Path) -> Iterable[tuple[int | str, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for index, row in enumerate(csv.DictReader(handle), 2):
                yield index, dict(row)
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    yield index, value
        return
    if suffix == ".json":
        try:
            value = _json(path)
        except Exception:
            return
        if isinstance(value, dict):
            yield 1, value
            for key, child in value.items():
                if isinstance(child, list):
                    for index, row in enumerate(child, 1):
                        if isinstance(row, dict):
                            yield f"{key}[{index}]", row
        elif isinstance(value, list):
            for index, row in enumerate(value, 1):
                if isinstance(row, dict):
                    yield index, row


def inspect_source_files(
    files: Iterable[Path],
    current_ids: set[str],
    historical_contract: Path,
    historical_match: Path,
    skip_paths: set[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    clip_id_needles = sorted(current_ids)
    for path in files:
        resolved = path.resolve()
        try:
            size = path.stat().st_size
        except OSError as exc:
            inventory.append({"source_file": str(path), "exists": False, "read_status": "STAT_ERROR", "error": str(exc)})
            continue
        row = {
            "source_file": str(path),
            "exists": True,
            "size_bytes": size,
            "suffix": path.suffix.lower(),
            "explicitly_requested": resolved in skip_paths,
            "read_status": "NOT_READ",
            "current_clip_id_hit_count": 0,
            "exact_field_hit_count": 0,
            "structured_candidate_count": 0,
            "verified_candidate_count": 0,
            "skip_reason": None,
            "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
        }
        if resolved in skip_paths:
            row["read_status"] = "READ_BY_DEDICATED_PARSER"
            row["skip_reason"] = "DEDICATED_CONTEXT_FULL_STREAM"
            inventory.append(row)
            continue
        if size > MAX_GENERIC_SCAN_BYTES:
            row["read_status"] = "SKIPPED_SIZE_LIMIT"
            row["skip_reason"] = f"TEXT_FILE_LARGER_THAN_{MAX_GENERIC_SCAN_BYTES}_BYTES"
            inventory.append(row)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            row["read_status"] = "READ"
            row["current_clip_id_hit_count"] = sum(1 for clip_id in clip_id_needles if clip_id in text)
            row["exact_field_hit_count"] = sum(len(re.findall(rf"[\\\"']{re.escape(field)}[\\\"']", text)) for field in MAPPING_KEYS)
            for record_number, structured_row in _structured_rows(path):
                candidate = _candidate_from_row(path, structured_row, record_number, current_ids, historical_contract, historical_match)
                if candidate is not None:
                    candidates.append(candidate)
                    row["structured_candidate_count"] += 1
                    if candidate["verification_status"] == "VERIFIED":
                        row["verified_candidate_count"] += 1
            if row["current_clip_id_hit_count"] and row["structured_candidate_count"] == 0:
                for clip_id in clip_id_needles:
                    if clip_id in text:
                        candidates.append({
                            "clip_id": clip_id,
                            "physicalai_t0_us": None,
                            "nurec_t0_us": None,
                            "offset_us": None,
                            "scale": 1.0,
                            "source_file": str(path),
                            "source_record": None,
                            "json_path": "text_content",
                            "mapping_type": "TEXT_CONTENT_ONLY",
                            "candidate_type": "TEXT_TIME_EVIDENCE",
                            "candidate_role": "DIAGNOSTIC_ONLY",
                            "verification_flag": False,
                            "verification_level": "LEVEL_0_NUMERIC_ONLY",
                            "verification_method": None,
                            "semantic_identity_source": None,
                            "semantic_identity": False,
                            "pair_count": 0,
                            "offset_residual_max_us": None,
                            "evidence_type": "LOCAL_CONTENT_DISCOVERY",
                            "evidence_description": "Clip ID and/or time terms found in text; no structured proof",
                            "current_300_member": True,
                            "candidate_status": "EXISTING_UNVERIFIED",
                            "verification_status": "UNVERIFIED",
                            "multiple_source_agreement": None,
                        })
        except Exception as exc:
            row["read_status"] = "READ_ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(row)
    return inventory, candidates


def _clip_metadata_inventory(manifest: dict[str, dict[str, Any]], context_index: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_clip[candidate["clip_id"]].append(candidate)
    rows: list[dict[str, Any]] = []
    for clip_id in sorted(manifest):
        item = manifest[clip_id]
        clip_dir = Path(str(item.get("nurec_clip_dir", "")))
        available = {name: (clip_dir / name).is_file() for name in KNOWN_METADATA_FILES}
        clip_candidates = by_clip.get(clip_id, [])
        levels = [candidate["verification_level"] for candidate in clip_candidates]
        semantic_count = sum(1 for candidate in clip_candidates if candidate["semantic_identity"] or candidate["candidate_role"] == "SEMANTIC_FRAME_CORRESPONDENCE")
        explicit_count = sum(1 for candidate in clip_candidates if candidate["verification_level"] in {"LEVEL_2_EXPLICIT_ORIGIN_METADATA", "HISTORICAL_ACCEPTED"})
        if any(candidate["verification_status"] == "VERIFIED" for candidate in clip_candidates):
            evidence_status = "VERIFIED_MAPPING_CANDIDATE_PRESENT"
        elif clip_candidates:
            evidence_status = "CANDIDATE_ONLY"
        else:
            evidence_status = "NO_LOCAL_TIME_EVIDENCE"
        rows.append({
            "clip_id": clip_id,
            "rig_trajectories_available": available["rig_trajectories.json"],
            "pose_record_available": available["pose_record.json"],
            "data_info_available": available["data_info.json"],
            "datasource_summary_available": available["datasource_summary.json"],
            "metadata_available": available["metadata.yaml"] or available["metadata.yml"],
            "parsed_config_available": available["parsed_config.yaml"] or available["parsed_config.yml"],
            "sequence_tracks_available": available["sequence_tracks.json"],
            "context_full_record_available": clip_id in context_index,
            "existing_mapping_candidate_count": len(clip_candidates),
            "semantic_correspondence_candidate_count": semantic_count,
            "explicit_origin_candidate_count": explicit_count,
            "highest_verification_level_found": _highest_level(levels),
            "evidence_status": evidence_status,
        })
    return rows


def _highest_level(levels: Iterable[str]) -> str:
    order = {
        "LEVEL_0_NUMERIC_ONLY": 0,
        "LEVEL_1_DECLARED_TIMESTAMP": 1,
        "LEVEL_2_EXPLICIT_ORIGIN_METADATA": 2,
        "LEVEL_3_SEMANTIC_CROSS_DOMAIN_PAIR": 3,
        "LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET": 4,
        "HISTORICAL_ACCEPTED": 5,
    }
    values = list(levels)
    if not values:
        return "NONE"
    return max(values, key=lambda value: order.get(value, -1))


def _sequence_range(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    try:
        value = _json(path)
    except Exception:
        return None, None
    values: list[int] = []

    def visit(child: Any) -> None:
        if isinstance(child, dict):
            for key, nested in child.items():
                lowered = str(key).lower()
                if "timestamp" in lowered and (lowered.endswith("_us") or lowered.endswith("s_us")):
                    values.extend(int(v) for v in _flatten_numbers(nested))
                elif isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(child, list):
            for nested in child:
                if isinstance(nested, (dict, list)):
                    visit(nested)

    visit(value)
    return (min(values), max(values)) if values else (None, None)


def _posthoc_support(path: Path, nurec_t0_us: int | None) -> dict[str, Any]:
    sequence_min, sequence_max = _sequence_range(path)
    if None in (sequence_min, sequence_max, nurec_t0_us):
        return {
            "status": "UNAVAILABLE",
            "sequence_min_us": sequence_min,
            "sequence_max_us": sequence_max,
            "query_min_us": nurec_t0_us,
            "query_max_us": nurec_t0_us + 5_000_000 if nurec_t0_us is not None else None,
            "mapping_verified": False,
            "support_horizon_us": 5_000_000,
        }
    query_max = nurec_t0_us + 5_000_000
    return {
        "status": "PASS" if sequence_min <= nurec_t0_us and sequence_max >= query_max else "CONTRADICTION",
        "sequence_min_us": sequence_min,
        "sequence_max_us": sequence_max,
        "query_min_us": nurec_t0_us,
        "query_max_us": query_max,
        "mapping_verified": False,
        "support_horizon_us": 5_000_000,
    }


def _load_structured_rows_for_known_file(path: Path) -> list[dict[str, Any]]:
    try:
        return [row for _, row in _structured_rows(path)]
    except Exception:
        return []


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = read_jsonl(Path(args.experiment_manifest))
    manifest = {str(row["clip_id"]): row for row in manifest_rows if row.get("clip_id")}
    current_ids = set(manifest)
    if len(current_ids) != EXPECTED_CLIP_COUNT:
        raise ValueError(f"EXPECTED_300_CLIPS_BUT_FOUND_{len(current_ids)}")
    historical_contract = Path(args.historical_contract).resolve()
    historical_match = Path(args.historical_match).resolve()
    context_full = Path(args.context_full)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    context_index: dict[str, dict[str, Any]] = {}
    context_candidates: list[dict[str, Any]] = []
    context_count = 0
    if context_full.is_file():
        context_index, context_count, context_candidates = parse_context_full(
            context_full, output_root / "context_full_time_evidence.csv", current_ids
        )
    else:
        (output_root / "context_full_time_evidence.csv").write_text("clip_id,json_path,field_name,value_type,count,min,max,first_values,unit_evidence,clock_domain_evidence,candidate_role\n", encoding="utf-8")

    data_roots = [Path(value) for value in args.data_root]
    # The context file is handled by its dedicated streaming parser, but stays
    # in explicit inventory with a precise scope status.
    output_root_resolved = output_root.resolve()
    output_mapping_resolved = Path(args.output_mapping).resolve()
    generic_files = [
        path
        for path in _iter_text_files([Path(args.repo_root) / "configs", *data_roots])
        if path.resolve() != output_mapping_resolved
        and output_root_resolved not in path.resolve().parents
    ]
    explicit_skip = {context_full.resolve()} if context_full.exists() else set()
    source_inventory, discovered_candidates = inspect_source_files(
        generic_files, current_ids, historical_contract, historical_match, explicit_skip
    )
    # Ensure explicitly named historical artifacts are inspected even if their
    # filename happens to be outside a root filter.
    for special in (historical_contract, historical_match):
        if special.exists() and special not in {Path(row["source_file"]).resolve() for row in source_inventory}:
            special_inventory, special_candidates = inspect_source_files(
                [special], current_ids, historical_contract, historical_match, set()
            )
            source_inventory.extend(special_inventory)
            discovered_candidates.extend(special_candidates)

    all_candidates = context_candidates + discovered_candidates
    # Deduplicate exact rows while preserving separate source evidence.
    unique_candidates: list[dict[str, Any]] = []
    seen_candidates: set[tuple[Any, ...]] = set()
    for candidate in all_candidates:
        key = (
            candidate.get("clip_id"), candidate.get("source_file"), candidate.get("source_record"),
            candidate.get("offset_us"), candidate.get("nurec_t0_us"), candidate.get("candidate_type"),
        )
        if key not in seen_candidates:
            seen_candidates.add(key)
            unique_candidates.append(candidate)
    all_candidates = unique_candidates

    # Add a direct, explicit inventory row for every known metadata file in the
    # current manifest, regardless of whether its filename contains "time".
    for clip_id, item in manifest.items():
        clip_dir = Path(str(item.get("nurec_clip_dir", "")))
        for name in KNOWN_METADATA_FILES:
            path = clip_dir / name
            if not path.is_file():
                continue
            if not any(row.get("source_file") == str(path) for row in source_inventory):
                source_inventory.append({
                    "source_file": str(path),
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                    "explicitly_requested": True,
                    "read_status": "READ_METADATA_SCOPE",
                    "current_clip_id_hit_count": 1,
                    "exact_field_hit_count": None,
                    "structured_candidate_count": 0,
                    "verified_candidate_count": 0,
                    "skip_reason": None,
                    "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
                })
            if name in {"rig_trajectories.json", "pose_record.json", "data_info.json", "sequence_tracks.json"}:
                if not any(row.get("clip_id") == clip_id and row.get("source_file") == str(path) for row in all_candidates):
                    all_candidates.append({
                        "clip_id": clip_id,
                        "physicalai_t0_us": None,
                        "nurec_t0_us": None,
                        "offset_us": None,
                        "scale": 1.0,
                        "source_file": str(path),
                        "source_record": None,
                        "json_path": "metadata_scope",
                        "mapping_type": "TIMELINE_ONLY",
                        "candidate_type": "KNOWN_METADATA_TIME_EVIDENCE",
                        "candidate_role": "NUREC_TIMELINE_ONLY",
                        "verification_flag": False,
                        "verification_level": "LEVEL_1_DECLARED_TIMESTAMP",
                        "verification_method": None,
                        "semantic_identity_source": None,
                        "semantic_identity": False,
                        "pair_count": 0,
                        "offset_residual_max_us": None,
                        "evidence_type": "LOCAL_METADATA_SCOPE",
                        "evidence_description": "Known per-clip metadata inspected; no PAI correspondence",
                        "current_300_member": True,
                        "candidate_status": "EXISTING_UNVERIFIED",
                        "verification_status": "UNVERIFIED",
                        "multiple_source_agreement": None,
                    })

    lightweight_inventory = _clip_metadata_inventory(manifest, context_index, all_candidates)

    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        if candidate.get("clip_id") in current_ids:
            by_clip[candidate["clip_id"]].append(candidate)

    prediction_index = load_t0_index(Path(args.prediction_jsonl) if args.prediction_jsonl else None)
    gt_index = load_t0_index(Path(args.ground_truth_jsonl) if args.ground_truth_jsonl else None)
    audit_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    final_status_counts: Counter[str] = Counter()
    verified_counts: Counter[str] = Counter()
    t0_consistent = 0
    t0_conflict = 0
    posthoc_pass = 0
    posthoc_conflict = 0
    sequence_available_count = 0

    for clip_id in sorted(current_ids):
        item = manifest[clip_id]
        t0 = verify_physicalai_t0(
            prediction_index.get(clip_id, []),
            gt_index.get(clip_id, []),
            _int(item.get("physicalai_t0_us")),
        )
        if t0["status"] == "CONSISTENT":
            t0_consistent += 1
        elif t0["status"] == "CONFLICTING":
            t0_conflict += 1
        clip_candidates = by_clip.get(clip_id, [])
        verified = [row for row in clip_candidates if row.get("verification_status") == "VERIFIED"]
        offsets = sorted({_int(row.get("offset_us")) for row in verified if _int(row.get("offset_us")) is not None})
        if t0["status"] == "CONFLICTING" or len(offsets) > 1:
            final_status = "CONFLICTING_EVIDENCE"
            conflict_rows.append({
                "clip_id": clip_id,
                "reason": "T0_CONFLICT" if t0["status"] == "CONFLICTING" else "MULTIPLE_VERIFIED_OFFSETS",
                "verified_sources": json.dumps([row.get("source_file") for row in verified]),
                "offsets_us": json.dumps(offsets),
            })
        elif offsets and t0["status"] == "CONSISTENT":
            offset = offsets[0]
            historical_present = any(row.get("verification_level") == "HISTORICAL_ACCEPTED" for row in verified)
            explicit_present = any(row.get("verification_level") == "LEVEL_2_EXPLICIT_ORIGIN_METADATA" for row in verified)
            pair_present = any(row.get("verification_level") == "LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET" for row in verified)
            if historical_present:
                final_status = "VERIFIED_REUSED_EXISTING"
                verified_counts["historical"] += 1
            elif explicit_present:
                final_status = "VERIFIED_REGENERATED_EXPLICIT_METADATA"
                verified_counts["explicit"] += 1
            elif pair_present:
                final_status = "VERIFIED_REGENERATED_ESTABLISHED_METHOD"
                verified_counts["semantic_pair"] += 1
            else:
                final_status = "MISSING_REQUIRED_EVIDENCE"
            if final_status.startswith("VERIFIED"):
                nurec_t0 = compute_nurec_t0(int(t0["physicalai_t0_us"]), offset)
                source_names = sorted({str(row.get("source_file")) for row in verified})
                pair_count = max(int(row.get("pair_count") or 0) for row in verified)
                residuals = [int(row["offset_residual_max_us"]) for row in verified if row.get("offset_residual_max_us") is not None]
                canonical_rows.append({
                    "clip_id": clip_id,
                    "physicalai_t0_us": t0["physicalai_t0_us"],
                    "nurec_t0_us": nurec_t0,
                    "offset_us": offset,
                    "scale": 1.0,
                    "mapping_type": "PER_CLIP_REBASE",
                    "verified": True,
                    "mapping_status": final_status,
                    "verification_level": "HISTORICAL_ACCEPTED" if historical_present else ("LEVEL_2_EXPLICIT_ORIGIN_METADATA" if explicit_present else "LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET"),
                    "verification_method": "ACCEPTED_HISTORICAL_FIVE_CLIP_CONTRACT" if historical_present else ("EQUIVALENT_MULTI_PAIR_CONSTANT_OFFSET" if pair_present else "EXPLICIT_ORIGIN_METADATA"),
                    "semantic_identity_source": ";".join(sorted({str(row.get("semantic_identity_source")) for row in verified if row.get("semantic_identity_source")})) or None,
                    "pair_count": pair_count,
                    "offset_residual_max_us": max(residuals) if residuals else (0 if historical_present else None),
                    "source": ";".join(source_names),
                    "per_clip_offset_rederived": False,
                })
        else:
            final_status = "MISSING_REQUIRED_EVIDENCE"
        if final_status not in FINAL_STATUSES:
            raise AssertionError(final_status)
        final_status_counts[final_status] += 1
        highest_level = _highest_level([row.get("verification_level", "LEVEL_0_NUMERIC_ONLY") for row in clip_candidates])
        level_rows.append({
            "clip_id": clip_id,
            "highest_verification_level_found": highest_level,
            "candidate_count": len(clip_candidates),
            "verified_candidate_count": len(verified),
            "semantic_pair_candidate_count": sum(1 for row in clip_candidates if row.get("semantic_identity") or row.get("candidate_role") == "SEMANTIC_FRAME_CORRESPONDENCE"),
            "explicit_origin_candidate_count": sum(1 for row in clip_candidates if row.get("verification_level") == "LEVEL_2_EXPLICIT_ORIGIN_METADATA"),
        })
        sequence_path = Path(str(item.get("sequence_tracks_path", "")))
        if sequence_path.is_file():
            sequence_available_count += 1
        nurec_t0 = next((row.get("nurec_t0_us") for row in canonical_rows if row["clip_id"] == clip_id), None)
        posthoc = _posthoc_support(sequence_path, nurec_t0)
        if posthoc["status"] == "PASS":
            posthoc_pass += 1
        elif posthoc["status"] == "CONTRADICTION":
            posthoc_conflict += 1
        posthoc_rows.append({"clip_id": clip_id, "sequence_tracks_available": sequence_path.is_file(), **posthoc})
        nodes = [
            {"id": "PAI_TIME_DOMAIN", "type": "clock_domain"},
            {"id": "NUREC_TIME_DOMAIN", "type": "clock_domain"},
            {"id": "prediction_t0", "type": "timestamp", "value_us": t0["physicalai_t0_us"]},
            {"id": "gt_t0", "type": "timestamp", "value_us": t0["physicalai_t0_us"] if t0["status"] == "CONSISTENT" else None},
        ]
        edges = [{"from": "prediction_t0", "to": "PAI_TIME_DOMAIN", "type": "SAME_CLOCK"}, {"from": "gt_t0", "to": "PAI_TIME_DOMAIN", "type": "SAME_CLOCK"}]
        if sequence_path.is_file():
            nodes.append({"id": "sequence_tracks", "type": "NuRec timeline", "path": str(sequence_path)})
            edges.append({"from": "sequence_tracks", "to": "NUREC_TIME_DOMAIN", "type": "DIAGNOSTIC_OVERLAP"})
        for index, candidate in enumerate(clip_candidates):
            if candidate.get("offset_us") is None and candidate.get("candidate_type") not in {"KNOWN_METADATA_TIME_EVIDENCE", "CONTEXT_FULL_TIME_EVIDENCE"}:
                continue
            node_id = f"candidate_{index}"
            nodes.append({"id": node_id, "type": candidate.get("candidate_type"), "source_file": candidate.get("source_file"), "verification_level": candidate.get("verification_level")})
            edge_type = "EXPLICIT_OFFSET" if candidate.get("verification_status") == "VERIFIED" else ("SEMANTIC_SAME_EVENT" if candidate.get("pair_count", 0) else "DIAGNOSTIC_OVERLAP")
            edges.append({"from": node_id, "to": "NUREC_TIME_DOMAIN", "type": edge_type, "offset_us": candidate.get("offset_us"), "pair_count": candidate.get("pair_count", 0)})
        graph_rows.append({"clip_id": clip_id, "nodes": nodes, "edges": edges, "mapping_status": final_status, "proof_rule": "EXPLICIT_OFFSET_OR_AT_LEAST_TWO_SEMANTIC_SAME_EVENT_EDGES"})
        local_sources = sorted({str(row.get("source_file")) for row in clip_candidates})
        if final_status.startswith("VERIFIED"):
            missing_requirement = "NONE"
            minimal_file = None
            why_needed = "No additional time-mapping file required for current evidence state"
        else:
            missing_requirement = "AT_LEAST_TWO_SEMANTIC_PAI_NUREC_TIME_PAIRS_OR_COMPLETE_EXPLICIT_ORIGIN_METADATA"
            minimal_file = "UNKNOWN_LIGHTWEIGHT_SEMANTIC_TIME_EVIDENCE"
            why_needed = "Current local timestamps identify domains/timelines but do not prove PAI-to-NuRec semantic identity"
        plan_rows.append({
            "clip_id": clip_id,
            "local_sources_checked": ";".join(local_sources),
            "highest_verification_level_found": highest_level,
            "missing_requirement": missing_requirement,
            "minimal_candidate_file": minimal_file,
            "why_needed": why_needed,
            "full_clip_download_required": False,
            "sequence_tracks_download_needed_for_time_mapping": False,
            "mapping_status": final_status,
            "sequence_tracks_available": sequence_path.is_file(),
        })
        audit_rows.append({
            "clip_id": clip_id,
            "final_status": final_status,
            "physicalai_t0_us": t0["physicalai_t0_us"],
            "physicalai_t0_status": t0["status"],
            "candidate_count": len(clip_candidates),
            "verified_candidate_count": len(verified),
            "offset_us": next((row.get("offset_us") for row in canonical_rows if row["clip_id"] == clip_id), None),
            "nurec_t0_us": nurec_t0,
            "highest_verification_level_found": highest_level,
            "sequence_tracks_available": sequence_path.is_file(),
            "posthoc_status": posthoc["status"],
            "posthoc_support_horizon_us": 5_000_000,
            "posthoc_is_verification": False,
        })

    # Regression guard for the one accepted mapping already in the canonical contract.
    regression = True
    expected = {"physicalai_t0_us": 5_100_000, "nurec_t0_us": 3_038_753_000, "offset_us": 3_033_653_000}
    actual = next((row for row in canonical_rows if row["clip_id"] == "028508ba-ef59-48d3-a95b-94eb92e3b063"), None)
    if actual is None or any(actual[key] != value for key, value in expected.items()):
        regression = False

    source_fields = ["source_file", "exists", "size_bytes", "suffix", "explicitly_requested", "read_status", "current_clip_id_hit_count", "exact_field_hit_count", "structured_candidate_count", "verified_candidate_count", "skip_reason", "evidence_origin"]
    _write_csv(output_root / "explicit_source_inventory.csv", source_inventory, source_fields)
    _write_csv(output_root / "lightweight_time_evidence_inventory.csv", lightweight_inventory, list(lightweight_inventory[0]))
    candidate_fields = [
        "clip_id", "physicalai_t0_us", "nurec_t0_us", "offset_us", "scale", "source_file", "source_record", "json_path",
        "mapping_type", "candidate_type", "candidate_role", "verification_flag", "verification_level", "verification_method",
        "semantic_identity_source", "semantic_identity", "pair_count", "offset_residual_max_us", "evidence_type", "evidence_description",
        "current_300_member", "candidate_status", "verification_status", "multiple_source_agreement",
    ]
    _write_csv(output_root / "local_time_mapping_source_inventory.csv", source_inventory, source_fields)
    _write_csv(output_root / "time_mapping_candidate_registry.csv", all_candidates, candidate_fields)
    _write_csv(output_root / "time_mapping_verification_levels.csv", level_rows, list(level_rows[0]))
    _write_csv(output_root / "time_mapping_semantic_pairs.csv", [row for row in all_candidates if row.get("semantic_identity") and row.get("pair_count", 0)], candidate_fields)
    _write_csv(output_root / "time_mapping_per_clip_audit.csv", audit_rows, list(audit_rows[0]))
    _write_csv(output_root / "time_mapping_conflicts.csv", conflict_rows, ["clip_id", "reason", "verified_sources", "offsets_us"])
    _write_csv(output_root / "time_mapping_posthoc_consistency.csv", posthoc_rows, list(posthoc_rows[0]))
    _write_csv(output_root / "full300_time_mapping_minimal_data_plan.csv", plan_rows, list(plan_rows[0]))
    with (output_root / "time_evidence_graph.jsonl").open("w", encoding="utf-8") as handle:
        for row in graph_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _write_json(output_root / "historical_mapping_method.json", {
        "HISTORICAL_MAPPING_METHOD": "PER_CLIP_REBASE_VALIDATED_BY_EXPLICIT_PAI_NUREC_POSE_TIMELINE",
        "HISTORICAL_IDENTITY_SOURCE": "same clip UUID plus accepted PAI egomotion and NuRec rig trajectory artifacts",
        "HISTORICAL_MIN_PAIR_COUNT": HISTORICAL_MIN_PAIR_COUNT,
        "HISTORICAL_OFFSET_CONSISTENCY_RULE": "offset_start_us == offset_end_us exactly; scale=1.0; relative durations agree; numerical pose validation is independent support",
        "historical_artifacts_immutable": True,
    })
    level_counts = Counter(row["highest_verification_level_found"] for row in level_rows)
    verified_total = sum(1 for row in canonical_rows if row["verified"])
    summary = {
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "explicit_lightweight_source_count": sum(1 for row in source_inventory if row.get("read_status") in {"READ", "READ_METADATA_SCOPE", "READ_BY_DEDICATED_PARSER"}),
        "context_full_record_count": context_count,
        "existing_verified_mapping_count": final_status_counts["VERIFIED_REUSED_EXISTING"],
        "regenerated_explicit_metadata_count": final_status_counts["VERIFIED_REGENERATED_EXPLICIT_METADATA"],
        "regenerated_semantic_pair_count": final_status_counts["VERIFIED_REGENERATED_ESTABLISHED_METHOD"],
        "total_verified_mapping_count": verified_total,
        "conflicting_mapping_count": final_status_counts["CONFLICTING_EVIDENCE"],
        "missing_mapping_count": final_status_counts["MISSING_REQUIRED_EVIDENCE"],
        "time_mapping_coverage_rate": verified_total / EXPECTED_CLIP_COUNT,
        "time_mapping_full300_status": "FULL300_VERIFIED" if verified_total == EXPECTED_CLIP_COUNT else "INCOMPLETE",
        "level_0_numeric_only_count": level_counts["LEVEL_0_NUMERIC_ONLY"],
        "level_1_declared_timestamp_count": level_counts["LEVEL_1_DECLARED_TIMESTAMP"],
        "level_2_explicit_origin_metadata_count": level_counts["LEVEL_2_EXPLICIT_ORIGIN_METADATA"],
        "level_3_semantic_cross_domain_pair_count": level_counts["LEVEL_3_SEMANTIC_CROSS_DOMAIN_PAIR"],
        "level_4_multi_pair_constant_offset_count": level_counts["LEVEL_4_MULTI_PAIR_CONSTANT_OFFSET"],
        "historical_accepted_count": level_counts["HISTORICAL_ACCEPTED"],
        "physicalai_t0_consistent_count": t0_consistent,
        "physicalai_t0_conflict_count": t0_conflict,
        "sequence_tracks_available_count": sequence_available_count,
        "time_mapping_and_sequence_ready_count": sum(1 for row in audit_rows if row["final_status"].startswith("VERIFIED") and row["sequence_tracks_available"]),
        "currently_runnable_cf_ttc_clip_count": sum(1 for row in audit_rows if row["final_status"].startswith("VERIFIED") and row["sequence_tracks_available"]),
        "post_hoc_sequence_consistency_pass_count": posthoc_pass,
        "post_hoc_sequence_consistency_conflict_count": posthoc_conflict,
        "historical_mapping_regression": not regression,
        "minimal_additional_data_clip_count": sum(1 for row in plan_rows if not row["mapping_status"].startswith("VERIFIED")),
        "full_clip_download_required_count": 0,
        "per_clip_offset_rederived_for_historical_clips": False,
        "obstacle_geometry_block": "CLOSED",
        "remaining_blockers": ["AT_LEAST_TWO_SEMANTIC_PAI_NUREC_TIME_PAIRS_OR_COMPLETE_EXPLICIT_ORIGIN_METADATA_FOR_UNMAPPED_CLIPS"] if verified_total < EXPECTED_CLIP_COUNT else [],
        "context_full_parser": "STREAMING_ONE_JSONL_RECORD_AT_A_TIME",
        "sequence_tracks_role": "DIAGNOSTIC_ONLY",
        "download_performed": False,
    }
    _write_json(output_root / "time_mapping_full300_summary.json", summary)

    mapping_path = Path(args.output_mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        for row in canonical_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment-manifest", type=Path, default=Path("configs/nurec_cf_ttc_full300_manifest.jsonl"))
    parser.add_argument("--historical-contract", type=Path, default=Path("configs/nurec_coordinate_time_contract_5clip.jsonl"))
    parser.add_argument("--historical-match", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1\pai_nurec_time_match.csv"))
    parser.add_argument("--prediction-jsonl", type=Path, default=Path(r"D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl"))
    parser.add_argument("--ground-truth-jsonl", type=Path, default=Path(r"D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl"))
    parser.add_argument("--context-full", type=Path, default=Path(r"D:\300_clip_nurec\01_context\full\nurec_context_full_300.jsonl"))
    parser.add_argument("--data-root", type=Path, action="append", default=[
        Path(r"D:\300_clip_nurec\05_data_contract"),
        Path(r"D:\300_clip_nurec\hf_probe"),
        Path(r"D:\300_clip_nurec\01_context"),
        Path(r"D:\300_clip_nurec\00_raw"),
    ])
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\time_mapping_full300_exhaustive_v2"))
    parser.add_argument("--output-mapping", type=Path, default=Path("configs/nurec_time_contract_full300.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
