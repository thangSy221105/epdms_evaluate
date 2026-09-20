"""Audit CF/TTC observation readiness from minimal NuRec inputs.

This audit deliberately consumes only ``sequence_tracks.json`` and an existing
verified per-clip time mapping.  It never derives a new offset, downloads data,
or treats the absence of an object row as a confirmed empty frame.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.prepare_nurec_obstacles import (
        CF_TOLERANCE_US,
        TTC_TOLERANCE_US,
        build_scorer_query_grid,
        classify_scorer_queries,
        load_sequence_tracks,
        normalize_sequence_tracks,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.prepare_nurec_obstacles import (
        CF_TOLERANCE_US,
        TTC_TOLERANCE_US,
        build_scorer_query_grid,
        classify_scorer_queries,
        load_sequence_tracks,
        normalize_sequence_tracks,
    )


EXPECTED_CLIP_COUNT = 300
EXPECTED_CF_QUERIES_PER_CLIP = 41
EXPECTED_TTC_QUERIES_PER_CLIP = 51
PHYSICALAI_T0_DEFAULT_US = 5_100_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def _clip_index(rows: list[dict[str, Any]], source_name: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Index experiment records without treating repeated prediction conditions as duplicates."""
    index: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        if not clip_id:
            errors.append(f"{source_name}:missing_clip_id")
            continue
        t0 = row.get("t0_us")
        record = index.setdefault(clip_id, {"clip_id": clip_id, "t0_values": set(), "record_count": 0})
        record["record_count"] += 1
        if t0 is not None:
            try:
                record["t0_values"].add(int(t0))
            except (TypeError, ValueError):
                errors.append(f"{source_name}:{clip_id}:invalid_t0_us")
    return index, errors


def load_experiment_clip_index(prediction_jsonl: Path, ground_truth_jsonl: Path) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    prediction, prediction_errors = _clip_index(_read_jsonl(prediction_jsonl), "prediction")
    ground_truth, gt_errors = _clip_index(_read_jsonl(ground_truth_jsonl), "ground_truth")
    clip_ids = sorted(set(prediction) | set(ground_truth))
    return clip_ids, prediction, ground_truth, prediction_errors + gt_errors


def load_verified_time_mapping(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load only explicit verified rows; never infer a row from timestamps."""
    mapping: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row_number, row in enumerate(_read_jsonl(path), 1):
        clip_id = str(row.get("clip_id") or "")
        if not clip_id:
            errors.append(f"line_{row_number}:missing_clip_id")
            continue
        required = ("physicalai_t0_us", "nurec_t0_us", "offset_us")
        numeric = all(isinstance(row.get(field), int) and not isinstance(row.get(field), bool) for field in required)
        verified = row.get("verified") is True
        source = str(row.get("source") or "")
        if clip_id in mapping:
            prior = mapping[clip_id]
            if any(prior.get(field) != row.get(field) for field in required) or prior.get("verified") != verified:
                errors.append(f"{clip_id}:CONFLICTING_DUPLICATE_MAPPING")
            continue
        mapping[clip_id] = {
            **row,
            "mapping_valid": bool(numeric and verified and source),
            "mapping_status": "VERIFIED_REUSED_ARTIFACT" if numeric and verified and source else "INVALID_UNVERIFIED",
        }
        if not numeric or not verified or not source:
            errors.append(f"{clip_id}:mapping_not_verified")
    return mapping, errors


def build_full300_manifest(
    clip_ids: list[str],
    prediction_index: dict[str, dict[str, Any]],
    ground_truth_index: dict[str, dict[str, Any]],
    nurec_root: Path | list[Path],
    time_mapping: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one manifest/inventory row per experiment clip, including missing rows."""
    roots = [nurec_root] if isinstance(nurec_root, Path) else list(nurec_root)
    manifest: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        prediction = prediction_index.get(clip_id, {})
        ground_truth = ground_truth_index.get(clip_id, {})
        t0_values = set(prediction.get("t0_values", set())) | set(ground_truth.get("t0_values", set()))
        physicalai_t0 = next(iter(t0_values)) if len(t0_values) == 1 else None
        clip_dir = roots[0] / clip_id
        sequence_path = clip_dir / "sequence_tracks.json"
        for candidate_root in roots:
            candidate_dir = candidate_root / clip_id
            candidate_path = candidate_dir / "sequence_tracks.json"
            if candidate_path.is_file():
                clip_dir = candidate_dir
                sequence_path = candidate_path
                break
        mapping = time_mapping.get(clip_id)
        mapping_artifact_valid = bool(mapping and mapping.get("mapping_valid"))
        mapping_t0_matches = bool(mapping_artifact_valid and physicalai_t0 is not None and mapping.get("physicalai_t0_us") == physicalai_t0)
        mapping_valid = mapping_t0_matches
        prediction_exists = clip_id in prediction_index
        gt_exists = clip_id in ground_truth_index
        sequence_exists = sequence_path.is_file()
        reasons: list[str] = []
        if not prediction_exists:
            reasons.append("PREDICTION_MISSING")
        if not gt_exists:
            reasons.append("GT_MISSING")
        if not sequence_exists:
            reasons.append("SEQUENCE_TRACKS_MISSING")
        if not mapping_valid:
            reasons.append("TIME_MAPPING_T0_MISMATCH" if mapping_artifact_valid and not mapping_t0_matches else "TIME_MAPPING_MISSING")
        stage1_ready = bool(prediction_exists and gt_exists and sequence_exists and mapping_valid and physicalai_t0 is not None)
        if physicalai_t0 is None:
            reasons.append("T0_MISSING_OR_CONFLICTING")
        record = {
            "clip_id": clip_id,
            "physicalai_t0_us": physicalai_t0,
            "nurec_t0_us": mapping.get("nurec_t0_us") if mapping_valid else None,
            "offset_us": mapping.get("offset_us") if mapping_valid else None,
            "offset_reused": bool(mapping_valid),
            "per_clip_offset_rederived": False,
            "time_mapping_verified": mapping_valid,
            "time_mapping_source": mapping.get("source") if mapping else None,
            "time_mapping_status": (mapping.get("mapping_status", "MISSING") if mapping_valid else ("INVALID_T0_MISMATCH" if mapping_artifact_valid else "MISSING")),
            "nurec_clip_dir": str(clip_dir),
            "sequence_tracks_path": str(sequence_path),
            "sequence_tracks_available": sequence_exists,
            "prediction_exists": prediction_exists,
            "ground_truth_exists": gt_exists,
            "stage1_input_ready": stage1_ready,
            "input_blockers": reasons,
        }
        manifest.append(record)
        input_rows.append({
            **record,
            "prediction_record_count": prediction.get("record_count", 0),
            "ground_truth_record_count": ground_truth.get("record_count", 0),
            "prediction_t0_values": sorted(prediction.get("t0_values", set())),
            "ground_truth_t0_values": sorted(ground_truth.get("t0_values", set())),
        })
        mapping_rows.append({
            "clip_id": clip_id,
            "physicalai_t0_us": physicalai_t0,
            "nurec_t0_us": mapping.get("nurec_t0_us") if mapping_valid else None,
            "offset_us": mapping.get("offset_us") if mapping_valid else None,
            "time_mapping_available": mapping_valid,
            "time_mapping_status": record["time_mapping_status"],
            "time_mapping_source": record["time_mapping_source"],
            "time_mapping_provenance": "EXISTING_VERIFIED_ARTIFACT" if mapping_valid else None,
            "per_clip_offset_rederived": False,
            "mapping_errors": ";".join("TIME_MAPPING_NOT_VERIFIED" for _ in [0]) if mapping and not mapping_valid else None,
        })
    return manifest, input_rows, mapping_rows


def _query_rows(clip_id: str, query_type: str, queries: list[int], rows: list[dict[str, Any]], tolerance_us: int) -> list[dict[str, Any]]:
    classified = classify_scorer_queries(queries, rows, tolerance_us=tolerance_us)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(classified):
        output.append({
            "clip_id": clip_id,
            "metric": query_type,
            "query_index": index,
            "query_timestamp_us": row["query_timestamp_us"],
            "nearest_object_timestamp_us": row["nearest_obstacle_timestamp_us"],
            "delta_us": row["obstacle_delta_us"],
            "object_count": row["object_count"],
            "observation_state": row["observation_state"],
            "confirmed_empty": False,
            "empty_semantics_status": "UNRESOLVED",
            "attestation_source": row["attestation_source"],
            "tolerance_us": tolerance_us,
            "source_file": "sequence_tracks.json",
        })
    return output


def _triage_row(clip_id: str, metric: str, query: int | None, problem: str, rows: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    timestamps = sorted({int(row["timestamp_us"]) for row in rows})
    minimum = timestamps[0] if timestamps else None
    maximum = timestamps[-1] if timestamps else None
    nearest = min(timestamps, key=lambda value: abs(value - query)) if query is not None and timestamps else None
    delta = abs(nearest - query) if nearest is not None and query is not None else None
    inside = bool(query is not None and minimum is not None and minimum <= query <= maximum)
    return {
        "clip_id": clip_id,
        "metric": metric,
        "query_timestamp_us": query,
        "nearest_object_timestamp_us": nearest,
        "delta_us": delta,
        "sequence_tracks_timestamp_min": minimum,
        "sequence_tracks_timestamp_max": maximum,
        "inside_object_timestamp_range": inside,
        "outside_object_timestamp_range": bool(query is not None and not inside),
        "classification": problem,
        "required_additional_evidence": reason,
        "empty_semantics_status": "UNRESOLVED",
    }


def _fallback_row(clip_id: str, problem_type: str, already: str, candidate: str, why: str) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "problem_type": problem_type,
        "already_available_files": already,
        "minimal_additional_file_candidate": candidate,
        "why_needed": why,
        "full_clip_download_required": False,
    }


def _coverage(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    objects = sum(row["observation_state"] == "OBJECTS_PRESENT" for row in rows)
    empty = sum(row["observation_state"] == "CONFIRMED_EMPTY" for row in rows)
    missing = sum(row["observation_state"] == "MISSING" for row in rows)
    total = len(rows)
    return {
        f"TOTAL_{prefix}_REQUIRED_QUERY_COUNT": total,
        f"TOTAL_{prefix}_OBJECT_PRESENT_COUNT": objects,
        f"TOTAL_{prefix}_CONFIRMED_EMPTY_COUNT": empty,
        f"TOTAL_{prefix}_MISSING_COUNT": missing,
        f"{prefix}_COMPLETE_OBSERVATION_RATE": (objects + empty) / total if total else 0.0,
    }


def run_audit(
    manifest: list[dict[str, Any]],
    output_root: Path,
    manifest_output: Path | None = None,
    expected_clip_count: int = EXPECTED_CLIP_COUNT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_output is not None:
        _write_jsonl(manifest_output, manifest)

    clip_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    ttc_rows: list[dict[str, Any]] = []
    triage: list[dict[str, Any]] = []
    fallback: dict[str, dict[str, Any]] = {}
    evaluated = 0
    sequence_available = 0
    mapping_available = 0
    stage1_ready = 0

    for item in manifest:
        clip_id = item["clip_id"]
        sequence_path = Path(item["sequence_tracks_path"])
        if item.get("sequence_tracks_available"):
            sequence_available += 1
        if item.get("time_mapping_verified"):
            mapping_available += 1
        if item.get("stage1_input_ready"):
            stage1_ready += 1
        base = {
            "clip_id": clip_id,
            "sequence_tracks_available": bool(item.get("sequence_tracks_available")),
            "time_mapping_available": bool(item.get("time_mapping_verified")),
            "stage1_input_ready": bool(item.get("stage1_input_ready")),
            "sequence_tracks_read_status": "NOT_ATTEMPTED",
            "sequence_error_count": 0,
            "evaluated": False,
            "nurec_t0_us": item.get("nurec_t0_us"),
            "offset_us": item.get("offset_us"),
            "per_clip_offset_rederived": False,
            "cf_required_query_count": 0,
            "cf_object_present_count": 0,
            "cf_confirmed_empty_count": 0,
            "cf_missing_count": 0,
            "ttc_required_query_count": 0,
            "ttc_object_present_count": 0,
            "ttc_confirmed_empty_count": 0,
            "ttc_missing_count": 0,
            "cf_ready": False,
            "ttc_ready": False,
            "readiness_status": "INPUT_BLOCKED",
            "blockers": list(item.get("input_blockers") or []),
        }
        if not item.get("stage1_input_ready"):
            if not item.get("sequence_tracks_available"):
                fallback[clip_id] = _fallback_row(
                    clip_id, "MISSING_SEQUENCE_TRACKS", "prediction_jsonl;ground_truth_jsonl",
                    "sequence_tracks.json", "Stage 1 needs the authoritative obstacle label timeline.",
                )
            if not item.get("time_mapping_verified"):
                fallback[clip_id] = _fallback_row(
                    clip_id, "MISSING_TIME_MAPPING", "prediction_jsonl;ground_truth_jsonl",
                    "verified per-clip time-mapping artifact", "A verified NuRec t0/offset is required; no offset is inferred.",
                ) if clip_id not in fallback else {
                    **fallback[clip_id],
                    "problem_type": "MISSING_SEQUENCE_TRACKS_AND_TIME_MAPPING",
                    "minimal_additional_file_candidate": "sequence_tracks.json + verified per-clip time-mapping artifact",
                    "why_needed": "Both Stage 1 inputs are unavailable; no full clip is required by this audit.",
                }
            for metric in ("CF", "TTC"):
                triage.append(_triage_row(clip_id, metric, None, "MISSING_TIME_MAPPING" if not item.get("time_mapping_verified") else "MISSING_SEQUENCE_TRACKS", [], "Obtain only the named minimal input; do not infer empty or time offset."))
            clip_rows.append(base)
            continue

        try:
            sequence_rows, sequence_errors, schema = load_sequence_tracks(sequence_path)
        except Exception as exc:  # defensive per-clip containment
            base["sequence_tracks_read_status"] = "READ_ERROR"
            base["blockers"].append(f"SEQUENCE_TRACKS_READ_ERROR:{type(exc).__name__}")
            fallback[clip_id] = _fallback_row(clip_id, "SEQUENCE_TRACKS_READ_ERROR", "verified time mapping", "sequence_tracks.json", str(exc))
            for metric in ("CF", "TTC"):
                triage.append(_triage_row(clip_id, metric, None, "OTHER", [], "Repair or replace the unreadable sequence_tracks.json."))
            clip_rows.append(base)
            continue

        base["sequence_tracks_read_status"] = "READ"
        base["sequence_error_count"] = len(sequence_errors)
        if sequence_errors:
            base["blockers"].append("SEQUENCE_TRACKS_ROW_ERRORS_NOT_SILENTLY_DROPPED")
            fallback[clip_id] = _fallback_row(clip_id, "SEQUENCE_TRACKS_ROW_ERRORS", "sequence_tracks.json;verified time mapping", "corrected sequence_tracks.json", "The audit does not evaluate a partially corrupted label file." )
            clip_rows.append(base)
            continue

        normalized = normalize_sequence_tracks(clip_id, sequence_rows)
        cf_queries, ttc_queries, grid = build_scorer_query_grid(int(item["nurec_t0_us"]), include_t0=True)
        current_cf = _query_rows(clip_id, "CF", cf_queries, normalized, CF_TOLERANCE_US)
        current_ttc = _query_rows(clip_id, "TTC", ttc_queries, normalized, TTC_TOLERANCE_US)
        cf_rows.extend(current_cf)
        ttc_rows.extend(current_ttc)
        evaluated += 1
        base["evaluated"] = True
        base["cf_required_query_count"] = len(current_cf)
        base["cf_object_present_count"] = sum(row["observation_state"] == "OBJECTS_PRESENT" for row in current_cf)
        base["cf_confirmed_empty_count"] = 0
        base["cf_missing_count"] = sum(row["observation_state"] == "MISSING" for row in current_cf)
        base["ttc_required_query_count"] = len(current_ttc)
        base["ttc_object_present_count"] = sum(row["observation_state"] == "OBJECTS_PRESENT" for row in current_ttc)
        base["ttc_confirmed_empty_count"] = 0
        base["ttc_missing_count"] = sum(row["observation_state"] == "MISSING" for row in current_ttc)
        base["cf_ready"] = base["cf_missing_count"] == 0
        base["ttc_ready"] = base["ttc_missing_count"] == 0
        base["readiness_status"] = "READY_OBJECT_EVIDENCE_ONLY" if base["cf_ready"] and base["ttc_ready"] else "OBSERVATION_MISSING"
        if base["cf_missing_count"] or base["ttc_missing_count"]:
            for row in current_cf + current_ttc:
                if row["observation_state"] != "MISSING":
                    continue
                timestamps = [int(value["timestamp_us"]) for value in normalized]
                minimum = min(timestamps) if timestamps else None
                maximum = max(timestamps) if timestamps else None
                query = int(row["query_timestamp_us"])
                inside = bool(minimum is not None and minimum <= query <= maximum)
                classification = "POSSIBLE_EMPTY_FRAME" if inside else "OUTSIDE_OBJECT_TIMELINE"
                triage.append(_triage_row(clip_id, row["metric"], query, classification, normalized, "Optional frame-level label evidence may distinguish empty from missing; object rows alone cannot."))
            fallback[clip_id] = _fallback_row(clip_id, "OBSERVATION_QUERY_MISSING", "sequence_tracks.json;verified time mapping", "authoritative frame-level label manifest or clipgt/obstacle.parquet", "Investigate only the missing queries; do not download the full NuRec clip." )
        clip_rows.append(base)

    cf_summary = _coverage(cf_rows, "CF")
    ttc_summary = _coverage(ttc_rows, "TTC")
    cf_ready_full = bool(len(manifest) == expected_clip_count and evaluated == expected_clip_count and cf_summary["TOTAL_CF_MISSING_COUNT"] == 0)
    ttc_ready_full = bool(len(manifest) == expected_clip_count and evaluated == expected_clip_count and ttc_summary["TOTAL_TTC_MISSING_COUNT"] == 0)
    cf_missing_clips = sorted({row["clip_id"] for row in cf_rows if row["observation_state"] == "MISSING"})
    ttc_missing_clips = sorted({row["clip_id"] for row in ttc_rows if row["observation_state"] == "MISSING"})
    fallback_rows = list(fallback.values())
    _write_csv(output_root / "full300_input_inventory.csv", [
        {key: value if not isinstance(value, (list, dict, set)) else json.dumps(value, ensure_ascii=False) for key, value in row.items()}
        for row in manifest
    ])
    _write_csv(output_root / "full300_time_mapping_inventory.csv", [
        {key: value if not isinstance(value, (list, dict, set)) else json.dumps(value, ensure_ascii=False) for key, value in row.items()}
        for row in [
            {
                "clip_id": item["clip_id"],
                "physicalai_t0_us": item.get("physicalai_t0_us"),
                "nurec_t0_us": item.get("nurec_t0_us"),
                "offset_us": item.get("offset_us"),
                "time_mapping_available": item.get("time_mapping_verified"),
                "time_mapping_status": item.get("time_mapping_status"),
                "time_mapping_source": item.get("time_mapping_source"),
                "time_mapping_provenance": "EXISTING_VERIFIED_ARTIFACT" if item.get("time_mapping_verified") else None,
                "per_clip_offset_rederived": False,
            }
            for item in manifest
        ]
    ])
    _write_csv(output_root / "full300_clip_readiness.csv", clip_rows)
    _write_csv(output_root / "full300_cf_queries.csv", cf_rows)
    _write_csv(output_root / "full300_ttc_queries.csv", ttc_rows)
    _write_csv(output_root / "cf_ttc_missing_query_triage.csv", triage)
    _write_csv(output_root / "minimal_fallback_data_plan.csv", fallback_rows)

    cf_summary_payload = {"expected_query_count": expected_clip_count * EXPECTED_CF_QUERIES_PER_CLIP, "evaluated_clip_count": evaluated, "full300_ready": cf_ready_full, **cf_summary}
    ttc_summary_payload = {"expected_query_count": expected_clip_count * EXPECTED_TTC_QUERIES_PER_CLIP, "evaluated_clip_count": evaluated, "full300_ready": ttc_ready_full, **ttc_summary}
    _write_json(output_root / "full300_cf_coverage_summary.json", cf_summary_payload)
    _write_json(output_root / "full300_ttc_coverage_summary.json", ttc_summary_payload)

    blockers: list[str] = []
    if sequence_available < expected_clip_count:
        blockers.append("SEQUENCE_TRACKS_MISSING_FOR_SOME_CLIPS")
    if mapping_available < expected_clip_count:
        blockers.append("VERIFIED_TIME_MAPPING_MISSING_FOR_SOME_CLIPS")
    if evaluated < expected_clip_count:
        blockers.append("FULL300_EVALUATION_INCOMPLETE")
    if cf_summary["TOTAL_CF_MISSING_COUNT"] or ttc_summary["TOTAL_TTC_MISSING_COUNT"]:
        blockers.append("OBSERVATION_QUERY_MISSING")
    blockers.append("LABEL_SET_EMPTY_SEMANTICS_UNRESOLVED")
    summary = {
        "BRANCH_SCOPE": "feat/cf-ttc-full300-minimal-readiness",
        "EXPECTED_CLIP_COUNT": expected_clip_count,
        "SEQUENCE_TRACKS_AVAILABLE_COUNT": sequence_available,
        "SEQUENCE_TRACKS_MISSING_COUNT": expected_clip_count - sequence_available,
        "TIME_MAPPING_AVAILABLE_COUNT": mapping_available,
        "TIME_MAPPING_MISSING_COUNT": expected_clip_count - mapping_available,
        "STAGE1_INPUT_READY_COUNT": stage1_ready,
        "FULL300_EVALUATED_CLIP_COUNT": evaluated,
        "EXPECTED_CF_QUERY_COUNT": expected_clip_count * EXPECTED_CF_QUERIES_PER_CLIP,
        "ACTUAL_CF_QUERY_COUNT": len(cf_rows),
        **cf_summary,
        "CF_DATA_READY_FULL_300": cf_ready_full,
        "EXPECTED_TTC_QUERY_COUNT": expected_clip_count * EXPECTED_TTC_QUERIES_PER_CLIP,
        "ACTUAL_TTC_QUERY_COUNT": len(ttc_rows),
        **ttc_summary,
        "TTC_DATA_READY_FULL_300": ttc_ready_full,
        "CLIPS_WITH_CF_MISSING": cf_missing_clips,
        "CLIPS_WITH_TTC_MISSING": ttc_missing_clips,
        "LABEL_SET_EMPTY_SEMANTICS_STATUS": "UNRESOLVED",
        "PHYSICAL_WORLD_OBSTACLE_COMPLETENESS": "NOT_CLAIMED",
        "FULL_CLIP_DOWNLOAD_REQUIRED_COUNT": sum(bool(row["full_clip_download_required"]) for row in fallback_rows),
        "MINIMAL_FALLBACK_CLIP_COUNT": len(fallback_rows),
        "PER_CLIP_OFFSET_REDERIVED": False,
        "OBSTACLE_GEOMETRY_BLOCK": "CLOSED",
        "AUTHORITATIVE_OBSTACLE_SOURCE": "sequence_tracks.json",
        "SEQUENCE_TRACKS_POSE_FRAME": "NCORE_LOCAL_WORLD",
        "SEQUENCE_TRACKS_POSE_FRAME_VERIFIED": True,
        "OBSTACLE_TRANSFORM_STATUS": "VERIFIED",
        "NORMALIZED_OBSTACLE_STATUS": "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES",
        "EMPTY_FRAMES_INFERRED_FROM_ABSENCE": False,
        "REMAINING_BLOCKERS": blockers,
        "RECOMMENDED_NEXT_STEP": "obtain only the minimal affected-clip inputs listed in minimal_fallback_data_plan.csv, then rerun this audit" if blockers[:-1] else "freeze full-300 CF/TTC obstacle readiness and move to scorer coordinate integration / DAC",
        "no_full_clip_download_performed": True,
    }
    contract = {
        "stage1_required_inputs": ["sequence_tracks.json", "verified per-clip time mapping"],
        "stage1_optional_inputs": ["rig_trajectories.json", "clipgt/obstacle.parquet", "clipgt/clip.parquet", "clipgt/association.parquet", "data_info.json", "metadata.yaml"],
        "query_grid_source": "scripts.prepare_nurec_obstacles.build_scorer_query_grid -> tools.epdms.observation_contract.build_ttc_projection_timestamps",
        "cf": {"queries_per_clip": EXPECTED_CF_QUERIES_PER_CLIP, "tolerance_us": CF_TOLERANCE_US, "includes_t0": True, "frequency_hz": 10.0, "horizon_s": 4.0},
        "ttc": {"queries_per_clip": EXPECTED_TTC_QUERIES_PER_CLIP, "tolerance_us": TTC_TOLERANCE_US, "horizon_s": 1.0, "step_s": 0.2},
        "classification": {"OBJECTS_PRESENT": "nearest object timestamp within tolerance", "MISSING": "otherwise", "CONFIRMED_EMPTY": "never inferred in Stage 1"},
        "time_policy": {"reuse_verified_per_clip_mapping": True, "global_offset_allowed": False, "offset_rederived": False},
        "readiness_rule": "all 300 clips evaluated and zero missing queries",
        "empty_semantics_status": "UNRESOLVED",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
    }
    _write_json(output_root / "full300_readiness_contract.json", contract)
    _write_json(output_root / "full300_readiness_final_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--ground-truth-jsonl", required=True)
    parser.add_argument("--nurec-root", required=True, action="append", help="Root containing one extracted clip directory per clip ID; repeat for existing pilot/full extraction roots")
    parser.add_argument("--time-mapping-jsonl", required=True, help="Existing verified per-clip mapping JSONL; never derived here")
    parser.add_argument("--manifest-output", default="configs/nurec_cf_ttc_full300_manifest.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pilot-regression-summary", default=None, help="Optional existing 5-clip regression summary to record without recomputing or changing full-300 counts")
    parser.add_argument("--expected-clip-count", type=int, default=EXPECTED_CLIP_COUNT)
    args = parser.parse_args()

    clip_ids, prediction, ground_truth, index_errors = load_experiment_clip_index(Path(args.prediction_jsonl), Path(args.ground_truth_jsonl))
    mapping, mapping_errors = load_verified_time_mapping(Path(args.time_mapping_jsonl))
    manifest, input_rows, mapping_rows = build_full300_manifest(clip_ids, prediction, ground_truth, [Path(value) for value in args.nurec_root], mapping)
    if index_errors:
        print("INPUT_INDEX_ERRORS=" + json.dumps(index_errors, ensure_ascii=False))
    if mapping_errors:
        print("TIME_MAPPING_INPUT_ERRORS=" + json.dumps(mapping_errors, ensure_ascii=False))
    summary = run_audit(manifest, Path(args.output_root), Path(args.manifest_output), args.expected_clip_count)
    if args.pilot_regression_summary:
        pilot = json.loads(Path(args.pilot_regression_summary).read_text(encoding="utf-8"))
        summary.update({
            "PILOT_REGRESSION_CLIP_COUNT": pilot.get("PILOT_CLIP_COUNT"),
            "PILOT_REGRESSION_CF_REQUIRED_QUERY_COUNT": pilot.get("TOTAL_CF_REQUIRED_QUERY_COUNT"),
            "PILOT_REGRESSION_CF_OBJECT_PRESENT_COUNT": pilot.get("TOTAL_CF_OBJECT_PRESENT_COUNT"),
            "PILOT_REGRESSION_CF_MISSING_COUNT": pilot.get("TOTAL_CF_MISSING_COUNT"),
            "PILOT_REGRESSION_TTC_REQUIRED_QUERY_COUNT": pilot.get("TOTAL_TTC_REQUIRED_QUERY_COUNT"),
            "PILOT_REGRESSION_TTC_OBJECT_PRESENT_COUNT": pilot.get("TOTAL_TTC_OBJECT_PRESENT_COUNT"),
            "PILOT_REGRESSION_TTC_MISSING_COUNT": pilot.get("TOTAL_TTC_MISSING_COUNT"),
            "PILOT_REGRESSION_SOURCE": str(args.pilot_regression_summary),
            "PILOT_REGRESSION_STATUS": "PASS" if pilot.get("TOTAL_CF_REQUIRED_QUERY_COUNT") == pilot.get("TOTAL_CF_OBJECT_PRESENT_COUNT") and pilot.get("TOTAL_TTC_REQUIRED_QUERY_COUNT") == pilot.get("TOTAL_TTC_OBJECT_PRESENT_COUNT") else "FAILED",
        })
        _write_json(Path(args.output_root) / "full300_readiness_final_summary.json", summary)
    print("FULL300_AUDIT_STATUS=" + ("READY" if summary["CF_DATA_READY_FULL_300"] and summary["TTC_DATA_READY_FULL_300"] else "PARTIAL"))
    for key in ("EXPECTED_CLIP_COUNT", "SEQUENCE_TRACKS_AVAILABLE_COUNT", "TIME_MAPPING_AVAILABLE_COUNT", "STAGE1_INPUT_READY_COUNT", "FULL300_EVALUATED_CLIP_COUNT", "ACTUAL_CF_QUERY_COUNT", "ACTUAL_TTC_QUERY_COUNT", "FULL_CLIP_DOWNLOAD_REQUIRED_COUNT", "MINIMAL_FALLBACK_CLIP_COUNT"):
        print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
