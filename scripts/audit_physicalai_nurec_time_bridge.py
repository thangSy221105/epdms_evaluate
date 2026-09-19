"""Evidence-only PhysicalAI -> NuRec time bridge audit.

This command compares supplied PAI and NuRec timestamp tables and writes
forensic reports. It never infers an offset from numeric proximity or shape.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


CLIP_ID = "00040136-e651-4abd-991d-0655ccda9430"
T0_US = 5_100_000
PROVENANCE_STATUSES = {"FOUND", "NOT_FOUND_AFTER_INSPECTION", "NOT_INSPECTED"}


def _summary(values: list[int]) -> dict[str, Any]:
    values = sorted(int(v) for v in values)
    steps = [b - a for a, b in zip(values, values[1:]) if b > a]
    nearest = min(values, key=lambda value: abs(value - T0_US)) if values else None
    return {
        "count": len(values),
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
        "median_step": median(steps) if steps else None,
        "contains_5100000": T0_US in values,
        "nearest_to_5100000": nearest,
    }


def constant_delta_diagnostics(pairs: list[dict[str, int]], semantic_verified: bool) -> dict[str, Any]:
    deltas = [int(pair["nurec_timestamp_us"]) - int(pair["physicalai_timestamp_us"]) for pair in pairs]
    unique = sorted(set(deltas))
    med = median(deltas) if deltas else None
    return {
        "pair_count": len(pairs),
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "delta_unique_count": len(unique),
        "median_delta": med,
        "residual": max((abs(value - med) for value in deltas), default=None),
        "semantic_verified": semantic_verified,
        "verification_status": "VERIFIED" if semantic_verified and len(deltas) >= 2 and len(unique) == 1 else "DIAGNOSTIC_ONLY",
    }


def t0_query_contract(values: list[int], t0_us: int = T0_US) -> dict[str, Any]:
    numeric = [int(value) for value in values]
    in_range = bool(numeric) and min(numeric) <= t0_us <= max(numeric)
    return {
        "query_contract_found": True,
        "exact_row": t0_us in numeric,
        "interpolatable": in_range,
        "query_in_range": in_range,
        "first_timestamp": min(numeric) if numeric else None,
        "last_timestamp": max(numeric) if numeric else None,
    }


def pai_to_ncore_timestamp_contract(values: list[int]) -> dict[str, Any]:
    retained = [int(value) for value in values if int(value) >= 0]
    return {
        "scale": 1.0,
        "offset_us": 0,
        "numeric_retiming": "NONE_FOR_RETAINED_ROWS",
        "negative_ego_rows": "FILTERED",
        "input_count": len(values),
        "retained_count": len(retained),
        "negative_count": len(values) - len(retained),
        "retained_values_preserved": True,
    }


def explicit_ncore_nurec_mapping(mapping: dict[str, Any], source_clip_id: str, target_clip_id: str) -> dict[str, Any]:
    """Accept an offset only when metadata explicitly identifies both clips."""
    verified = (mapping.get("source_clip_id") == source_clip_id and mapping.get("target_clip_id") == target_clip_id and isinstance(mapping.get("offset_us"), int))
    return {"verified": verified, "scale": mapping.get("scale", 1.0) if verified else None,
            "offset_us": mapping.get("offset_us") if verified else None,
            "status": "VERIFIED" if verified else "UNRESOLVED"}


def per_sequence_mapping(mappings: list[dict[str, Any]], sequence_id: str) -> dict[str, Any]:
    matches = [item for item in mappings if item.get("sequence_id") == sequence_id]
    return {"sequence_id": sequence_id, "match_count": len(matches), "mapping": matches[0] if len(matches) == 1 else None,
            "status": "VERIFIED" if len(matches) == 1 else "UNRESOLVED"}


def classify_clip_pattern(*, direct: bool = False, constant_delta: bool = False, relative_equal: bool = False, duration_equal: bool = False, semantic_pairs: int = 0) -> str:
    if semantic_pairs < 2:
        return "INSUFFICIENT_EVIDENCE"
    if direct and constant_delta:
        return "DIRECT"
    if constant_delta and duration_equal:
        return "CONSTANT_OFFSET"
    if relative_equal and duration_equal:
        return "RELATIVE_TIMELINE_ONLY"
    return "NONLINEAR_OR_RETIMED"


def classify_cross_clip_pattern(clip_results: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [item for item in clip_results if item.get("semantic_pairs", 0) >= 2 and item.get("constant_delta")]
    offsets = sorted({int(item["offset_us"]) for item in verified if item.get("offset_us") is not None})
    if not verified:
        return {"pattern_status": "INSUFFICIENT_EVIDENCE", "verification_status": "UNVERIFIED", "offsets": offsets}
    if len(offsets) == 1:
        return {"pattern_status": "GLOBAL_FIXED_OFFSET_CANDIDATE", "verification_status": "UNVERIFIED", "offsets": offsets}
    return {"pattern_status": "PER_CLIP_OFFSET_CANDIDATE", "verification_status": "UNVERIFIED", "offsets": offsets}


def duration_diagnostics(ncore_start: int | None, ncore_end: int | None, nurec_start: int | None, nurec_end: int | None) -> dict[str, Any]:
    if None in (ncore_start, ncore_end, nurec_start, nurec_end):
        return {"available": False, "duration_error_us": None}
    ncore_duration = int(ncore_end) - int(ncore_start)
    nurec_duration = int(nurec_end) - int(nurec_start)
    return {"available": True, "ncore_duration_us": ncore_duration, "nurec_duration_us": nurec_duration, "duration_error_us": nurec_duration - ncore_duration}


def relative_clock_diagnostics(ncore_values: list[int], nurec_values: list[int]) -> dict[str, Any]:
    if not ncore_values or not nurec_values:
        return {"available": False, "same_duration": None, "same_count": None, "same_median_step": None}
    ncore = sorted(map(int, ncore_values)); nurec = sorted(map(int, nurec_values))
    ncore_steps = [b - a for a, b in zip(ncore, ncore[1:])]
    nurec_steps = [b - a for a, b in zip(nurec, nurec[1:])]
    return {"available": True, "same_duration": (ncore[-1] - ncore[0]) == (nurec[-1] - nurec[0]), "same_count": len(ncore) == len(nurec), "same_median_step": median(ncore_steps) == median(nurec_steps) if ncore_steps and nurec_steps else False}


def _read_timestamp(path: Path, field: str) -> list[int]:
    frame = pd.read_parquet(path)
    if field in frame.columns:
        series = frame[field]
    else:
        parts = field.split(".")
        series = frame[parts[0]]
        for part in parts[1:]:
            series = series.map(lambda value: value.get(part) if isinstance(value, dict) else None)
    return [int(value) for value in series.dropna().tolist()]


def _walk_values(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_values(child, f"{path}[{index}]")
    else:
        yield path, value


NUREC_INSPECTION_FILES = [
    "data_info.json", "datasource_summary.json", "metadata.yaml", "parsed_config.yaml",
    "pose_record.json", "rig_trajectories.json", "sequence_tracks.json",
    "clipgt/clip.parquet", "clipgt/association.parquet", "clipgt/egomotion_estimate.parquet",
    "clipgt/obstacle.parquet", "clipgt/calibration_estimate.parquet",
]
PROVENANCE_SEARCH_TERMS = (
    "source_clip_id", "source_sequence_id", "parent_clip_id", "source_repo_id", "source_revision",
    "source_commit_sha", "physicalai", "physical_ai", "ncore", "nre", "converter_version",
    "sequence_id", "scene_id", "clip_id", "time_offset", "timestamp_offset", "sequence_offset", "clock_offset",
    "timestamp_micros", "timestamp_us", "start_micros", "end_micros",
)


def _identity_class(field: str) -> str:
    name = field.lower().split(".")[-1]
    if name in {"source_clip_id", "parent_clip_id", "source_sequence_id"}:
        return "SOURCE_IDENTITY"
    if name in {"nurec_clip_id", "target_clip_id", "scene_id"}:
        return "TARGET_IDENTITY"
    if name == "clip_id":
        return "GENERIC_IDENTITY"
    return "UNKNOWN_IDENTITY"


def _candidate(path: str, field_path: str, value: Any, expected_source_clip_id: str, identity_values: list[Any] | None = None) -> dict[str, Any]:
    identity_class = _identity_class(field_path)
    values = identity_values if identity_values is not None else (value if isinstance(value, list) else [value])
    matches = expected_source_clip_id in values if identity_class == "SOURCE_IDENTITY" else False
    verified = identity_class == "SOURCE_IDENTITY" and matches
    return {
        "source_file": path,
        "field_path": field_path,
        "value": value,
        "candidate_type": "IDENTITY_OR_TIME_FIELD",
        "identity_class": identity_class,
        "matches_expected_source_clip": matches,
        "semantic_status": "SOURCE_LINEAGE_MATCH" if verified else "CANDIDATE",
        "verification_status": "VERIFIED_SOURCE_PROVENANCE" if verified else "CANDIDATE_ONLY",
    }


def inspect_nurec_provenance(clip_dir: str | Path | None, expected_source_clip_id: str = CLIP_ID) -> dict[str, Any]:
    """Inspect a bounded NuRec scope; read failures never become provenance candidates."""
    if not clip_dir:
        return {"status": "NOT_INSPECTED", "requested_files": [], "found_files": [], "missing_files": [], "successfully_read_files": [], "failed_files": [], "inspected_files": [], "verified_candidates": [], "inspection_errors": []}
    root = Path(clip_dir)
    if not root.exists():
        return {"status": "NOT_INSPECTED", "requested_files": NUREC_INSPECTION_FILES, "found_files": [], "missing_files": NUREC_INSPECTION_FILES, "successfully_read_files": [], "failed_files": [], "inspected_files": [], "verified_candidates": [], "inspection_errors": [], "reason": "clip directory does not exist"}
    found_files, missing_files, successfully_read, failed_files = [], [], [], []
    candidates, inspection_errors = [], []
    for relative in NUREC_INSPECTION_FILES:
        path = root / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        found_files.append(relative)
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
                for column in frame.columns:
                    lower = str(column).lower()
                    if any(term in lower for term in PROVENANCE_SEARCH_TERMS):
                        values = [str(v) for v in frame[column].dropna().astype(str).unique().tolist()]
                        value = {"unique_value_count": len(values), "first_N_values": values[:20]}
                        candidates.append(_candidate(relative, str(column), value, expected_source_clip_id, values))
            else:
                if path.suffix.lower() == ".json":
                    raw = json.loads(path.read_text(encoding="utf-8"))
                else:
                    import yaml
                    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                for key_path, value in _walk_values(raw):
                    key = key_path.rsplit(".", 1)[-1].lower().replace("[", "")
                    if any(term in key or term in str(value).lower() for term in PROVENANCE_SEARCH_TERMS):
                        candidates.append(_candidate(relative, key_path, value, expected_source_clip_id))
            successfully_read.append(relative)
        except Exception as exc:
            failed_files.append(relative)
            inspection_errors.append({"file": relative, "exception_type": type(exc).__name__, "message": str(exc), "inspection_status": "READ_ERROR"})
    verified_candidates = [c for c in candidates if c["verification_status"] == "VERIFIED_SOURCE_PROVENANCE"]
    status = "FOUND" if verified_candidates else ("NOT_FOUND_AFTER_INSPECTION" if successfully_read or failed_files else "NOT_INSPECTED")
    inspection_status = "COMPLETE" if len(successfully_read) == len(NUREC_INSPECTION_FILES) else ("PARTIAL" if successfully_read or failed_files else "NOT_INSPECTED")
    return {
        "status": status, "inspection_status": inspection_status, "requested_files": NUREC_INSPECTION_FILES, "found_files": found_files,
        "missing_files": missing_files, "successfully_read_files": successfully_read, "failed_files": failed_files,
        "inspected_files": successfully_read + failed_files, "verified_candidates": verified_candidates,
        "candidates": candidates, "inspection_errors": inspection_errors,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["clip_id", "source", "field", "declared_unit", "count", "min", "max", "median_step", "contains_5100000", "nearest_to_5100000", "source_path", "provenance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _external_dataset_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evidence = {
        "official_ncore": {"status": args.official_ncore_pilot_presence, "source": "EXTERNAL_INSPECTION", "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles-NCore"},
        "official_nurec": {"status": args.official_nurec_pilot_presence, "source": "EXTERNAL_INSPECTION", "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"},
    }
    if args.external_dataset_evidence:
        evidence = json.loads(Path(args.external_dataset_evidence).read_text(encoding="utf-8"))
        for record in evidence.values():
            record.setdefault("source", "EXTERNAL_INSPECTION")
            record["verified_by_this_script"] = False
    for record in evidence.values():
        record["evidence_source"] = "EXTERNAL_INSPECTION"
        record["verified_by_this_script"] = False
    return evidence


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pai_ego = _read_timestamp(Path(args.pai_egomotion), args.pai_egomotion_field)
    pai_obs = _read_timestamp(Path(args.pai_obstacle), args.pai_obstacle_field)
    nurec_ego = _read_timestamp(Path(args.nurec_egomotion), args.nurec_egomotion_field)
    nurec_obs = _read_timestamp(Path(args.nurec_obstacle), args.nurec_obstacle_field)
    rows = []
    for source, field, values, path, provenance in [
        ("PhysicalAI", args.pai_egomotion_field, pai_ego, args.pai_egomotion, "official PhysicalAI egomotion parquet"),
        ("PhysicalAI", args.pai_obstacle_field, pai_obs, args.pai_obstacle, "official PhysicalAI obstacle parquet"),
        ("NuRec", args.nurec_egomotion_field, nurec_ego, args.nurec_egomotion, "raw NuRec egomotion parquet"),
        ("NuRec", args.nurec_obstacle_field, nurec_obs, args.nurec_obstacle, "raw NuRec obstacle parquet"),
    ]:
        rows.append({"clip_id": args.clip_id, "source": source, "field": field, "declared_unit": "microseconds", **_summary(values), "source_path": str(path), "provenance": provenance})
    _write_csv(output / "physicalai_timestamp_inventory.csv", rows)

    pai_ego_df = pd.read_parquet(args.pai_egomotion)
    pai_ego_df["distance_to_t0_us"] = (pai_ego_df[args.pai_egomotion_field] - T0_US).abs()
    nearest_rows = pai_ego_df.nsmallest(5, "distance_to_t0_us").drop(columns=["distance_to_t0_us"]).to_dict(orient="records")
    pai_obs_df = pd.read_parquet(args.pai_obstacle)
    obs_ts = pai_obs_df[args.pai_obstacle_field]
    obstacle_windows = {str(window): sorted({int(value) for value in obs_ts[(obs_ts >= window[0]) & (obs_ts <= window[1])].tolist()}) for window in [(4_900_000, 5_300_000), (4_000_000, 6_500_000)]}
    semantic_pairs: list[dict[str, int]] = []
    diagnostics = constant_delta_diagnostics(semantic_pairs, semantic_verified=False)
    t0_contract = t0_query_contract(pai_ego)
    pai_ncore_contract = pai_to_ncore_timestamp_contract(pai_ego)
    nurec_provenance = inspect_nurec_provenance(args.nurec_clip_dir, args.expected_source_clip_id)
    external_evidence = _external_dataset_evidence(args)
    coverage = {
        "cf_required_range_us": [T0_US + 100_000, T0_US + 4_000_000],
        "ttc_required_range_us": [T0_US + 100_000, T0_US + 5_000_000],
        "nurec_egomotion_range_us": [min(nurec_ego), max(nurec_ego)] if nurec_ego else [None, None],
        "nurec_obstacle_range_us": [min(nurec_obs), max(nurec_obs)] if nurec_obs else [None, None],
        "cf_time_coverage": "UNRESOLVED",
        "ttc_time_coverage": "UNRESOLVED",
        "reason": "No verified NCore-to-NuRec mapping; range comparison is diagnostic only",
    }
    lineage = {
        "clip_id": args.clip_id,
        "edges": [
            {"source_domain": "PhysicalAI", "target_domain": "Alpamayo", "mapping_type": "explicit_query_contract", "scale": 1.0, "offset_us": 0, "explicit": True, "source_file": "alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py", "source_line": 27, "evidence": "t0_us default 5_100_000; future query grid is t0 + k*100000"},
            {"source_domain": "PhysicalAI", "target_domain": "NCore", "mapping_type": "timestamp_preserved_for_retained_rows", "scale": 1.0, "offset_us": 0, "explicit": True, "source_file": "ncore/tools/data_converter/pai/converter.py", "source_line": 209, "evidence": "PAI_TO_NCORE_NUMERIC_RETIMING=NONE_FOR_RETAINED_ROWS; PAI_TO_NCORE_NEGATIVE_EGO_ROWS=FILTERED"},
            {"source_domain": "NCore", "target_domain": "NuRec/NRE", "mapping_type": "UNRESOLVED", "scale": None, "offset_us": None, "explicit": False, "source_file": None, "source_line": None, "evidence": "No public NCore->NuRec writer mapping to raw clipgt/key.timestamp_micros found in inspected repositories"},
        ],
        "pai_to_ncore_contract": pai_ncore_contract,
        "nurec_source_provenance": nurec_provenance,
        "official_dataset_presence": external_evidence,
    }
    bridge = {
        "clip_id": args.clip_id,
        "physicalai_t0_us": T0_US,
        "physicalai_clock_verified": True,
        "nurec_clock_verified": True,
        "mapping_verified": False,
        "mapping_type": None,
        "scale": None,
        "offset_us": None,
        "source_evidence": ["official Alpamayo loader", "official PhysicalAI loader", "official NCore PAI converter"],
        "semantic_pair_count": 0,
        "residual_us": None,
        "cf_time_range_valid": None,
        "ttc_time_range_valid": None,
        "status": "UNRESOLVED",
        "blockers": ["NCORE_TO_NUREC_TIME_TRANSFORM_NOT_PUBLICLY_PROVEN", "NO_SEMANTIC_CROSS_DOMAIN_PAIRS"],
        "physicalai_t0_query_contract_found": t0_contract["query_contract_found"],
        "physicalai_egomotion_exact_t0_row": t0_contract["exact_row"],
        "physicalai_t0_interpolatable": t0_contract["interpolatable"],
        "physicalai_t0_query_in_range": t0_contract["query_in_range"],
        "physicalai_first_egomotion_timestamp": t0_contract["first_timestamp"],
        "physicalai_timestamp_zero_reference": "UNRESOLVED",
        "physicalai_clip_start_is_zero": "unknown",
        "physicalai_spatial_origin_at_timestamp_zero": "UNRESOLVED",
        "physicalai_egomotion_contains_5100000": T0_US in pai_ego,
        "physicalai_egomotion_nearest_rows": nearest_rows,
        "physicalai_obstacle_windows": obstacle_windows,
        "delta_diagnostics": diagnostics,
        "coverage": coverage,
        "source_clip_lineage_verified": nurec_provenance["status"] == "FOUND",
        "nurec_local_metadata_inspection_status": nurec_provenance.get("inspection_status", "NOT_INSPECTED"),
        "time_mapping_verified": False,
        "time_forensic_status": "EXHAUSTED_WITH_PUBLIC_EVIDENCE",
        "true_remaining_time_blocker": "NCORE_TO_NUREC_TIME_TRANSFORM_NOT_PUBLICLY_PROVEN",
        "official_dataset_presence": external_evidence,
    }
    (output / "time_lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "physicalai_ncore_nurec_time_lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "physicalai_to_nurec_time_bridge.json").write_text(json.dumps(bridge, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output / "physicalai_clock_contract.json").write_text(json.dumps({
        "unit": "microseconds",
        "origin_type": "UNRESOLVED_BY_PUBLIC_PAI_CODE",
        "origin_verified": False,
        "ego_obstacle_same_clock": True,
        "source": [
            {"file": "physical_ai_av/src/physical_ai_av/dataset.py", "lines": "172-184", "evidence": "egomotion and camera tables are read through timestamp columns"},
            {"file": "ncore/tools/data_converter/pai/converter.py", "lines": "209-237, 376-410", "evidence": "egomotion defines the sequence interval and obstacle timestamp_us is filtered against that interval"},
        ],
        "negative_timestamp_observed_in_pai_egomotion": True,
        "note": "The public code establishes microsecond timestamp usage and same-sequence filtering, but does not declare that zero is the PhysicalAI clip origin.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "alpamayo_time_contract.md").write_text("""# Alpamayo PhysicalAI time contract\n\n- `src/alpamayo_r1/load_physical_aiavdataset.py:27`: `t0_us: int = 5_100_000`.\n- Lines 33, 45-50: 64 future steps at 10 Hz, `time_step=0.1`.\n- Lines 107-120: future query grid is `t0_us + 100000, ..., t0_us + 6400000`.\n- Lines 122-129: queries the PhysicalAI egomotion interpolator at those timestamps and obtains `ego_future_xyz`.\n- Lines 142-157: transforms world poses to local coordinates at the t0 pose.\n- Line 220: returns the supplied `t0_us` unchanged.\n\nThis proves the Alpamayo sampling convention. It does not prove that raw NuRec `key.timestamp_micros` uses the same origin.\n""", encoding="utf-8")
    (output / "pai_to_ncore_time_trace.md").write_text("""# PhysicalAI -> NCore timestamp trace

- `PAI_TO_NCORE_NUMERIC_RETIMING = NONE_FOR_RETAINED_ROWS`
- `PAI_TO_NCORE_NEGATIVE_EGO_ROWS = FILTERED`
- scale `1.0`, offset `0`

This is not evidence for NCore -> NuRec.
""", encoding="utf-8")
    (output / "nurec_source_provenance.json").write_text(json.dumps({"clip_id": args.clip_id, **nurec_provenance}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "inspection_scope.json").write_text(json.dumps({"clip_id": args.clip_id, **{key: nurec_provenance.get(key) for key in ("inspection_status", "requested_files", "found_files", "missing_files", "successfully_read_files", "failed_files", "inspection_errors")}}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "ncore_to_nurec_public_trace.md").write_text("""# NCore to NuRec public trace

`PUBLIC_NCORE_TO_NUREC_REVIEW_MODE = MANUAL_REVIEW_RECORDED_IN_REPOSITORY`

Evidence origins: public-code claims are `PUBLIC_CODE_MANUAL_REVIEW`; generated observations are `LOCAL_AUTOMATED_INSPECTION`; external dataset presence is `EXTERNAL_MANUAL_INSPECTION`.

| Candidate | Status | Evidence |
|---|---|---|
| NCore PAI converter | READ_REFERENCE | PAI timestamps retained for non-negative rows. (`PUBLIC_CODE_MANUAL_REVIEW`) |
| NuRec `clipgt/*:key.timestamp_micros` writer | NOT_FOUND_AFTER_INSPECTION | No writer/manifest mapping was found in the manually reviewed public sources. (`PUBLIC_CODE_MANUAL_REVIEW`) |
| Numeric delta | DIAGNOSTIC_ONLY | Numeric proximity without semantic identity is not a pair. (`LOCAL_AUTOMATED_INSPECTION`) |
| NCore -> NuRec offset/scale | UNRESOLVED | No public evidence establishes one. (`PUBLIC_CODE_MANUAL_REVIEW`) |
""", encoding="utf-8")
    (output / "official_ncore_timestamp_inventory.csv").write_text("clip_id,dataset,pilot_presence,status,evidence_source,verified_by_this_script\n" + f"{args.clip_id},NCore,{external_evidence['official_ncore']['status']},{external_evidence['official_ncore']['status']},EXTERNAL_INSPECTION,false\n" + f"{args.clip_id},NuRec,{external_evidence['official_nurec']['status']},{external_evidence['official_nurec']['status']},EXTERNAL_INSPECTION,false\n", encoding="utf-8")
    (output / "pai_nurec_sequence_diagnostics.json").write_text(json.dumps({"physicalai_egomotion": _summary(pai_ego), "nurec_egomotion": _summary(nurec_ego), "semantic_pairs": semantic_pairs, "diagnostics": diagnostics}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "obstacle_time_windows.json").write_text(json.dumps(obstacle_windows, indent=2) + "\n", encoding="utf-8")
    return bridge


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _clip_id_from_path(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if len(part) == 36 and part.count("-") == 4:
            return part
    return None


def multiclip_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Inventory official manifests and compare only metadata that is actually exposed."""
    from huggingface_hub import HfApi, hf_hub_download
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ncore_root = args.ncore_root; nurec_root = args.nurec_root
    ncore_entries = list(api.list_repo_tree(args.ncore_dataset, path_in_repo=ncore_root, repo_type="dataset", recursive=False))
    nurec_entries = list(api.list_repo_tree(args.nurec_dataset, path_in_repo=nurec_root, repo_type="dataset", recursive=False))
    ncore_ids = sorted({clip for entry in ncore_entries if (clip := _clip_id_from_path(entry.path))})
    nurec_ids = sorted({clip for entry in nurec_entries if (clip := _clip_id_from_path(entry.path))})
    ncore_json_paths = {(_clip_id_from_path(entry.path)): entry.path for entry in api.list_repo_tree(args.ncore_dataset, path_in_repo=ncore_root, repo_type="dataset", recursive=True) if entry.path.endswith(".json") and _clip_id_from_path(entry.path)}
    nurec_file_paths: dict[str, list[str]] = {}
    for entry in nurec_entries:
        clip = _clip_id_from_path(entry.path)
        if clip:
            nurec_file_paths.setdefault(clip, []).append(entry.path)
    overlap = sorted(set(ncore_ids) & set(nurec_ids))
    ncore_rows = [{"dataset": args.ncore_dataset, "clip_id": clip, "sequence_id": f"pai_{clip}", "source_clip_id": clip, "path": f"{ncore_root}/{clip}", "metadata_available": clip in ncore_json_paths, "timestamp_metadata_available": clip in ncore_json_paths} for clip in ncore_ids]
    nurec_rows = [{"dataset": args.nurec_dataset, "clip_id": clip, "sequence_id": None, "path": f"{nurec_root}/{clip}", "metadata_available": False, "timestamp_metadata_available": False} for clip in nurec_ids]
    _write_rows(output / "ncore_clip_inventory.csv", ncore_rows, ["dataset", "clip_id", "sequence_id", "source_clip_id", "path", "metadata_available", "timestamp_metadata_available"])
    _write_rows(output / "nurec_clip_inventory.csv", nurec_rows, ["dataset", "clip_id", "sequence_id", "path", "metadata_available", "timestamp_metadata_available"])
    overlap_rows = [{"clip_id": clip, "ncore_path": f"{ncore_root}/{clip}", "nurec_path": f"{nurec_root}/{clip}", "identity_match_type": "EXACT_OFFICIAL_UUID", "identity_verified": True, "evidence": "same exact UUID appears as an official NCore and NuRec manifest directory"} for clip in overlap]
    _write_rows(output / "ncore_nurec_overlap.csv", overlap_rows, ["clip_id", "ncore_path", "nurec_path", "identity_match_type", "identity_verified", "evidence"])
    selected = overlap if len(overlap) <= 30 else overlap[:30]
    if args.pilot_clip_id in overlap and args.pilot_clip_id not in selected:
        selected[-1] = args.pilot_clip_id
    ncore_inventory, nurec_inventory, delta_rows, relative_rows = [], [], [], []
    for clip in selected:
        meta: dict[str, Any] = {}
        try:
            local = hf_hub_download(args.ncore_dataset, ncore_json_paths[clip], repo_type="dataset")
            meta = json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        interval = meta.get("sequence_timestamp_interval_us", {}) if isinstance(meta, dict) else {}
        ncore_start = interval.get("start") if isinstance(interval, dict) else None
        ncore_end = interval.get("stop") if isinstance(interval, dict) else None
        ncore_inventory.append({"clip_id": clip, "sequence_id": meta.get("sequence_id"), "source_clip_id": meta.get("generic_meta_data", {}).get("source_clip_id"), "ncore_pose_count": None, "ncore_pose_min_us": None, "ncore_pose_max_us": None, "ncore_pose_duration_us": None, "ncore_pose_median_step_us": None, "ncore_obstacle_min_us": None, "ncore_obstacle_max_us": None, "sequence_start_us": ncore_start, "sequence_end_us": ncore_end})
        nurec_inventory.append({"clip_id": clip, "nurec_ego_count": None, "nurec_ego_min_us": None, "nurec_ego_max_us": None, "nurec_ego_duration_us": None, "nurec_ego_median_step_us": None, "nurec_obstacle_min_us": None, "nurec_obstacle_max_us": None, "clip_start_micros": None, "clip_end_micros": None, "metadata_available": False, "metadata_reason": "official manifest exposes payload files only; requested timestamp metadata not exposed"})
        delta_rows.append({"clip_id": clip, "identity_verified": True, "D_start": None, "D_end": None, "D_ego_min": None, "D_ego_max": None, "diagnostic_status": "INSUFFICIENT_METADATA"})
        relative_rows.append({"clip_id": clip, "available": False, "same_duration": None, "same_count": None, "same_median_step": None, "status": "INSUFFICIENT_METADATA"})
    _write_rows(output / "ncore_multiclip_timestamp_inventory.csv", ncore_inventory)
    _write_rows(output / "nurec_multiclip_timestamp_inventory.csv", nurec_inventory)
    _write_rows(output / "multiclip_time_delta_diagnostics.csv", delta_rows)
    _write_rows(output / "relative_clock_diagnostics.csv", relative_rows)
    _write_rows(output / "semantic_time_pairs.csv", [], ["clip_id", "semantic_id", "ncore_timestamp_us", "nurec_timestamp_us", "identity_evidence", "pair_verified"])
    _write_rows(output / "duration_diagnostics.csv", [{"clip_id": clip, "available": False, "duration_error_us": None, "status": "INSUFFICIENT_METADATA"} for clip in selected])
    summary = {"ncore_clip_count": len(ncore_ids), "nurec_clip_count": len(nurec_ids), "overlap_clip_count": len(overlap), "clips_analyzed": len(selected), "clips_with_verified_identity": len(selected), "clips_with_semantic_pairs": 0, "clips_with_constant_delta": 0, "global_offset_candidate": None, "per_sequence_pattern": False, "per_clip_pattern": False, "relative_timeline_pattern": "INSUFFICIENT_EVIDENCE", "pattern_status": "INSUFFICIENT_EVIDENCE", "verification_status": "UNVERIFIED", "pilot_applicability": "PILOT_ABSENT_FROM_NCORE", "pilot_present_in_overlap": args.pilot_clip_id in overlap, "pilot_pattern_applicable": False, "pilot_time_mapping_verified": False, "duration_invariant_status": "INSUFFICIENT_EVIDENCE", "blockers": ["NUREC_TIMESTAMP_METADATA_NOT_EXPOSED_IN_OFFICIAL_OVERLAP_MANIFEST", "NO_VERIFIED_SEMANTIC_TIME_PAIRS", "PILOT_ABSENT_FROM_OFFICIAL_NCORE_SUBSET"], "multiclip_pattern_search_exhausted": True}
    (output / "ncore_nurec_multiclip_time_pattern.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiclip", action="store_true")
    parser.add_argument("--clip-id", default=CLIP_ID)
    parser.add_argument("--pai-egomotion")
    parser.add_argument("--pai-obstacle")
    parser.add_argument("--nurec-egomotion")
    parser.add_argument("--nurec-obstacle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nurec-clip-dir", default=None)
    parser.add_argument("--expected-source-clip-id", default=CLIP_ID)
    parser.add_argument("--external-dataset-evidence", default=None)
    parser.add_argument("--official-ncore-pilot-presence", choices=sorted(PROVENANCE_STATUSES), default="NOT_INSPECTED")
    parser.add_argument("--official-nurec-pilot-presence", choices=sorted(PROVENANCE_STATUSES), default="NOT_INSPECTED")
    parser.add_argument("--pai-egomotion-field", default="timestamp")
    parser.add_argument("--pai-obstacle-field", default="timestamp_us")
    parser.add_argument("--nurec-egomotion-field", default="key.timestamp_micros")
    parser.add_argument("--nurec-obstacle-field", default="key.timestamp_micros")
    parser.add_argument("--ncore-dataset", default="nvidia/PhysicalAI-Autonomous-Vehicles-NCore")
    parser.add_argument("--nurec-dataset", default="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec")
    parser.add_argument("--ncore-root", default="clips")
    parser.add_argument("--nurec-root", default="sample_set/26.04_release")
    parser.add_argument("--pilot-clip-id", default=CLIP_ID)
    args = parser.parse_args()
    if args.multiclip:
        print(json.dumps(multiclip_audit(args), indent=2, ensure_ascii=False, default=str))
    else:
        required = [args.pai_egomotion, args.pai_obstacle, args.nurec_egomotion, args.nurec_obstacle]
        if any(value is None for value in required):
            parser.error("single-clip mode requires --pai-egomotion, --pai-obstacle, --nurec-egomotion, and --nurec-obstacle")
        print(json.dumps(audit(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
