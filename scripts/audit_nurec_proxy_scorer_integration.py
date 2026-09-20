"""Readiness audit and bounded pilot for the NuRec safety proxy scorer.

This script is deliberately separate from the long-running evaluator.  It
loads the frozen time/coordinate/observation contracts, decorates rows in
memory, emits one record per logical condition, and stops before a full run if
authoritative DAC inputs are not production-ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.epdms.config import EvaluationConfig
from tools.epdms.io_jsonl import iter_jsonl
from tools.epdms.map_loader import inspect_clip_map_status
from tools.epdms.nurec_inputs import (
    decorate_inputs,
    load_coordinate_contract,
    load_observation_readiness,
    load_time_mapping,
)
from tools.epdms.schemas import EvaluationScoreRecord
from tools.epdms.score_record import evaluate_single_condition


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("clip_id")): row for row in iter_jsonl(path)}


def observation_rows(readiness: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for clip_id, item in sorted(readiness.items()):
        rows.append({
            "clip_id": clip_id,
            "cf_required_query_count": item.cf_required,
            "cf_object_present_count": item.cf_object_present,
            "cf_confirmed_empty_count": item.cf_confirmed_empty,
            "cf_missing_count": item.cf_missing,
            "cf_ready": item.cf_ready,
            "ttc_required_query_count": item.ttc_required,
            "ttc_object_present_count": item.ttc_object_present,
            "ttc_confirmed_empty_count": item.ttc_confirmed_empty,
            "ttc_missing_count": item.ttc_missing,
            "ttc_ready": item.ttc_ready,
            "observation_ready": item.observation_ready,
            "label_set_empty_semantics_status": "NOT_SUPPORTED_BY_UPSTREAM_EXPORT_CONTRACT",
            "physical_world_obstacle_completeness": "NOT_CLAIMED",
            "source": "full300_readiness_audit",
        })
    return rows


def build_map_inventory(filtered_dir: Path, clip_ids: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    statuses: Dict[str, Dict[str, Any]] = {}
    for clip_id in clip_ids:
        result = inspect_clip_map_status(filtered_dir, clip_id)
        statuses[clip_id] = result
        clip_dir = filtered_dir / clip_id / "clipgt"
        lane = clip_dir / "lane.parquet"
        intersection = clip_dir / "intersection_area.parquet"
        available = result.get("status") in {"OK", "PARTIAL", "NO_DRIVABLE_POLYGON"}
        for source_type, path in (("lane.parquet", lane), ("intersection_area.parquet", intersection)):
            rows.append({
                "clip_id": clip_id,
                "map_source_type": source_type,
                "path": str(path),
                "available": bool(path.is_file()),
                "coordinate_frame": "NCORE_LOCAL_WORLD" if path.is_file() else None,
                "georeference_available": False,
                "drivable_geometry_available": bool(result.get("valid_polygon_count", 0) > 0),
                "source_provenance": "local_filtered_context_clipgt; frame provenance retained from frozen NuRec contract",
                "map_status": result.get("status"),
                "usable_for_strict_scoring": bool(result.get("usable_for_strict_scoring", False)),
            })
    return rows, statuses


def make_blocker_inventory(contract: Dict[str, Any], readiness: Dict[str, Any], map_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    map_source_available = sum(1 for row in map_statuses.values() if row.get("status") in {"OK", "PARTIAL", "NO_DRIVABLE_POLYGON"})
    return {
        "profile": "nurec_safety_proxy_v1",
        "gates": [
            {
                "gate": "official_profile",
                "current_requirement": "official NAVSIM implementation",
                "current_input_state": "intentionally not implemented",
                "already_verified_by_prior_contract": False,
                "actual_missing_requirement": "official metric implementation",
                "required_code_change": "none in this branch",
                "semantic_change_required": False,
            },
            {
                "gate": "coordinate_contract",
                "current_requirement": "declared common frame and anchor contract",
                "current_input_state": "raw rows omit machine-readable declarations",
                "already_verified_by_prior_contract": True,
                "actual_missing_requirement": "adapter plumbing only",
                "required_code_change": "accepted frozen contract loader",
                "semantic_change_required": False,
            },
            {
                "gate": "observation_evidence",
                "current_requirement": "all CF/TTC queries observed or explicitly empty",
                "current_input_state": f"{sum(1 for item in readiness.values() if item.observation_ready)}/{len(readiness)} clips complete",
                "already_verified_by_prior_contract": False,
                "actual_missing_requirement": "317 missing query timestamps remain missing",
                "required_code_change": "emit explicit incomplete records",
                "semantic_change_required": False,
            },
            {
                "gate": "dac_map",
                "current_requirement": "authoritative drivable geometry with validated transform",
                "current_input_state": f"{map_source_available} clips have local geometry; map-frame transform is not integrated",
                "already_verified_by_prior_contract": False,
                "actual_missing_requirement": "production DAC transform validation",
                "required_code_change": "none until authoritative transform input exists",
                "semantic_change_required": False,
            },
        ],
        "coordinate_contract_id": contract.get("contract_id"),
        "full300_observation_count": len(readiness),
        "map_source_available_count": map_source_available,
    }


def validate_pilot(records: List[Dict[str, Any]], expected_count: int) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record.get("overall_score_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    scored = [row for row in records if row.get("overall_score_status") == "SCORED"]
    formula_errors = []
    range_errors = []
    for row in scored:
        values = [row.get(name) for name in ("collision_free_proxy", "ttc_proxy", "dac_proxy", "progress_gt_proxy", "future_comfort_proxy", "nurec_safety_proxy_v1")]
        if any(value is None or not (0.0 <= float(value) <= 1.0) for value in values):
            range_errors.append(row.get("record_key"))
        expected = float(row["collision_free_proxy"]) * float(row["dac_proxy"]) * (5.0 * float(row["ttc_proxy"]) + 5.0 * float(row["progress_gt_proxy"]) + 2.0 * float(row["future_comfort_proxy"])) / 12.0
        if not math.isclose(expected, float(row["nurec_safety_proxy_v1"]), rel_tol=1e-12, abs_tol=1e-12):
            formula_errors.append(row.get("record_key"))
    keys = [row.get("record_key") for row in records]
    return {
        "expected_record_count": expected_count,
        "actual_record_count": len(records),
        "unique_record_count": len(set(keys)),
        "no_record_dropped": len(records) == expected_count and len(set(keys)) == expected_count,
        "status_counts": status_counts,
        "scored_count": len(scored),
        "formula_validation_status": "PASS" if not formula_errors else "FAIL",
        "formula_errors": formula_errors,
        "range_validation_status": "PASS" if not range_errors else "FAIL",
        "range_errors": range_errors,
        "alpha_zero_condition_identity_preserved": len({row.get("record_key") for row in records if float(row.get("alpha", 0.0)) == 0.0}) == len([row for row in records if float(row.get("alpha", 0.0)) == 0.0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "epdms_300.json")
    parser.add_argument("--coordinate-contract", type=Path, default=REPO_ROOT / "configs" / "nurec_coordinate_contract_full300.json")
    parser.add_argument("--readiness-csv", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5\full300_clip_readiness.csv"))
    parser.add_argument("--time-mapping-jsonl", type=Path, default=REPO_ROOT / "configs" / "nurec_time_contract_full300.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\proxy_scorer_integration_v1"))
    parser.add_argument("--pilot-clips", type=int, default=5)
    args = parser.parse_args()

    config = EvaluationConfig.from_file(args.config)
    contract = load_coordinate_contract(args.coordinate_contract)
    readiness = load_observation_readiness(args.readiness_csv)
    time_mapping = load_time_mapping(args.time_mapping_jsonl)
    pred_rows = list(iter_jsonl(config.prediction_jsonl))
    gt_map = load_index(config.ground_truth_jsonl)
    ctx_map = load_index(config.context_jsonl)
    clip_ids = sorted(readiness)

    dump_json(args.output_dir / "scorer_blocker_inventory.json", make_blocker_inventory(contract, readiness, {}))
    write_csv(args.output_dir / "observation_readiness_full300.csv", observation_rows(readiness))

    map_rows, map_statuses = build_map_inventory(config.context_filtered_dir, clip_ids)
    write_csv(args.output_dir / "map_source_inventory.csv", map_rows)
    map_validation = []
    dac_rows = []
    for clip_id in clip_ids:
        map_res = map_statuses[clip_id]
        geometry = bool(map_res.get("usable_for_strict_scoring"))
        map_validation.append({
            "clip_id": clip_id,
            "source_frame": "NCORE_LOCAL_WORLD" if geometry else None,
            "target_frame": "EGO_AT_T0",
            "transform_status": "MAP_TRANSFORM_MISSING",
            "validation_status": "PROVENANCE_ONLY_NOT_SCORING_READY",
            "fitted_transform": False,
            "reason": "No production map transform is integrated in this branch",
        })
        dac_rows.append({
            "clip_id": clip_id,
            "map_source_status": map_res.get("status"),
            "drivable_geometry_available": geometry,
            "dac_status": "MAP_TRANSFORM_MISSING" if geometry else "MAP_MISSING",
            "dac_ready": False,
            "dac_proxy_policy": "null_when_unavailable",
            "source_frame": "NCORE_LOCAL_WORLD" if geometry else None,
            "target_frame": "EGO_AT_T0",
        })
    write_csv(args.output_dir / "map_transform_validation.csv", map_validation)
    write_csv(args.output_dir / "dac_readiness_full300.csv", dac_rows)
    dump_json(args.output_dir / "coordinate_contract_integration.json", {
        "status": "VERIFIED_FROZEN_ADAPTER",
        "contract": contract,
        "clip_count": len(clip_ids),
        "world_to_nre_applied_to_ego": False,
        "per_clip_offset_rederived": False,
        "source_rows_edited": False,
    })
    dump_json(args.output_dir / "scorer_blocker_inventory.json", make_blocker_inventory(contract, readiness, map_statuses))

    ready_clips = [clip_id for clip_id in clip_ids if readiness[clip_id].observation_ready]
    pilot_clip_ids = ready_clips[: max(1, min(args.pilot_clips, len(ready_clips)))]
    pred_by_clip: Dict[str, List[Dict[str, Any]]] = {clip_id: [] for clip_id in pilot_clip_ids}
    for row in pred_rows:
        clip_id = str(row.get("clip_id", ""))
        if clip_id in pred_by_clip:
            pred_by_clip[clip_id].append(row)

    records: List[Dict[str, Any]] = []
    for clip_id in pilot_clip_ids:
        item = readiness[clip_id]
        time_record = time_mapping.get(clip_id)
        if not time_record or not time_record.get("verified"):
            continue
        raw_gt = gt_map.get(clip_id)
        raw_context = ctx_map.get(clip_id)
        if raw_gt is None or raw_context is None:
            continue
        for pred in sorted(pred_by_clip[clip_id], key=lambda row: (str(row.get("mode")), float(row.get("alpha", 0.0)))):
            pred_i, gt_i, ctx_i = decorate_inputs(pred, raw_gt, raw_context, contract, time_record, item)
            try:
                record = evaluate_single_condition(
                    pred_row=pred_i,
                    context_row=ctx_i,
                    gt_row=gt_i,
                    vehicle=config.vehicle,
                    horizon_s=config.horizon_s,
                    frequency_hz=config.frequency_hz,
                    lane_polygons=[],
                    map_status="MAP_TRANSFORM_MISSING",
                    touch_is_collision=config.touch_is_collision,
                    ttc_horizon_s=config.ttc_horizon_s,
                    progress_stationary_threshold_m=config.progress_stationary_threshold_m,
                    strict_mode=True,
                    metric_profile="nurec_safety_proxy_v1",
                )
            except Exception as exc:
                record = EvaluationScoreRecord(
                    record_key=f"{clip_id}|{pred.get('mode', 'unknown')}|{pred.get('alpha', 0.0)}",
                    clip_id=clip_id,
                    mode=str(pred.get("mode", "unknown")),
                    alpha=float(pred.get("alpha", 0.0) or 0.0),
                    valid=False,
                    failure_stage="integration_pilot",
                    failure_type=type(exc).__name__,
                    failure_reason=str(exc),
                    metric_status="INPUT_ERROR",
                    overall_score_status="INPUT_ERROR",
                )
            record_dict = record.to_dict()
            record_dict["pilot_clip"] = True
            record_dict["physical_world_obstacle_completeness"] = "NOT_CLAIMED"
            record_dict["label_set_empty_semantics_status"] = "NOT_SUPPORTED_BY_UPSTREAM_EXPORT_CONTRACT"
            records.append(record_dict)

    write_jsonl(args.output_dir / "pilot_score_records.jsonl", records)
    pilot_validation = validate_pilot(records, len(pilot_clip_ids) * 16)
    dump_json(args.output_dir / "pilot_score_validation.json", pilot_validation)

    map_available = sum(1 for row in map_statuses.values() if row.get("status") in {"OK", "PARTIAL", "NO_DRIVABLE_POLYGON"})
    dump_json(args.output_dir / "proxy_scorer_readiness_summary.json", {
        "profile": "nurec_safety_proxy_v1",
        "expected_clip_count": len(clip_ids),
        "observation_ready_clip_count": len(ready_clips),
        "observation_incomplete_clip_count": len(clip_ids) - len(ready_clips),
        "cf_object_present_count": sum(item.cf_object_present for item in readiness.values()),
        "cf_missing_count": sum(item.cf_missing for item in readiness.values()),
        "ttc_object_present_count": sum(item.ttc_object_present for item in readiness.values()),
        "ttc_missing_count": sum(item.ttc_missing for item in readiness.values()),
        "map_source_available_clip_count": map_available,
        "dac_ready_clip_count": 0,
        "dac_not_ready_clip_count": len(clip_ids),
        "scorer_pilot_clip_count": len(pilot_clip_ids),
        "scorer_pilot_expected_record_count": len(pilot_clip_ids) * 16,
        "scorer_pilot_actual_record_count": len(records),
        "scorer_pilot_validation": pilot_validation,
        "full4800_run_executed": False,
        "full4800_expected_record_count": 4800,
        "full4800_actual_record_count": None,
        "official_navsim_profile_status": "NOT_IMPLEMENTED",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
        "label_set_empty_semantics_status": "NOT_SUPPORTED_BY_UPSTREAM_EXPORT_CONTRACT",
        "scorer_evaluator_semantics_changed": False,
        "remaining_blockers": ["AUTHORITATIVE_MAP_TRANSFORM_NOT_INTEGRATED", "13_CLIPS_HAVE_INCOMPLETE_OBSERVATION_EVIDENCE"],
        "recommended_next_step": "finish only authoritative map/DAC inputs; do not reopen time, obstacle geometry, or empty-frame semantics",
    })

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "observation_ready_clip_count": len(ready_clips),
        "observation_incomplete_clip_count": len(clip_ids) - len(ready_clips),
        "map_source_available_clip_count": map_available,
        "dac_ready_clip_count": 0,
        "pilot_clip_count": len(pilot_clip_ids),
        "pilot_record_count": len(records),
        "pilot_validation": pilot_validation,
    }, indent=2))


if __name__ == "__main__":
    main()
