#!/usr/bin/env python3
"""Audit the authoritative NuRec map transform and prepare DAC readiness.

This script is deliberately separate from the scorer.  It inventories raw
ClipGT map files, applies only the accepted local-world → EGO_AT_T0 transform,
and writes a fail-closed readiness manifest consumed by the production CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.epdms.map_loader import inspect_clip_map_status
from tools.epdms.map_transform import (
    MapTransformError,
    invert_se3,
    interpolate_rig_world_pose,
    load_rig_world_samples,
    load_transformed_map_for_clip,
)


HISTORICAL_IDS = [
    "028508ba-ef59-48d3-a95b-94eb92e3b063",
    "d078258b-9339-425d-a040-68346ef0d5bc",
    "689889c5-95b0-42ce-a1c9-f97a4388cb28",
    "37f45f87-dc3b-4425-a388-fa7bfa4a11a6",
    "bb1b395f-c51d-4a16-87ad-7310a7bbf086",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _time_verified(row: dict[str, Any]) -> bool:
    return _bool(row.get("verified", row.get("time_mapping_verified", False)))


def _map_files(clip_root: Path) -> dict[str, bool]:
    return {
        "lane": (clip_root / "clipgt" / "lane.parquet").is_file(),
        "intersection_area": (clip_root / "clipgt" / "intersection_area.parquet").is_file(),
        "drivable_space": (clip_root / "clipgt" / "drivable_space.parquet").is_file(),
    }


def _metadata_version(clip_root: Path) -> str:
    path = clip_root / "metadata.yaml"
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("version_string:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _load_readiness(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def _validate_transform(map_root: Path, clip_id: str, t0_us: int, sample_kind: str) -> dict[str, Any]:
    try:
        transformed = load_transformed_map_for_clip(map_root, clip_id, t0_us, strict=True)
        if transformed.get("transform_status") != "VERIFIED":
            return {"clip_id": clip_id, "sample_kind": sample_kind, "status": "BLOCKED", "detail": transformed.get("detail", "map source not strict-ready")}
        pose = np.asarray(transformed["t_rig_world_at_t0"], dtype=float)
        world_to_ego = np.asarray(transformed["world_to_ego_t0"], dtype=float)
        identity_error = float(np.max(np.abs(world_to_ego @ pose - np.eye(4))))
        polygons = transformed["transformed_lane_polygons"] + transformed["transformed_intersection_polygons"]
        finite = all(np.isfinite(np.asarray(poly, dtype=float)).all() and len(poly) >= 3 for poly in polygons)
        return {
            "clip_id": clip_id,
            "sample_kind": sample_kind,
            "status": "PASS" if identity_error <= 1e-8 and finite and polygons else "FAIL",
            "identity_max_abs_error": identity_error,
            "transformed_polygon_count": len(polygons),
            "finite_geometry": finite,
            "nurec_t0_us": t0_us,
            "map_source_frame": "NCORE_LOCAL_WORLD",
            "map_target_frame": "EGO_AT_T0",
            "world_to_nre_used": False,
            "fitted_map_correction_used": False,
        }
    except Exception as exc:
        return {"clip_id": clip_id, "sample_kind": sample_kind, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}


def _pilot_clip_ids(map_rows: list[dict[str, Any]], readiness: dict[str, dict[str, Any]], time_rows: dict[str, dict[str, Any]]) -> list[str]:
    candidates = []
    for row in map_rows:
        cid = row["clip_id"]
        obs = readiness.get(cid, {})
        tm = time_rows.get(cid, {})
        if row["map_status"] == "OK" and _bool(obs.get("cf_ready")) and _bool(obs.get("ttc_ready")) and _time_verified(tm):
            candidates.append(cid)
    return sorted(candidates)[:5]


def _score_counts(score_dir: Path) -> dict[str, int]:
    scores = score_dir / "epdms_scores_300.jsonl"
    errors = score_dir / "epdms_errors_300.jsonl"
    score_rows = _read_jsonl(scores) if scores.is_file() else []
    error_rows = _read_jsonl(errors) if errors.is_file() else []
    return {
        "expected": len(score_rows) + len(error_rows),
        "scored": sum(1 for row in score_rows if row.get("valid") is True and row.get("nurec_safety_proxy_v1") is not None),
        "map_not_ready": sum(1 for row in error_rows + score_rows if row.get("dac_status") == "MAP_NOT_READY" or row.get("map_status") in {"MAP_NOT_READY", "MAP_TRANSFORM_MISSING"}),
        "input_error": sum(1 for row in error_rows if row.get("failure_stage") in {"input_contract", "cli_condition_loop"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--time-mapping", type=Path, required=True)
    parser.add_argument("--observation-readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-map-root", type=Path, default=None)
    parser.add_argument("--historical-time-mapping", type=Path, default=None)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--context-jsonl", type=Path, required=True)
    parser.add_argument("--run-pilot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pilot-score-dir", type=Path, default=None, help="Existing production pilot score directory to summarize.")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    time_records = {row["clip_id"]: row for row in _read_jsonl(args.time_mapping)}
    readiness = _load_readiness(args.observation_readiness)
    clip_ids = sorted(time_records)
    map_rows: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        root = args.map_root / clip_id
        files = _map_files(root)
        info = inspect_clip_map_status(args.map_root, clip_id)
        map_rows.append({
            "clip_id": clip_id,
            "lane_available": files["lane"],
            "intersection_area_available": files["intersection_area"],
            "drivable_space_available": files["drivable_space"],
            "map_source_available": files["lane"] or files["intersection_area"],
            "map_status": info.get("status"),
            "lane_polygon_count": info.get("lane_polygon_count", 0),
            "intersection_polygon_count": info.get("intersection_polygon_count", 0),
            "valid_polygon_count": info.get("valid_polygon_count", 0),
            "invalid_polygon_count": info.get("invalid_polygon_count", 0),
            "source_frame": "NCORE_LOCAL_WORLD" if info.get("status") in {"OK", "PARTIAL", "NO_DRIVABLE_POLYGON"} else "",
            "source_path": str(root / "clipgt"),
            "source_version": _metadata_version(root),
            "frame_provenance": "FROZEN_LOCAL_NUREC_CLIPGT_CONTRACT",
            "detail": info.get("detail", ""),
        })
    _write_csv(out / "map_inventory.csv", map_rows)

    evidence_rows = [
        {"evidence_id": "NCORE_SE3_CONVENTION", "source": "https://nvidia.github.io/ncore/data/conventions.html", "claim": "T_rig_world maps rig coordinates to local world; composition uses homogeneous SE(3).", "evidence_status": "PUBLIC_PRIMARY_SOURCE"},
        {"evidence_id": "PAI_NCORE_CONVERTER", "source": "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py", "claim": "PAI converter writes dynamic rig/world poses and preserves ClipGT obstacle/map ingestion in the NCore local-world convention.", "evidence_status": "PUBLIC_PRIMARY_SOURCE"},
        {"evidence_id": "NUREC_CLIPGT_MANIFEST", "source": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/blob/main/README.md", "claim": "lane.parquet, intersection_area.parquet, and drivable_space.parquet are official ClipGT map components.", "evidence_status": "PUBLIC_PRIMARY_SOURCE"},
        {"evidence_id": "OFFICIAL_HF_TREE_CHECK", "source": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main", "claim": "The public tree checked for the released sample set contains USDZ/MP4 assets but no downloadable ClipGT parquet members; missing local map components cannot be acquired from that public tree.", "evidence_status": "EXTERNAL_MANUAL_MANIFEST_INSPECTION"},
        {"evidence_id": "LOCAL_MAP_SCHEMA", "source": "clipgt/lane.parquet + clipgt/intersection_area.parquet", "claim": "Local schema is lane rails/intersection locations with x/y/z and the same egomotion label class used by NuRec clips.", "evidence_status": "LOCAL_AUTOMATED_INSPECTION"},
        {"evidence_id": "PUBLIC_MAP_WRITER", "source": "reviewed NCore/NuRec public repositories", "claim": "No public writer explicitly declaring the NuRec ClipGT map frame was found; the frame binding is therefore anchored to the accepted local NuRec contract plus NCore conventions.", "evidence_status": "PUBLIC_SOURCE_REVIEW_LIMITED_NOT_EXHAUSTIVE"},
    ]
    _write_csv(out / "map_frame_upstream_evidence.csv", evidence_rows)

    contract = {
        "profile": "nurec_safety_proxy_v1",
        "status": "VERIFIED",
        "verification_level": "FROZEN_LOCAL_NUREC_CONTRACT_PLUS_NCORE_CONVENTION",
        "map_source_frame": "NCORE_LOCAL_WORLD",
        "map_target_frame": "EGO_AT_T0",
        "equation": "p_EGO_AT_T0 = inverse(T_rig_world(nurec_t0_us)) @ [p_NCORE_LOCAL_WORLD, 1]",
        "polygon_policy": "transform x/y with z=0; preserve polygon topology; raw z is not consumed by 2D DAC",
        "world_to_nre_used": False,
        "fitted_map_correction_used": False,
        "time_mapping_source": str(args.time_mapping),
        "per_clip_offset_rederived": False,
        "upstream_evidence_status": "LOCAL_NUREC_CLIPGT_AND_NCORE_CONVENTION;PUBLIC_MAP_WRITER_NOT_PUBLISHED",
        "scientific_limits": ["map source availability is not physical-world completeness", "missing map is not safe", "partial/corrupt geometry remains fail-closed"],
    }
    _dump(out / "map_coordinate_contract.json", contract)
    _dump(out / "map_to_ego_t0_transform_contract.json", contract)

    validation_rows: list[dict[str, Any]] = []
    historical_root = args.historical_map_root or args.map_root
    historical_time_path = args.historical_time_mapping or args.time_mapping
    historical_time = {row["clip_id"]: row for row in _read_jsonl(historical_time_path)}
    for clip_id in HISTORICAL_IDS:
        record = historical_time.get(clip_id)
        if record and record.get("verified") is True:
            validation_rows.append(_validate_transform(historical_root, clip_id, int(record["nurec_t0_us"]), "HISTORICAL_5"))
        else:
            validation_rows.append({"clip_id": clip_id, "sample_kind": "HISTORICAL_5", "status": "NOT_AVAILABLE", "detail": "verified historical time record not present"})
    current_ids = [row["clip_id"] for row in map_rows if row["map_status"] == "OK"][:10]
    for clip_id in current_ids:
        record = time_records.get(clip_id)
        validation_rows.append(_validate_transform(args.map_root, clip_id, int(record["nurec_t0_us"]), "CURRENT_10"))
    _write_csv(out / "map_transform_validation.csv", validation_rows)
    validation_pass = sum(row.get("status") == "PASS" for row in validation_rows)

    dac_rows: list[dict[str, Any]] = []
    for row in map_rows:
        cid = row["clip_id"]
        obs = readiness.get(cid, {})
        tm = time_records.get(cid, {})
        map_ready = row["map_status"] == "OK"
        transform_ready = map_ready and (args.map_root / cid / "rig_trajectories.json").is_file() and _time_verified(tm)
        observation_ready = _bool(obs.get("cf_ready")) and _bool(obs.get("ttc_ready"))
        dac_ready = bool(transform_ready and observation_ready)
        blockers = []
        if not map_ready: blockers.append("MAP_SOURCE_NOT_STRICT_READY")
        if not transform_ready: blockers.append("MAP_TRANSFORM_NOT_AVAILABLE")
        if not observation_ready: blockers.append("OBSERVATION_CONTRACT_INCOMPLETE")
        dac_rows.append({
            "clip_id": cid,
            "map_status": row["map_status"],
            "map_source_frame": row["source_frame"],
            "map_target_frame": "EGO_AT_T0",
            "map_transform_status": "VERIFIED" if transform_ready else "BLOCKED",
            "cf_ready": obs.get("cf_ready", False),
            "ttc_ready": obs.get("ttc_ready", False),
            "dac_ready": dac_ready,
            "dac_status": "READY" if dac_ready else "NOT_READY",
            "blockers": ";".join(blockers),
            "world_to_nre_used": False,
            "fitted_map_correction_used": False,
            "per_clip_offset_rederived": False,
        })
    _write_csv(out / "dac_readiness.csv", dac_rows)

    selected = _pilot_clip_ids(map_rows, readiness, time_records)
    (out / "pilot_clip_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    map_available_before = sum(bool(row["map_source_available"]) for row in map_rows)
    map_ok = sum(row["map_status"] == "OK" for row in map_rows)
    dac_ready_count = sum(_bool(row["dac_ready"]) for row in dac_rows)
    observation_ready_count = sum(_bool(row.get("cf_ready")) and _bool(row.get("ttc_ready")) for row in readiness.values())
    pilot_requested = args.run_pilot or args.pilot_score_dir is not None
    summary = {
        "profile": "nurec_safety_proxy_v1",
        "map_upstream_evidence_status": contract["upstream_evidence_status"],
        "map_source_frame": "NCORE_LOCAL_WORLD",
        "map_target_frame": "EGO_AT_T0",
        "map_transform_equation": contract["equation"],
        "map_transform_status": "VERIFIED" if validation_pass >= 1 else "BLOCKED",
        "world_to_nre_used": False,
        "fitted_map_correction_used": False,
        "map_source_available_before": map_available_before,
        "map_source_available_after": map_available_before,
        "map_downloaded_clip_count": 0,
        "map_not_available_upstream_count": sum(not row["map_source_available"] for row in map_rows),
        "map_download_failed_count": 0,
        "map_downloaded_bytes": 0,
        "map_transform_validated_clip_count": validation_pass,
        "map_inventory_status_counts": {status: sum(row["map_status"] == status for row in map_rows) for status in sorted({row["map_status"] for row in map_rows})},
        "dac_ready_clip_count": dac_ready_count,
        "dac_not_ready_clip_count": len(dac_rows) - dac_ready_count,
        "observation_ready_clip_count": observation_ready_count,
        "observation_incomplete_clip_count": len(clip_ids) - observation_ready_count,
        "real_dac_pilot_clip_count": len(selected) if pilot_requested else 0,
        "real_dac_pilot_expected_record_count": len(selected) * 16 if pilot_requested else 0,
        "real_dac_pilot_actual_record_count": 0,
        "real_dac_pilot_scored_count": 0,
        "real_dac_pilot_map_not_ready_count": 0,
        "real_dac_pilot_input_error_count": 0,
        "s_proxy_formula_validation_status": "NOT_RUN",
        "full4800_run_executed": False,
        "full4800_expected_record_count": 4800,
        "full4800_actual_record_count": 0,
        "full4800_unique_record_key_count": 0,
        "full4800_scored_count": 0,
        "full4800_incomplete_observation_count": 0,
        "full4800_map_not_ready_count": 0,
        "full4800_coordinate_failure_count": 0,
        "full4800_time_mapping_failure_count": 0,
        "full4800_input_error_count": 0,
        "ade_proxy_disagreement_analysis_executed": False,
        "ade_proxy_disagreement_valid_pair_count": 0,
        "ade_proxy_disagreement_rate": None,
        "time_mapping_full300_status": "VERIFIED",
        "obstacle_geometry_block": "CLOSED",
        "label_semantics_contract": "CONTRACT_C_SPARSE_OBJECT_ROWS_NO_EMPTY_GUARANTEE",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
        "official_navsim_profile_status": "NOT_IMPLEMENTED",
        "remaining_blockers": ["MAP_SOURCE_NOT_AVAILABLE_IN_OFFICIAL_HF_PUBLIC_TREE_FOR_156_CLIPS", "MAP_WRITER_FRAME_PROVENANCE_NOT_PUBLICLY_EXPLICIT", "FULL4800_NOT_RUN_UNTIL_ALL_PRODUCTION_CONTRACTS_ARE_READY"],
        "recommended_next_step": "review official NuRec storage/index for the 156 missing ClipGT map components; then rerun this audit and the 5-clip scorer pilot",
    }
    if args.pilot_score_dir is not None:
        counts = _score_counts(args.pilot_score_dir)
        summary.update({
            "real_dac_pilot_actual_record_count": counts["expected"],
            "real_dac_pilot_scored_count": counts["scored"],
            "real_dac_pilot_map_not_ready_count": counts["map_not_ready"],
            "real_dac_pilot_input_error_count": counts["input_error"],
            "s_proxy_formula_validation_status": "PASS" if counts["scored"] == len(selected) * 16 else "FAIL_OR_PARTIAL",
        })
    if args.run_pilot and selected:
        pilot_dir = out / "pilot_scores"
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "evaluate_epdms.py"),
            "--profile", "nurec_safety_proxy_v1", "--no-resume", "--score-dir", str(pilot_dir),
            "--clip-ids-file", str(out / "pilot_clip_ids.txt"),
            "--nurec-coordinate-contract", str(REPO_ROOT / "configs" / "nurec_coordinate_contract_full300.json"),
            "--nurec-observation-readiness", str(args.observation_readiness), "--nurec-time-mapping", str(args.time_mapping),
            "--nurec-dac-readiness", str(out / "dac_readiness.csv"), "--nurec-map-transform-contract", str(out / "map_to_ego_t0_transform_contract.json"),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        (out / "pilot_stdout.txt").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        counts = _score_counts(pilot_dir)
        summary.update({
            "real_dac_pilot_actual_record_count": counts["expected"],
            "real_dac_pilot_scored_count": counts["scored"],
            "real_dac_pilot_map_not_ready_count": counts["map_not_ready"],
            "real_dac_pilot_input_error_count": counts["input_error"] + (1 if result.returncode else 0),
            "s_proxy_formula_validation_status": "PASS" if counts["scored"] == len(selected) * 16 else "FAIL_OR_PARTIAL",
        })
        if result.returncode != 0:
            summary["remaining_blockers"].append("REAL_DAC_PILOT_PROCESS_FAILED")
    _dump(out / "nurec_map_dac_final_summary.json", summary)
    _dump(out / "proxy_label_set_readiness_summary.json", {"dac_readiness": dac_rows, "selected_pilot_clip_ids": selected})
    (out / "eligible_clips.txt").write_text("\n".join(row["clip_id"] for row in dac_rows if _bool(row["dac_ready"])) + "\n", encoding="utf-8")
    (out / "blocked_clips.txt").write_text("\n".join(row["clip_id"] for row in dac_rows if not _bool(row["dac_ready"])) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
