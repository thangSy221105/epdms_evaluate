"""Build additive LK/DDC diagnostics from existing proxy records.

The script intentionally consumes an existing score JSONL and writes a new
versioned output directory.  It does not rerun or mutate the frozen proxy
scorer, and it never fills official EPDMS fields from proxy values.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.epdms.condition_identity import record_key_from_prediction
from tools.epdms.lane_metrics import (
    audit_lane_direction_contract,
    compute_lane_direction_alignment_diagnostic,
    compute_lk_proxy,
    load_lane_centerlines_for_clip,
)
from tools.epdms.metric_vector import build_partial_metric_vector
from tools.epdms.score_record import _prediction_waypoints
from tools.epdms.time_contract import normalize_trajectory_timeline


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_t0(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip_id") and row.get("nurec_t0_us"):
                result[str(row["clip_id"])] = int(row["nurec_t0_us"])
    return result


def selected_xy(prediction: dict[str, Any], alpha: float) -> np.ndarray:
    selected = _prediction_waypoints(prediction, alpha, target_future_poses=40, t0_us=int(prediction["t0_us"]))
    timeline = normalize_trajectory_timeline(
        selected,
        t0_us=int(prediction["t0_us"]),
        expected_frequency_hz=10.0,
        expected_horizon_s=4.0,
        role="prediction",
        strict_grid=True,
    )
    if timeline.includes_t0:
        return np.column_stack([timeline.x, timeline.y])
    return np.vstack([[0.0, 0.0], np.column_stack([timeline.x, timeline.y])])


def gt_xy(gt: dict[str, Any]) -> np.ndarray:
    points = gt.get("ego_future_xyz") or []
    return np.vstack([[0.0, 0.0], [[float(row[0]), float(row[1])] for row in points[:40]]])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--map-transform-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-ids-file", type=Path)
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()

    records = read_jsonl(args.records_jsonl)
    predictions = {record_key_from_prediction(row): row for row in read_jsonl(args.prediction_jsonl)}
    ground_truth = {str(row["clip_id"]): row for row in read_jsonl(args.ground_truth_jsonl) if row.get("clip_id")}
    t0_by_clip = read_t0(args.map_transform_validation)
    available = sorted({str(row.get("clip_id")) for row in records if row.get("clip_id")})
    if args.clip_ids_file:
        requested = [line.strip() for line in args.clip_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        available = [clip for clip in requested if clip in set(available)]
    if args.max_clips is not None:
        available = available[: max(0, args.max_clips)]

    lane_contract = audit_lane_direction_contract(args.map_root, available)
    (args.output_dir / "lane_direction_contract.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lane_direction_contract.json").write_text(json.dumps(lane_contract, indent=2) + "\n", encoding="utf-8")

    bundles: dict[str, dict[str, Any]] = {}
    for clip_id in available:
        if clip_id in t0_by_clip:
            bundles[clip_id] = load_lane_centerlines_for_clip(args.map_root, clip_id, t0_by_clip[clip_id])
        else:
            bundles[clip_id] = {"status": "MAP_TRANSFORM_NOT_READY", "lanes": [], "intersections": []}

    write_csv(args.output_dir / "lane_centerline_validation.csv", [
        {
            "clip_id": clip_id,
            "lane_centerline_status": bundle.get("status"),
            "lane_count": bundle.get("lane_count", len(bundle.get("lanes", []))),
            "intersection_count": bundle.get("intersection_count", len(bundle.get("intersections", []))),
            "world_to_ego_available": bool(bundle.get("world_to_ego")),
        }
        for clip_id, bundle in bundles.items()
    ])
    write_csv(args.output_dir / "lane_schema_inventory.csv", [
        {"field": field, "count": count, "source": "clipgt/lane.parquet"}
        for field, count in sorted(lane_contract.get("field_counts", {}).items())
    ])

    vector_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    gt_lk_cache: dict[str, dict[str, Any] | None] = {}
    for record in records:
        clip_id = str(record.get("clip_id"))
        if clip_id not in set(available):
            continue
        key = str(record.get("record_key"))
        prediction = predictions.get(key)
        bundle = bundles[clip_id]
        if record.get("valid") is not True:
            # Preserve the full 4,800-row additive vector without turning a
            # proxy scorer failure into a newly computed metric result.
            lk = {
                "lk_proxy": None,
                "status": "SOURCE_SCORE_INVALID",
                "associated_lane_ids": [],
            }
            gt_result = None
            diagnostic = {"status": "SOURCE_SCORE_INVALID"}
        elif prediction is None:
            lk = {"lk_proxy": None, "status": "INPUT_ERROR", "associated_lane_ids": []}
            gt_result = None
            diagnostic = {"status": "INPUT_ERROR"}
        else:
            try:
                xy = selected_xy(prediction, float(record.get("alpha", 0.0)))
                lk = compute_lk_proxy(xy, bundle)
                diagnostic = compute_lane_direction_alignment_diagnostic(xy, lk)
                if clip_id not in gt_lk_cache:
                    gt_lk_cache[clip_id] = (
                        compute_lk_proxy(gt_xy(ground_truth[clip_id]), bundle)
                        if clip_id in ground_truth
                        else None
                    )
                gt_result = gt_lk_cache[clip_id]
            except Exception as exc:
                lk = {"lk_proxy": None, "status": "INPUT_ERROR", "error": str(exc), "associated_lane_ids": []}
                gt_result = None
                diagnostic = {"status": "INPUT_ERROR", "error": str(exc)}
        vector_rows.append(build_partial_metric_vector(record, lk_result=lk, ddc_result={"ddc_proxy": None, "status": lane_contract["status"]}, lane_direction_diagnostic=diagnostic, gt_lk_result=gt_result))
        validation_rows.append({
            "clip_id": clip_id,
            "condition": key,
            "lk_proxy": lk.get("lk_proxy"),
            "lk_proxy_status": lk.get("status"),
            "max_lateral_deviation_m": lk.get("max_lateral_deviation_m"),
            "max_continuous_violation_s": lk.get("max_continuous_violation_s"),
            "gt_lk_proxy": gt_result.get("lk_proxy") if gt_result else None,
            "gt_max_lateral_deviation_m": gt_result.get("max_lateral_deviation_m") if gt_result else None,
            "gt_max_continuous_violation_s": gt_result.get("max_continuous_violation_s") if gt_result else None,
            "intersection_excluded_frames": lk.get("intersection_excluded_frames", 0),
            "status": lk.get("status"),
        })

    write_jsonl(args.output_dir / "partial_metric_vector_pilot.jsonl", vector_rows)
    write_csv(args.output_dir / "lk_proxy_validation.csv", validation_rows)
    write_csv(args.output_dir / "ddc_proxy_validation.csv", [
        {
            "clip_id": row["clip_id"],
            "condition": row["condition"],
            "ddc_proxy": None,
            "ddc_proxy_status": lane_contract["status"],
        }
        for row in validation_rows
    ])
    write_csv(args.output_dir / "lane_direction_diagnostics.csv", [
        {
            "clip_id": row["clip_id"],
            "condition": row["condition"],
            "lane_direction_alignment_fraction": vector_rows[index].get("lane_direction_alignment_diagnostic", {}).get("lane_direction_alignment_fraction"),
            "max_heading_misalignment_deg": vector_rows[index].get("lane_direction_alignment_diagnostic", {}).get("max_heading_misalignment_deg"),
            "opposite_heading_frame_count": vector_rows[index].get("lane_direction_alignment_diagnostic", {}).get("opposite_heading_frame_count"),
            "opposite_motion_distance_m": vector_rows[index].get("lane_direction_alignment_diagnostic", {}).get("opposite_motion_distance_m"),
            "status": vector_rows[index].get("lane_direction_alignment_diagnostic", {}).get("status"),
        }
        for index, row in enumerate(validation_rows)
    ])
    summary = {
        "metric_profile": "epdms_partial_vector_lk_ddc_v1",
        "clip_count": len(available),
        "expected_record_count": len([row for row in records if str(row.get("clip_id")) in set(available)]),
        "actual_record_count": len(vector_rows),
        "lk_proxy_valid_count": sum(row.get("lk_proxy") is not None for row in vector_rows),
        "lk_proxy_null_count": sum(row.get("lk_proxy") is None for row in vector_rows),
        "lk_proxy_zero_count": sum(row.get("lk_proxy") == 0.0 for row in vector_rows),
        "lk_proxy_one_count": sum(row.get("lk_proxy") == 1.0 for row in vector_rows),
        "ddc_proxy_valid_count": 0,
        "ddc_proxy_null_count": len(vector_rows),
        "official_lk_populated_count": 0,
        "official_ddc_populated_count": 0,
        "official_epdms_stage1_populated_count": 0,
        "lane_direction_contract_status": lane_contract["status"],
        "nurec_safety_proxy_v1_formula_changed": False,
    }
    (args.output_dir / "partial_metric_vector_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
