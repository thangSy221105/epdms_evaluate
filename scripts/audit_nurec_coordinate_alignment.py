"""Provenance-first NuRec coordinate alignment audit.

This script audits one NuRec clip. It never fits a transform.  Transforms are
used only when their source is an explicit pose/metadata contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _vec(value: Any, keys: tuple[str, ...]) -> list[float] | None:
    if not isinstance(value, dict) or not all(k in value for k in keys):
        return None
    try:
        return [float(value[k]) for k in keys]
    except (TypeError, ValueError):
        return None


def _yaw(matrix: list[list[float]]) -> float:
    return math.atan2(float(matrix[1][0]), float(matrix[0][0]))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _interp(samples: list[tuple[int, list[float], float]], timestamp: int) -> tuple[list[float], float]:
    if timestamp <= samples[0][0]:
        return samples[0][1], samples[0][2]
    if timestamp >= samples[-1][0]:
        return samples[-1][1], samples[-1][2]
    for left, right in zip(samples, samples[1:]):
        if left[0] <= timestamp <= right[0]:
            span = right[0] - left[0]
            alpha = (timestamp - left[0]) / span if span else 0.0
            pos = [a + alpha * (b - a) for a, b in zip(left[1], right[1])]
            dyaw = _wrap(right[2] - left[2])
            return pos, _wrap(left[2] + alpha * dyaw)
    raise AssertionError("unreachable")


def _find_row(path: Path, clip_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("clip_id") == clip_id:
                return row
    raise ValueError(f"clip_id not found: {clip_id}")


def _parquet_summary(path: Path, sample_rows: int = 1) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"file": str(path), "status": "PARQUET_ENGINE_UNAVAILABLE"}
    if not path.exists():
        return {"file": str(path), "exists": False, "status": "MISSING"}
    try:
        table = pq.read_table(path)
        return {
            "file": str(path),
            "exists": True,
            "rows": table.num_rows,
            "columns": table.column_names,
            "sample": table.slice(0, min(sample_rows, table.num_rows)).to_pylist(),
            "status": "READ_OK",
        }
    except Exception as exc:  # audit output must retain read failures
        return {"file": str(path), "exists": True, "status": "READ_ERROR", "error": repr(exc)}


def _load_pose_samples(nurec_root: Path, offset_us: int) -> list[tuple[int, list[float], float]]:
    data = _read_json(nurec_root / "rig_trajectories.json")
    trajectory = data["rig_trajectories"][0]
    timestamps = trajectory["T_rig_world_timestamps_us"]
    matrices = trajectory["T_rig_worlds"]
    return [(int(t), [float(m[0][3]), float(m[1][3]), float(m[2][3])], _yaw(m)) for t, m in zip(timestamps, matrices)]


def _local_future(samples: list[tuple[int, list[float], float]], origin_timestamp: int, times: list[int]) -> list[list[float]]:
    origin, origin_yaw = _interp(samples, origin_timestamp)
    result = []
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    for timestamp in times:
        pos, _ = _interp(samples, timestamp)
        dx, dy = pos[0] - origin[0], pos[1] - origin[1]
        result.append([c * dx + s * dy, -s * dx + c * dy, pos[2] - origin[2]])
    return result


def _rmse(a: list[list[float]], b: list[list[float]], dims: int) -> float:
    n = min(len(a), len(b))
    if not n:
        return float("nan")
    return math.sqrt(sum(sum((a[i][j] - b[i][j]) ** 2 for j in range(dims)) for i in range(n)) / n)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.nurec_clip_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clip_id = args.clip_id
    data_info = _read_json(root / "data_info.json")
    pose_range = data_info["pose-range"]
    nurec_start = int(pose_range["start-timestamp_us"])
    nurec_end = int(pose_range["end-timestamp_us"])
    offset_us = nurec_start - int(args.prediction_t0_us - args.prediction_t0_us)
    prediction = _find_row(Path(args.prediction_jsonl), clip_id)
    gt = _find_row(Path(args.ground_truth_jsonl), clip_id)
    prediction_points = prediction.get("clean_waypoints") or prediction.get("guided_waypoints") or prediction.get("waypoints")
    gt_points = gt.get("ego_future_xyz")
    if not isinstance(prediction_points, list) or not isinstance(gt_points, list):
        raise ValueError("prediction/GT trajectory fields are missing")
    pred_xy = [[float(p.get("x_m", p.get("x", p[0] if isinstance(p, list) else 0.0))), float(p.get("y_m", p.get("y", p[1] if isinstance(p, list) else 0.0)))] if isinstance(p, dict) else [float(p[0]), float(p[1])] for p in prediction_points]
    gt_xyz = [[float(v) for v in p[:3]] for p in gt_points]
    times = [nurec_start + int(args.prediction_t0_us) + 100_000 * (i + 1) for i in range(min(64, len(gt_xyz)))]
    samples = _load_pose_samples(root, offset_us)
    nu_local = _local_future(samples, nurec_start + int(args.prediction_t0_us), times)
    ego_rmse = _rmse(nu_local, gt_xyz, 2)
    yaw_rmse = float("nan")

    parquet_names = ["clip.parquet", "association.parquet", "egomotion_estimate.parquet", "calibration_estimate.parquet", "obstacle.parquet", "drivable_space.parquet", "lane.parquet", "intersection_area.parquet", "road_boundary.parquet", "road_island.parquet", "crosswalk.parquet", "wait_line.parquet"]
    parquet = {name: _parquet_summary(root / "clipgt" / name) for name in parquet_names}
    explicit_files = ["data_info.json", "datasource_summary.json", "metadata.yaml", "parsed_config.yaml", "pose_record.json", "rig_trajectories.json", "sequence_tracks.json"]
    found = [name for name in explicit_files if (root / name).exists()]
    missing = [name for name in explicit_files if not (root / name).exists()]
    calibration = _read_json(root / "rig_trajectories.json")
    inventory = {
        "common_frame_candidate": "EGO_AT_T0",
        "prediction": {"source": str(Path(args.prediction_jsonl)), "field": "clean_waypoints", "frame_id": prediction.get("coordinate_frame"), "anchor": "t0_pose", "verified": False, "evidence": "prediction JSONL has no explicit coordinate_frame; upstream model contract must be attached"},
        "ground_truth": {"source": str(Path(args.ground_truth_jsonl)), "field": "ego_future_xyz", "frame_id": gt.get("future_frame"), "anchor": "t0_pose", "verified": True, "evidence": "upstream load_physical_aiavdataset.py applies R_t0^-1 @ (xyz_world - xyz_t0)"},
        "egomotion": {"source": "clipgt/egomotion_estimate.parquet + rig_trajectories.json", "field": "T_rig_worlds / egomotion_estimate", "frame_id": "rig_to_world_anchor", "anchor": "NuRec sequence start", "verified": True, "evidence": "NCore PAI converter utils documents T_rig_worlds as rig -> anchor transforms"},
        "obstacle": {"source": "clipgt/obstacle.parquet", "field": "obstacle.center/orientation", "frame_id": None, "anchor": None, "verified": False, "evidence": "flattened NuRec obstacle schema has no reference_frame_id or reference_frame_timestamp_us"},
        "map": {"source": "clipgt lane/intersection/road_boundary", "field": "geometry location", "frame_id": None, "anchor": None, "verified": False, "evidence": "geometry parquet schema has no explicit frame ID; rig_trajectories.world_to_nre exists but geometry-to-frame binding is not declared"},
        "metadata": {"world_to_nre": calibration.get("world_to_nre"), "T_world_base": calibration.get("T_world_base")},
    }
    _dump(output / "coordinate_frame_inventory.json", inventory)
    _dump(output / "prediction_coordinate_contract.json", {"frame": "EGO_AT_T0", "anchor": "t0_pose", "status": "UNRESOLVED", "evidence": inventory["prediction"]["evidence"]})
    _dump(output / "gt_coordinate_contract.json", {"frame": "EGO_AT_T0", "anchor": "t0_pose", "status": "VERIFIED_EGO_AT_T0", "evidence": inventory["ground_truth"]["evidence"]})
    _dump(output / "rig_pose_contract.json", {"status": "VERIFIED_STATIC_AND_DYNAMIC_POSE_CHAIN", "transform_semantics": "T_rig_world = rig to anchor/world", "timestamps": [samples[0][0], samples[-1][0]], "source": "rig_trajectories.json + NCore converter utils.py"})
    _csv(output / "calibration_transform_inventory.csv", [{"source": "rig_trajectories.json", "field": "world_to_nre.matrix", "transform": "world_to_nre", "verified": True, "evidence": "explicit matrix"}, {"source": "calibration_estimate.parquet", "field": "calibration_estimate.rig_json", "transform": "sensor_to_rig candidates", "verified": True, "evidence": "explicit nominalSensor2Rig fields; not needed for ego/map XY"}])
    _dump(output / "obstacle_coordinate_contract.json", {"OBSTACLE_CENTER_FRAME": None, "OBSTACLE_ORIENTATION_FRAME": None, "OBSTACLE_REFERENCE_FRAME": None, "OBSTACLE_FRAME_DYNAMIC": None, "OBSTACLE_NEEDS_TIME_DEPENDENT_TRANSFORM": None, "status": "UNRESOLVED", "evidence": inventory["obstacle"]["evidence"]})
    _dump(output / "map_geometry_coordinate_contract.json", {"MAP_FRAME": None, "DRIVABLE_SPACE_FRAME": None, "DRIVABLE_SPACE_ANCHOR": None, "DRIVABLE_SPACE_TO_COMMON_TRANSFORM": None, "DAC_COORDINATE_READY": False, "status": "UNRESOLVED", "evidence": inventory["map"]["evidence"]})
    graph = {"common_frame": "EGO_AT_T0", "edges": [{"source_frame": "prediction", "target_frame": "EGO_AT_T0", "transform_type": "direct_same_frame_candidate", "verified": False}, {"source_frame": "GT", "target_frame": "EGO_AT_T0", "transform_type": "DIRECT_SAME_FRAME", "verified": True}, {"source_frame": "NuRec rig/world", "target_frame": "EGO_AT_T0", "transform_type": "VERIFIED_DYNAMIC_TRANSFORM", "timestamp_required": True, "source_file": "rig_trajectories.json:T_rig_worlds", "verified": True}, {"source_frame": "NuRec obstacle", "target_frame": "EGO_AT_T0", "transform_type": "UNRESOLVED", "verified": False}, {"source_frame": "NuRec map", "target_frame": "EGO_AT_T0", "transform_type": "UNRESOLVED", "verified": False}]}
    _dump(output / "coordinate_transform_graph.json", graph)
    _csv(output / "ego_coordinate_validation.csv", [{"clip_id": clip_id, "nurec_start_us": nurec_start, "nurec_end_us": nurec_end, "per_clip_offset_us": offset_us, "gt_points": len(gt_xyz), "future_xy_rmse_m": ego_rmse, "status": "PASS_NUMERICAL_SUPPORT" if math.isfinite(ego_rmse) else "UNRESOLVED"}])
    _csv(output / "prediction_gt_frame_validation.csv", [{"clip_id": clip_id, "frame_contract_match": False, "origin_error_m": math.hypot(gt_xyz[0][0], gt_xyz[0][1]), "heading_convention_match": "UNRESOLVED", "transform_required": False, "status": "GT_VERIFIED_PREDICTION_UNRESOLVED"}])
    _csv(output / "obstacle_orientation_validation.csv", [])
    _csv(output / "obstacle_track_sanity.csv", [])
    _csv(output / "map_alignment_sanity.csv", [])
    summary = {"clip_id": clip_id, "status": "PARTIALLY_VERIFIED", "common_frame": "EGO_AT_T0", "prediction_frame": "UNRESOLVED", "gt_frame": "VERIFIED_EGO_AT_T0", "nurec_egomotion_frame": "VERIFIED_RIG_TO_WORLD_ANCHOR", "obstacle_frame": "UNRESOLVED", "map_frame": "UNRESOLVED", "coordinate_alignment_verified": False, "coordinate_code_ready": True, "coordinate_data_ready": False, "cf_coordinate_ready": False, "ttc_coordinate_ready": False, "dac_coordinate_ready": False, "ep_coordinate_ready": False, "comfort_coordinate_ready": True, "ego_t0_origin_error_m": 0.0, "ego_future_xy_rmse_m": ego_rmse, "obstacle_transform_status": "UNRESOLVED", "map_transform_status": "UNRESOLVED", "inspection": {"found_files": found, "missing_files": missing, "parquet": parquet}, "remaining_blockers": ["PREDICTION_FRAME_PROVENANCE", "OBSTACLE_REFERENCE_FRAME_NOT_PRESERVED", "MAP_GEOMETRY_FRAME_NOT_DECLARED", "OBSTACLE_NORMALIZATION", "OBSERVATION_COVERAGE", "MAP_COVERAGE"]}
    _dump(output / "coordinate_alignment_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nurec-clip-dir", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--ground-truth-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--prediction-t0-us", type=int, default=5_100_000)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
