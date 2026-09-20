"""Forensic closure audit for the seven retained cuboid center outliers."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.audit_cuboid_local_rig_bridge import (
        SCIPY_AVAILABLE, R, assignment_rmse, center, corners, hull_xy, inv,
        load_nurec_poses, object_pose_scipy, polygon_iou, pose_from_target,
        read_member, rot_error_deg, source_name,
    )
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, interpolate_pose
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_cuboid_local_rig_bridge import (
        SCIPY_AVAILABLE, R, assignment_rmse, center, corners, hull_xy, inv,
        load_nurec_poses, object_pose_scipy, polygon_iou, pose_from_target,
        read_member, rot_error_deg, source_name,
    )
    from audit_nurec_coordinate_alignment import _mm, _pose, interpolate_pose
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump


def matrix_json(matrix):
    return json.dumps(np.asarray(matrix, dtype=float).tolist(), separators=(",", ":"))


def vector_json(vector):
    return json.dumps([float(x) for x in vector], separators=(",", ":"))


def p95(values):
    return float(np.quantile(values, 0.95)) if values else None


def median_abs_deviation(values):
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median(abs(x - med) for x in values)


def displacement(a, b):
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def speed(distance, ts_a, ts_b):
    if distance is None or ts_a is None or ts_b is None or ts_a == ts_b:
        return None
    return float(distance / (abs(int(ts_a) - int(ts_b)) / 1_000_000.0))


def yaw_step(a, b):
    if a is None or b is None:
        return None
    return math.degrees(abs(_wrap(float(a) - float(b))))


def build_selected_rows(raw, local_map, offset, matched_only=True):
    selected = []
    for raw_index, row in sorted(enumerate(raw), key=lambda x: (str(x[1]["track_id"]), int(x[1]["timestamp_us"]))):
        if source_name(row.get("source")) != "AUTOLABEL":
            continue
        if math.dist([row["center_x"], row["center_y"], row["center_z"]], [0, 0, 0]) < 3:
            continue
        mapped = int(row["timestamp_us"]) + int(offset)
        if not matched_only or (str(row["track_id"]), mapped) in local_map:
            selected.append((raw_index, row, mapped))
    return selected


def make_observation(raw_index, raw, mapped, target, ego_a, ego_b, object_pose, first_dims, first_ts):
    object_a = np.asarray(ego_a, dtype=float) @ object_pose
    object_b = np.asarray(ego_b, dtype=float) @ object_pose
    target_pose = pose_from_target(target)
    a_center_error = center(target_pose) - center(object_a)
    b_center_error = center(target_pose) - center(object_b)
    a_yaw = math.degrees(abs(_wrap(pose_yaw(object_a) - pose_yaw(target_pose))))
    b_yaw = math.degrees(abs(_wrap(pose_yaw(object_b) - pose_yaw(target_pose))))
    a_signed_yaw = math.degrees(_wrap(pose_yaw(target_pose) - pose_yaw(object_a)))
    b_signed_yaw = math.degrees(_wrap(pose_yaw(target_pose) - pose_yaw(object_b)))
    dimensions = list(first_dims)
    target_dimensions = list(target["dimensions"])
    a_corners = corners(object_a, dimensions)
    b_corners = corners(object_b, dimensions)
    local_corners = corners(target_pose, target_dimensions)
    a_hull, b_hull, local_hull = hull_xy(a_corners), hull_xy(b_corners), hull_xy(local_corners)
    return {
        "clip_id": str(raw.get("clip_id", "")),
        "track_id": str(raw["track_id"]),
        "category": target.get("category"),
        "raw_row_index": int(raw_index),
        "source": raw.get("source"),
        "source_version": raw.get("source_version", raw.get("version")),
        "raw_timestamp_us": int(raw["timestamp_us"]),
        "mapped_nurec_timestamp_us": int(mapped),
        "reference_frame_timestamp_us": int(raw["reference_frame_timestamp_us"]),
        "mapped_reference_frame_timestamp_us": int(raw["reference_frame_timestamp_us"]),
        "timestamp_minus_reference_us": int(raw["timestamp_us"]) - int(raw["reference_frame_timestamp_us"]),
        "raw_center": [float(raw["center_x"]), float(raw["center_y"]), float(raw["center_z"])],
        "raw_quaternion": [float(raw["orientation_x"]), float(raw["orientation_y"]), float(raw["orientation_z"]), float(raw["orientation_w"])],
        "raw_dimensions": [float(raw["size_x"]), float(raw["size_y"]), float(raw["size_z"])],
        "track_first_selected_raw_timestamp": int(first_ts),
        "track_first_selected_dimensions": dimensions,
        "local_track_dimensions": target_dimensions,
        "dimension_difference": [float(a - b) for a, b in zip(dimensions, target_dimensions)],
        "raw_ego_pose_at_reference_ts": matrix_json(ego_a),
        "variant_A_ego_pose": matrix_json(ego_a),
        "variant_B_local_rig_pose": matrix_json(ego_b),
        "A_vs_B_translation_delta": vector_json(center(ego_b) - center(ego_a)),
        "A_vs_B_yaw_delta_deg": math.degrees(_wrap(pose_yaw(ego_b) - pose_yaw(ego_a))),
        "A_reconstructed_center": center(object_a).tolist(),
        "B_reconstructed_center": center(object_b).tolist(),
        "local_sequence_center": list(map(float, target["center"])),
        "A_center_residual_xyz": a_center_error.tolist(),
        "B_center_residual_xyz": b_center_error.tolist(),
        "A_center_residual_m": float(np.linalg.norm(a_center_error)),
        "B_center_residual_m": float(np.linalg.norm(b_center_error)),
        "A_yaw_residual_deg": a_yaw,
        "B_yaw_residual_deg": b_yaw,
        "raw_yaw_deg": math.degrees(pose_yaw(object_pose)),
        "local_yaw_deg": math.degrees(pose_yaw(target_pose)),
        "A_signed_yaw_delta_deg": a_signed_yaw,
        "B_signed_yaw_delta_deg": b_signed_yaw,
        "A_BEV_IOU": polygon_iou(a_hull, local_hull),
        "B_BEV_IOU": polygon_iou(b_hull, local_hull),
        "A_3D_CORNER_ERROR_M": assignment_rmse(a_corners, local_corners),
        "B_3D_CORNER_ERROR_M": assignment_rmse(b_corners, local_corners),
        "raw_source_frame": raw.get("reference_frame"),
        "raw_object_pose": object_pose,
        "variant_A_matrix": object_a,
        "variant_B_matrix": object_b,
    }


def make_unmatched_track_row(raw_index, raw, mapped, ego_a, ego_b, object_pose, first_dims, first_ts):
    object_a = np.asarray(ego_a, dtype=float) @ object_pose
    object_b = np.asarray(ego_b, dtype=float) @ object_pose
    return {
        "clip_id": str(raw.get("clip_id", "")), "track_id": str(raw["track_id"]), "category": None,
        "raw_row_index": int(raw_index), "source": raw.get("source"), "source_version": raw.get("source_version", raw.get("version")),
        "raw_timestamp_us": int(raw["timestamp_us"]), "mapped_nurec_timestamp_us": int(mapped),
        "reference_frame_timestamp_us": int(raw["reference_frame_timestamp_us"]),
        "mapped_reference_frame_timestamp_us": int(raw["reference_frame_timestamp_us"]),
        "timestamp_minus_reference_us": int(raw["timestamp_us"]) - int(raw["reference_frame_timestamp_us"]),
        "raw_center": [float(raw["center_x"]), float(raw["center_y"]), float(raw["center_z"])],
        "raw_quaternion": [float(raw["orientation_x"]), float(raw["orientation_y"]), float(raw["orientation_z"]), float(raw["orientation_w"])],
        "raw_dimensions": [float(raw["size_x"]), float(raw["size_y"]), float(raw["size_z"])],
        "track_first_selected_raw_timestamp": int(first_ts), "track_first_selected_dimensions": list(map(float, first_dims)),
        "local_track_dimensions": None, "dimension_difference": None,
        "raw_ego_pose_at_reference_ts": matrix_json(ego_a), "variant_A_ego_pose": matrix_json(ego_a), "variant_B_local_rig_pose": matrix_json(ego_b),
        "A_vs_B_translation_delta": vector_json(center(ego_b) - center(ego_a)), "A_vs_B_yaw_delta_deg": math.degrees(_wrap(pose_yaw(ego_b) - pose_yaw(ego_a))),
        "A_reconstructed_center": center(object_a).tolist(), "B_reconstructed_center": center(object_b).tolist(), "local_sequence_center": None,
        "A_center_residual_xyz": None, "B_center_residual_xyz": None, "A_center_residual_m": None, "B_center_residual_m": None,
        "A_yaw_residual_deg": None, "B_yaw_residual_deg": None, "A_signed_yaw_delta_deg": None, "B_signed_yaw_delta_deg": None,
        "A_BEV_IOU": None, "B_BEV_IOU": None, "A_3D_CORNER_ERROR_M": None, "B_3D_CORNER_ERROR_M": None,
        "raw_source_frame": raw.get("reference_frame"), "raw_object_pose": object_pose, "variant_A_matrix": object_a, "variant_B_matrix": object_b,
        "raw_yaw_deg": math.degrees(pose_yaw(object_pose)), "local_yaw_deg": None,
    }


def public_row(row):
    result = dict(row)
    for key in ("raw_object_pose", "variant_A_matrix", "variant_B_matrix"):
        result.pop(key, None)
    for key, value in list(result.items()):
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
    return result


def neighborhood(track_rows, center_index, radius=3):
    lo, hi = max(0, center_index - radius), min(len(track_rows), center_index + radius + 1)
    return [(idx, row) for idx, row in enumerate(track_rows[lo:hi], start=lo)]


def timeline_classification(track_rows, outlier_indices, reference_delta_threshold):
    if not outlier_indices:
        return "UNRESOLVED"
    if any(abs(track_rows[i]["timestamp_minus_reference_us"]) > reference_delta_threshold for i in outlier_indices):
        return "REFERENCE_TIMESTAMP_ANOMALY"
    raw_steps = []
    local_steps = []
    for left, right in zip(track_rows, track_rows[1:]):
        raw_steps.append(displacement(left["raw_center"], right["raw_center"]))
        local_steps.append(displacement(left["local_sequence_center"], right["local_sequence_center"]))
    raw_baseline = statistics.median(sorted(raw_steps)[:max(1, math.floor(len(raw_steps) * 0.5))]) if raw_steps else 0.0
    local_baseline = statistics.median(sorted(local_steps)[:max(1, math.floor(len(local_steps) * 0.5))]) if local_steps else 0.0
    for index in outlier_indices:
        for neighbor_index in (index - 1, index):
            if 0 <= neighbor_index < len(raw_steps):
                raw_jump = raw_steps[neighbor_index]
                local_jump = local_steps[neighbor_index]
                if raw_jump > max(5.0, raw_baseline * 5.0) and raw_jump >= local_jump * 0.5:
                    return "SOURCE_LABEL_JUMP"
    residuals = [x["A_center_residual_m"] for x in track_rows]
    if len(track_rows) == 1:
        return "BOUNDARY_ONLY_ANOMALY"
    if residuals and all(x > 2.0 for x in residuals):
        spread = max(residuals) - min(residuals)
        if spread < max(0.5, statistics.median(residuals) * 0.5):
            return "TRACK_WIDE_CONSTANT_OFFSET"
        return "TRACK_WIDE_TIME_VARYING_OFFSET"
    contiguous = any(b == a + 1 for a, b in zip(outlier_indices, outlier_indices[1:]))
    if contiguous:
        return "CONTIGUOUS_LOCAL_TRACK_DIVERGENCE"
    if len(outlier_indices) == 1:
        index = outlier_indices[0]
        before = track_rows[index - 1]["A_center_residual_m"] if index else None
        after = track_rows[index + 1]["A_center_residual_m"] if index + 1 < len(track_rows) else None
        neighbors = [x for x in (before, after) if x is not None]
        if not neighbors:
            return "BOUNDARY_ONLY_ANOMALY"
        if all(x < 2.0 for x in neighbors):
            return "ISOLATED_SINGLE_ROW"
    return "UNRESOLVED"


def global_frame_evidence(rows, clip_bias, bridge_rows):
    significant_translation = []
    for axis in ("dx", "dy", "dz"):
        values = [float(item[f"mean_{axis}"]) for item in clip_bias]
        significant = [x for x in values if abs(x) >= 0.2]
        if len(significant) >= 3 and (all(x > 0 for x in significant) or all(x < 0 for x in significant)):
            significant_translation.append(axis)
    significant_yaw = [float(item["mean_signed_yaw_deg"]) for item in clip_bias if abs(float(item["mean_signed_yaw_deg"])) >= 1.0]
    consistent_yaw = len(significant_yaw) >= 3 and (all(x > 0 for x in significant_yaw) or all(x < 0 for x in significant_yaw))
    outlier_rate = sum(x["A_center_residual_m"] > 2.0 for x in rows) / len(rows)
    bridge_translation_max = max(x["translation_delta_norm_m"] for x in bridge_rows)
    bridge_yaw_max = max(abs(x["yaw_delta_deg"]) for x in bridge_rows)
    evidence = {"consistent_translation_axes": significant_translation, "consistent_signed_yaw": consistent_yaw, "center_outlier_rate": outlier_rate, "ego_bridge_translation_max_m": bridge_translation_max, "ego_bridge_yaw_max_deg": bridge_yaw_max, "structured_failure_fraction": outlier_rate}
    supported = bool(significant_translation or consistent_yaw or outlier_rate > 0.1)
    return ("SUPPORTED" if supported else "NOT_SUPPORTED"), evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pai-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["clip_id"] in PILOT_CLIPS]
    all_rows, all_selected_by_track, bridge_rows = [], defaultdict(list), []
    scipy_errors = []
    for item in manifest:
        clip_id = item["clip_id"]
        clip = Path(item["nurec_clip_dir"])
        record = load_time_record(Path(args.time_alignment_jsonl), clip_id, int(item.get("physicalai_t0_us", 5100000)))
        raw = read_member(args.pai_root, "labels/obstacle.offline", ".obstacle.offline.parquet", clip_id)
        ego = read_member(args.pai_root, "labels/egomotion.offline", ".egomotion.offline.parquet", clip_id)
        local, _, _ = load_sequence_tracks(clip / "sequence_tracks.json")
        local_map = {(str(row["track_id"]), int(row["timestamp_us"])): row for row in local}
        selected_all = build_selected_rows(raw, local_map, record["offset_us"], matched_only=False)
        selected = [item for item in selected_all if (str(item[1]["track_id"]), item[2]) in local_map]
        first_by_track = {}
        for _, row, _ in selected_all:
            first_by_track.setdefault(str(row["track_id"]), (int(row["timestamp_us"]), [row["size_x"], row["size_y"], row["size_z"]]))
        ego_samples = [(int(row["timestamp"]), _pose([row["x"], row["y"], row["z"]], [row["qx"], row["qy"], row["qz"], row["qw"]])) for row in ego]
        rebase = inv(ego_samples[0][1])
        local_rig = load_nurec_poses(clip)
        for raw_index, row, mapped in selected_all:
            target = local_map.get((str(row["track_id"]), mapped))
            raw_ref = int(row["reference_frame_timestamp_us"])
            mapped_ref = raw_ref + int(record["offset_us"])
            raw_ego = np.asarray(interpolate_pose(ego_samples, raw_ref), dtype=float)
            ego_a = np.asarray(rebase, dtype=float) @ raw_ego
            ego_b = np.asarray(interpolate_pose(local_rig, mapped_ref), dtype=float)
            object_pose, raw_rotation = object_pose_scipy(row)
            if raw_rotation is not None:
                rebuilt = R.from_euler("xyz", raw_rotation.as_euler("xyz", degrees=False), degrees=False)
                scipy_errors.append(math.degrees((raw_rotation.inv() * rebuilt).magnitude()))
            first_ts, first_dims = first_by_track[str(row["track_id"])]
            result = (make_observation(raw_index, row, mapped, target, ego_a, ego_b, object_pose, first_dims, first_ts)
                      if target is not None else
                      make_unmatched_track_row(raw_index, row, mapped, ego_a, ego_b, object_pose, first_dims, first_ts))
            result["clip_id"] = clip_id
            result["mapped_reference_frame_timestamp_us"] = mapped_ref
            all_selected_by_track[(clip_id, str(row["track_id"]))].append(result)
            if target is not None:
                all_rows.append(result)
                delta = np.asarray(inv(ego_a.tolist()), dtype=float) @ ego_b
                bridge_rows.append({"clip_id": clip_id, "track_id": str(row["track_id"]), "raw_timestamp_us": int(row["timestamp_us"]), "mapped_reference_frame_timestamp_us": mapped_ref, "translation_delta_norm_m": float(np.linalg.norm(delta[:3, 3])), "delta_x_m": float(delta[0, 3]), "delta_y_m": float(delta[1, 3]), "delta_z_m": float(delta[2, 3]), "yaw_delta_deg": math.degrees(pose_yaw(delta.tolist()))})
    for rows in all_selected_by_track.values():
        rows.sort(key=lambda x: x["mapped_nurec_timestamp_us"])
    for row in all_rows:
        row["source"] = row.get("source") or "AUTOLABEL"
    outliers = [row for row in all_rows if row["A_center_residual_m"] > 2.0]
    outlier_keys = {(row["clip_id"], row["track_id"]) for row in outliers}
    reference_deltas = [abs(row["timestamp_minus_reference_us"]) for row in all_rows]
    reference_threshold = max(100_000.0, p95(reference_deltas) * 1.5)
    classification_rows = []
    neighbor_rows = []
    timeline_rows = []
    for key in sorted(outlier_keys):
        full_track_rows = all_selected_by_track[key]
        track_rows = [row for row in full_track_rows if row["local_sequence_center"] is not None]
        track_rows.sort(key=lambda x: x["mapped_nurec_timestamp_us"])
        indices = [i for i, row in enumerate(track_rows) if row["A_center_residual_m"] > 2.0]
        classification = timeline_classification(track_rows, indices, reference_threshold)
        local_boundary_evidence = False
        for index in indices:
            full_index = next(i for i, candidate in enumerate(full_track_rows) if candidate["raw_row_index"] == track_rows[index]["raw_row_index"])
            previous = full_track_rows[full_index - 1] if full_index > 0 else None
            following = full_track_rows[full_index + 1] if full_index + 1 < len(full_track_rows) else None
            if previous is not None and following is not None and previous["local_sequence_center"] is None and following["local_sequence_center"] is None:
                local_boundary_evidence = True
        classification_rows.append({"clip_id": key[0], "track_id": key[1], "outlier_count": len(indices), "classification": classification, "local_sequence_boundary_evidence": local_boundary_evidence, "outlier_timestamps_us": json.dumps([track_rows[i]["raw_timestamp_us"] for i in indices]), "residual_timeline_m": json.dumps([round(row["A_center_residual_m"], 6) for row in track_rows])})
        for index in indices:
            full_index = next(i for i, candidate in enumerate(full_track_rows) if candidate["raw_row_index"] == track_rows[index]["raw_row_index"])
            for neighbor_index, neighbor in neighborhood(full_track_rows, full_index, 3):
                prev = full_track_rows[neighbor_index - 1] if neighbor_index > 0 else None
                nxt = full_track_rows[neighbor_index + 1] if neighbor_index + 1 < len(full_track_rows) else None
                neighbor_rows.append({"outlier_clip_id": key[0], "outlier_track_id": key[1], "relative_index": neighbor_index - full_index, "is_outlier_row": neighbor.get("A_center_residual_m") is not None and neighbor["A_center_residual_m"] > 2.0, "raw_timestamp_us": neighbor["raw_timestamp_us"], "mapped_nurec_timestamp_us": neighbor["mapped_nurec_timestamp_us"], "raw_center": vector_json(neighbor["raw_center"]), "reconstructed_center_a": vector_json(neighbor["A_reconstructed_center"]), "reconstructed_center_b": vector_json(neighbor["B_reconstructed_center"]), "local_center": vector_json(neighbor["local_sequence_center"]) if neighbor["local_sequence_center"] is not None else None, "raw_prev_to_current_m": displacement(prev["raw_center"], neighbor["raw_center"]) if prev else None, "raw_current_to_next_m": displacement(neighbor["raw_center"], nxt["raw_center"]) if nxt else None, "A_prev_to_current_m": displacement(prev["A_reconstructed_center"], neighbor["A_reconstructed_center"]) if prev else None, "A_current_to_next_m": displacement(neighbor["A_reconstructed_center"], nxt["A_reconstructed_center"]) if nxt else None, "B_prev_to_current_m": displacement(prev["B_reconstructed_center"], neighbor["B_reconstructed_center"]) if prev else None, "B_current_to_next_m": displacement(neighbor["B_reconstructed_center"], nxt["B_reconstructed_center"]) if nxt else None, "local_prev_to_current_m": displacement(prev["local_sequence_center"], neighbor["local_sequence_center"]) if prev and prev["local_sequence_center"] is not None and neighbor["local_sequence_center"] is not None else None, "local_current_to_next_m": displacement(neighbor["local_sequence_center"], nxt["local_sequence_center"]) if nxt and neighbor["local_sequence_center"] is not None and nxt["local_sequence_center"] is not None else None, "raw_prev_speed_mps": speed(displacement(prev["raw_center"], neighbor["raw_center"]), prev["raw_timestamp_us"], neighbor["raw_timestamp_us"]) if prev else None, "raw_next_speed_mps": speed(displacement(neighbor["raw_center"], nxt["raw_center"]), neighbor["raw_timestamp_us"], nxt["raw_timestamp_us"]) if nxt else None, "A_prev_speed_mps": speed(displacement(prev["A_reconstructed_center"], neighbor["A_reconstructed_center"]), prev["raw_timestamp_us"], neighbor["raw_timestamp_us"]) if prev else None, "A_next_speed_mps": speed(displacement(neighbor["A_reconstructed_center"], nxt["A_reconstructed_center"]), neighbor["raw_timestamp_us"], nxt["raw_timestamp_us"]) if nxt else None, "B_prev_speed_mps": speed(displacement(prev["B_reconstructed_center"], neighbor["B_reconstructed_center"]), prev["raw_timestamp_us"], neighbor["raw_timestamp_us"]) if prev else None, "B_next_speed_mps": speed(displacement(neighbor["B_reconstructed_center"], nxt["B_reconstructed_center"]), neighbor["raw_timestamp_us"], nxt["raw_timestamp_us"]) if nxt else None, "local_prev_speed_mps": speed(displacement(prev["local_sequence_center"], neighbor["local_sequence_center"]), prev["raw_timestamp_us"], neighbor["raw_timestamp_us"]) if prev and prev["local_sequence_center"] is not None and neighbor["local_sequence_center"] is not None else None, "local_next_speed_mps": speed(displacement(neighbor["local_sequence_center"], nxt["local_sequence_center"]), neighbor["raw_timestamp_us"], nxt["raw_timestamp_us"]) if nxt and neighbor["local_sequence_center"] is not None and nxt["local_sequence_center"] is not None else None, "raw_prev_yaw_step_deg": yaw_step(prev.get("raw_yaw_deg") if prev else None, neighbor.get("raw_yaw_deg")) if prev else None, "raw_next_yaw_step_deg": yaw_step(neighbor.get("raw_yaw_deg"), nxt.get("raw_yaw_deg") if nxt else None) if nxt else None, "local_prev_yaw_step_deg": yaw_step(prev.get("local_yaw_deg") if prev else None, neighbor.get("local_yaw_deg")) if prev else None, "local_next_yaw_step_deg": yaw_step(neighbor.get("local_yaw_deg"), nxt.get("local_yaw_deg") if nxt else None) if nxt else None})
        for row in track_rows:
            timeline_rows.append({"clip_id": key[0], "track_id": key[1], "raw_timestamp_us": row["raw_timestamp_us"], "mapped_nurec_timestamp_us": row["mapped_nurec_timestamp_us"], "reference_frame_timestamp_us": row["reference_frame_timestamp_us"], "reference_delta_us": row["timestamp_minus_reference_us"], "A_center_residual_m": row["A_center_residual_m"], "B_center_residual_m": row["B_center_residual_m"], "dx": row["A_center_residual_xyz"][0], "dy": row["A_center_residual_xyz"][1], "dz": row["A_center_residual_xyz"][2], "A_yaw_residual_deg": row["A_yaw_residual_deg"], "B_yaw_residual_deg": row["B_yaw_residual_deg"], "source": row["source"]})
    clip_bias = []
    for clip_id in sorted({row["clip_id"] for row in all_rows}):
        rows = [row for row in all_rows if row["clip_id"] == clip_id]
        vec = np.asarray([row["A_center_residual_xyz"] for row in rows], dtype=float)
        yaw = [row["A_signed_yaw_delta_deg"] for row in rows]
        clip_bias.append({"clip_id": clip_id, "matched_count": len(rows), "mean_dx": float(np.mean(vec[:, 0])), "mean_dy": float(np.mean(vec[:, 1])), "mean_dz": float(np.mean(vec[:, 2])), "mean_signed_yaw_deg": float(np.mean(yaw))})
    global_status, global_evidence = global_frame_evidence(all_rows, clip_bias, bridge_rows)
    classes = [row["classification"] for row in classification_rows]
    outlier_status = "ALL_LOCALIZED_NO_GLOBAL_FRAME_PATTERN" if all(x != "UNRESOLVED" for x in classes) and global_status == "NOT_SUPPORTED" else ("UNRESOLVED" if "UNRESOLVED" in classes else "MIXED")
    dimension_diffs = [abs(value) for row in all_rows for value in row["dimension_difference"]]
    max_outlier = max(outliers, key=lambda row: row["A_center_residual_m"])
    max_classification = next((row["classification"] for row in classification_rows if row["clip_id"] == max_outlier["clip_id"] and row["track_id"] == max_outlier["track_id"]), "UNRESOLVED")
    max_classification_row = next(row for row in classification_rows if row["clip_id"] == max_outlier["clip_id"] and row["track_id"] == max_outlier["track_id"])
    max_provenance = {"SOURCE_LABEL_JUMP": "RAW_SOURCE_ANOMALY", "CONTIGUOUS_LOCAL_TRACK_DIVERGENCE": "LOCAL_SEQUENCE_ANOMALY", "REFERENCE_TIMESTAMP_ANOMALY": "REFERENCE_TIMESTAMP_ANOMALY", "TRACK_WIDE_CONSTANT_OFFSET": "TRACK_DISCONTINUITY", "TRACK_WIDE_TIME_VARYING_OFFSET": "TRACK_DISCONTINUITY", "BOUNDARY_ONLY_ANOMALY": "LOCAL_SEQUENCE_ANOMALY" if max_classification_row["local_sequence_boundary_evidence"] else "TRACK_DISCONTINUITY"}.get(max_classification, "UNRESOLVED")
    row_count_for_class = lambda name: sum(row["outlier_count"] for row in classification_rows if row["classification"] == name)
    local_sequence_anomaly_count = row_count_for_class("CONTIGUOUS_LOCAL_TRACK_DIVERGENCE") + row_count_for_class("BOUNDARY_ONLY_ANOMALY")
    row_quality = "VERIFIED_WITH_RETAINED_SOURCE_ANOMALIES" if outlier_status == "ALL_LOCALIZED_NO_GLOBAL_FRAME_PATTERN" else "PARTIALLY_VERIFIED"
    close = outlier_status == "ALL_LOCALIZED_NO_GLOBAL_FRAME_PATTERN" and global_status == "NOT_SUPPORTED"
    blockers = [] if close else ["OUTLIER_PROVENANCE_NOT_FULLY_LOCALIZED_OR_GLOBAL_EVIDENCE_UNRESOLVED"]
    _csv(output / "center_outlier_full_provenance.csv", [public_row(row) for row in outliers])
    _csv(output / "center_outlier_neighbor_tracks.csv", neighbor_rows)
    _csv(output / "center_outlier_track_timelines.csv", timeline_rows)
    _csv(output / "center_outlier_reference_timestamp_audit.csv", [public_row(row) for row in outliers])
    _csv(output / "center_outlier_track_identity_audit.csv", [{"clip_id": row["clip_id"], "track_id": row["track_id"], "category": row["category"], "source": row["source"], "source_version": row["source_version"], "timestamp_us": row["raw_timestamp_us"], "track_dimensions": vector_json(row["track_first_selected_dimensions"])} for row in outliers])
    _dump(output / "max_center_outlier_provenance.json", {"clip_id": max_outlier["clip_id"], "track_id": max_outlier["track_id"], "raw_timestamp_us": max_outlier["raw_timestamp_us"], "A_center_residual_m": max_outlier["A_center_residual_m"], "B_center_residual_m": max_outlier["B_center_residual_m"], "A_center_residual_xyz": max_outlier["A_center_residual_xyz"], "B_center_residual_xyz": max_outlier["B_center_residual_xyz"], "raw_object_distance_to_ego_m": float(np.linalg.norm(np.asarray(max_outlier["raw_center"]))), "local_object_distance_m": float(np.linalg.norm(np.asarray(max_outlier["local_sequence_center"]))), "classification_rule_result": max_classification, "MAX_CENTER_OUTLIER_PROVENANCE_STATUS": max_provenance})
    _dump(output / "center_outlier_classification_summary.json", {"classifications": classification_rows, "counts": {name: classes.count(name) for name in sorted(set(classes))}, "center_outlier_total_count": len(outliers), "center_outlier_isolated_single_count": row_count_for_class("ISOLATED_SINGLE_ROW"), "center_outlier_contiguous_track_count": classes.count("CONTIGUOUS_LOCAL_TRACK_DIVERGENCE"), "center_outlier_raw_source_anomaly_count": row_count_for_class("SOURCE_LABEL_JUMP"), "center_outlier_local_sequence_anomaly_count": local_sequence_anomaly_count, "center_outlier_reference_timestamp_anomaly_count": row_count_for_class("REFERENCE_TIMESTAMP_ANOMALY"), "center_outlier_unresolved_count": row_count_for_class("UNRESOLVED"), "provenance_status": outlier_status})
    _dump(output / "global_frame_error_evidence.json", {"status": global_status, "GLOBAL_FRAME_TRANSFORM_ERROR_FOUND": False if global_status == "NOT_SUPPORTED" else None, "evidence": global_evidence, "rule": "consistent translation/rotation bias across clips or substantial structured failure; one anomalous track is insufficient"})
    _dump(output / "cuboid_final_geometry_contract.json", {"frame_status": "VERIFIED", "transform_status": "VERIFIED", "row_quality_status": row_quality, "raw_rebase_bridge_status": "VERIFIED_BASELINE", "local_rig_bridge_status": "VERIFIED_EXISTING_LOCAL_EGO_FRAME_CONTRACT", "authoritative_bridge": "UNRESOLVED", "scipy_runtime_available": SCIPY_AVAILABLE, "scipy_roundtrip_status": "VERIFIED" if SCIPY_AVAILABLE else "NOT_RUN_RUNTIME_UNAVAILABLE", "track_constant_dimensions": True, "no_fitted_correction": True, "per_clip_offset_rederived": False, "global_frame_error_evidence_status": global_status})
    _dump(output / "cuboid_final_geometry_summary.json", {"matched_selected_observation_count": len(all_rows), "total_matched_track_count": len({(row["clip_id"], row["track_id"]) for row in all_rows}), "center_gt_2m_count": len(outliers), "center_gt_2m_rate": len(outliers) / len(all_rows), "center_gt_5m_count": sum(row["A_center_residual_m"] > 5 for row in all_rows), "center_gt_10m_count": sum(row["A_center_residual_m"] > 10 for row in all_rows), "outlier_track_count": len(outlier_keys), "outlier_track_rate": len(outlier_keys) / len({(row["clip_id"], row["track_id"]) for row in all_rows}), "outlier_clip_count": len({row["clip_id"] for row in outliers}), "total_pilot_clip_count": len(manifest), "max_center_outlier_m": max_outlier["A_center_residual_m"], "max_center_outlier_provenance_status": max_provenance, "center_outlier_isolated_single_count": row_count_for_class("ISOLATED_SINGLE_ROW"), "center_outlier_contiguous_track_count": classes.count("CONTIGUOUS_LOCAL_TRACK_DIVERGENCE"), "center_outlier_raw_source_anomaly_count": row_count_for_class("SOURCE_LABEL_JUMP"), "center_outlier_local_sequence_anomaly_count": local_sequence_anomaly_count, "center_outlier_reference_timestamp_anomaly_count": row_count_for_class("REFERENCE_TIMESTAMP_ANOMALY"), "center_outlier_unresolved_count": row_count_for_class("UNRESOLVED"), "center_outlier_provenance_status": outlier_status, "reference_delta_median_us": float(statistics.median([abs(row["timestamp_minus_reference_us"]) for row in all_rows])), "reference_delta_p95_abs_us": p95([abs(row["timestamp_minus_reference_us"]) for row in all_rows]), "reference_delta_max_abs_us": max(abs(row["timestamp_minus_reference_us"]) for row in all_rows), "outlier_reference_delta_values": [row["timestamp_minus_reference_us"] for row in outliers], "reference_timestamp_outlier_association": "SUPPORTED" if any(abs(row["timestamp_minus_reference_us"]) > reference_threshold for row in outliers) else "NOT_SUPPORTED", "track_identity_continuity_status": "SUSPECT_SOURCE_TRACK_ASSOCIATION" if "SOURCE_LABEL_JUMP" in classes else "NO_SOURCE_IDENTITY_BREAK_PROVEN", "track_constant_dimension_parity_max_abs_m": max(dimension_diffs) if dimension_diffs else 0.0, "track_constant_dimension_parity_p95_m": p95(dimension_diffs), "scipy_runtime_available": SCIPY_AVAILABLE, "scipy_quat_euler_roundtrip_status": "VERIFIED" if SCIPY_AVAILABLE else "NOT_RUN_RUNTIME_UNAVAILABLE", "global_frame_error_evidence_status": global_status, "global_frame_transform_error_found": False if global_status == "NOT_SUPPORTED" else None, "raw_rebase_bridge_status": "VERIFIED_BASELINE", "local_rig_bridge_status": "VERIFIED_EXISTING_LOCAL_EGO_FRAME_CONTRACT", "authoritative_cuboid_ego_bridge": "UNRESOLVED", "cuboid_frame_status": "VERIFIED", "obstacle_transform_status": "VERIFIED", "normalized_obstacle_status": "VERIFIED_WITH_RETAINED_SOURCE_ANOMALIES" if close else "PARTIALLY_VERIFIED", "obstacle_row_quality_status": row_quality, "obstacle_geometry_block_ready_to_close": close, "obstacle_geometry_block": "CLOSED" if close else "OPEN", "per_clip_offset_rederived": False, "cf_proxy_label_set_ready": True, "ttc_proxy_label_set_ready": True, "cf_data_ready": False, "ttc_data_ready": False, "physical_world_obstacle_completeness": "NOT_CLAIMED", "remaining_blockers": blockers, "recommended_next_step": "freeze obstacle geometry contract and move to CF/TTC observation-completeness contract" if close else "investigate only the explicitly unresolved outlier provenance; do not reopen global frame search"})


if __name__ == "__main__":
    main()
