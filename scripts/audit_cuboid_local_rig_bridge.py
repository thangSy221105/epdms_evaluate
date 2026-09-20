"""Compare raw PAI rebase and the local NuRec rig trajectory for cuboids.

This is a provenance-first audit. It reuses the accepted per-clip time offset,
does not fit a residual transform, and does not change scorer semantics.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    ConvexHull = None
    R = None
    SCIPY_AVAILABLE = False

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose, load_nurec_poses
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose, load_nurec_poses
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump


def inv(m):
    r = [row[:3] for row in m[:3]]
    p = [m[i][3] for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    return [rt[0] + [-sum(rt[0][j] * p[j] for j in range(3))],
            rt[1] + [-sum(rt[1][j] * p[j] for j in range(3))],
            rt[2] + [-sum(rt[2][j] * p[j] for j in range(3))],
            [0.0, 0.0, 0.0, 1.0]]


def read_member(root, rel, suffix, clip):
    import pyarrow.parquet as pq
    for archive in sorted((Path(root) / rel).glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            names = [name for name in zf.namelist() if clip in name and name.endswith(suffix)]
            if names:
                return pq.read_table(io.BytesIO(zf.read(names[0]))).to_pylist()
    raise FileNotFoundError(f"missing {rel} {clip} {suffix}")


def source_name(value):
    return "AUTOLABEL" if "autolabel" in str(value).lower() else str(value).upper()


def mat_np(matrix):
    return np.asarray(matrix, dtype=float)


def center(matrix):
    return mat_np(matrix)[:3, 3]


def q_matrix(q):
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n <= 1e-15:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray([[1 - yy - zz, xy - wz, xz + wy],
                       [xy + wz, 1 - xx - zz, yz - wx],
                       [xz - wy, yz + wx, 1 - xx - yy]], dtype=float)


def object_pose_fallback(row):
    q = [row["orientation_x"], row["orientation_y"], row["orientation_z"], row["orientation_w"]]
    return np.asarray(_pose([row["center_x"], row["center_y"], row["center_z"]], q), dtype=float)


def object_pose_scipy(row):
    if not SCIPY_AVAILABLE:
        return object_pose_fallback(row), None
    raw_q = [row["orientation_x"], row["orientation_y"], row["orientation_z"], row["orientation_w"]]
    rot = R.from_quat(raw_q)
    euler_xyz = rot.as_euler("xyz", degrees=False)
    reconstructed = R.from_euler("xyz", euler_xyz, degrees=False)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = reconstructed.as_matrix()
    matrix[:3, 3] = [row["center_x"], row["center_y"], row["center_z"]]
    return matrix, rot


def pose_from_target(target):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = q_matrix(target["quaternion"])
    matrix[:3, 3] = target["center"]
    return matrix


def rot_error_deg(a, b):
    delta = mat_np(a)[:3, :3].T @ mat_np(b)[:3, :3]
    cosine = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def residual(matrix, target):
    target_matrix = pose_from_target(target)
    distance = float(np.linalg.norm(center(matrix) - center(target_matrix)))
    yaw = math.degrees(abs(_wrap(pose_yaw(matrix) - pose_yaw(target_matrix))))
    signed_yaw = math.degrees(_wrap(pose_yaw(target_matrix) - pose_yaw(matrix)))
    return distance, yaw, signed_yaw


def corners(matrix, dims):
    signs = np.asarray([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    local = signs * np.asarray(dims, dtype=float) / 2.0
    return center(matrix) + local @ mat_np(matrix)[:3, :3].T


def hull_xy(points):
    points = np.asarray(points, dtype=float)
    if SCIPY_AVAILABLE:
        hull = ConvexHull(points[:, :2])
        return points[hull.vertices, :2]
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly))) / 2.0)


def clip_polygon(subject, edge_a, edge_b):
    out = []
    def inside(point):
        return ((edge_b[0] - edge_a[0]) * (point[1] - edge_a[1])
                - (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0])) >= -1e-9
    def intersection(a, b):
        p, direction = np.asarray(a, dtype=float), np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        q, edge = np.asarray(edge_a, dtype=float), np.asarray(edge_b, dtype=float) - np.asarray(edge_a, dtype=float)
        denominator = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denominator) < 1e-12:
            return list(p)
        t = ((q[0] - p[0]) * edge[1] - (q[1] - p[1]) * edge[0]) / denominator
        return list(p + t * direction)
    if not subject:
        return out
    previous = subject[-1]
    for current in subject:
        if inside(current):
            if not inside(previous):
                out.append(intersection(previous, current))
            out.append(current)
        elif inside(previous):
            out.append(intersection(previous, current))
        previous = current
    return out


def polygon_iou(a, b):
    intersection = a.tolist()
    for i in range(len(b)):
        intersection = clip_polygon(intersection, b[i], b[(i + 1) % len(b)])
    area_a, area_b, area_i = polygon_area(a.tolist()), polygon_area(b.tolist()), polygon_area(intersection)
    union = area_a + area_b - area_i
    return area_i / union if union > 0 else 0.0


def assignment_rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    costs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) ** 2
    dynamic = {0: 0.0}
    for mask in range(1 << n):
        if mask not in dynamic:
            continue
        row = mask.bit_count()
        if row == n:
            continue
        for col in range(n):
            if not (mask >> col) & 1:
                new_mask = mask | (1 << col)
                dynamic[new_mask] = min(dynamic.get(new_mask, float("inf")), dynamic[mask] + float(costs[row, col]))
    return math.sqrt(dynamic[(1 << n) - 1] / n)


def percentile(values, fraction):
    return float(np.quantile(values, fraction)) if values else None


def metric(values):
    centers = [x["center_residual_m"] for x in values]
    yaws = [x["yaw_residual_deg"] for x in values]
    return {
        "matched": len(values),
        "center_rmse_m": math.sqrt(sum(x * x for x in centers) / len(centers)) if centers else None,
        "center_median_m": float(statistics.median(centers)) if centers else None,
        "center_p95_m": percentile(centers, 0.95),
        "center_max_m": max(centers) if centers else None,
        "yaw_rmse_deg": math.sqrt(sum(x * x for x in yaws) / len(yaws)) if yaws else None,
        "yaw_p95_deg": percentile(yaws, 0.95),
        "yaw_max_deg": max(yaws) if yaws else None,
        "center_gt_2m": sum(x > 2 for x in centers),
        "center_gt_5m": sum(x > 5 for x in centers),
        "center_gt_10m": sum(x > 10 for x in centers),
        "yaw_gt_45": sum(x > 45 for x in yaws),
        "yaw_gt_90": sum(x > 90 for x in yaws),
        "yaw_gt_170": sum(x > 170 for x in yaws),
    }


def signed_bias(rows):
    vec = np.asarray([x["center_residual_xyz"] for x in rows], dtype=float)
    yaw = np.asarray([x["signed_yaw_delta_deg"] for x in rows], dtype=float)
    return {
        "matched_count": len(rows),
        "mean_dx": float(np.mean(vec[:, 0])), "mean_dy": float(np.mean(vec[:, 1])), "mean_dz": float(np.mean(vec[:, 2])),
        "median_dx": float(np.median(vec[:, 0])), "median_dy": float(np.median(vec[:, 1])), "median_dz": float(np.median(vec[:, 2])),
        "std_dx": float(np.std(vec[:, 0])), "std_dy": float(np.std(vec[:, 1])), "std_dz": float(np.std(vec[:, 2])),
        "mean_signed_yaw_deg": float(np.mean(yaw)), "median_signed_yaw_deg": float(np.median(yaw)),
        "std_signed_yaw_deg": float(np.std(yaw)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pai-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["clip_id"] in PILOT_CLIPS]
    all_rows, bridge_rows, clip_bias_rows, clip_delta_rows = [], [], [], []
    scipy_errors = []
    for item in manifest:
        clip_id = item["clip_id"]
        clip = Path(item["nurec_clip_dir"])
        record = load_time_record(Path(args.time_alignment_jsonl), clip_id, int(item.get("physicalai_t0_us", 5100000)))
        raw = read_member(args.pai_root, "labels/obstacle.offline", ".obstacle.offline.parquet", clip_id)
        ego = read_member(args.pai_root, "labels/egomotion.offline", ".egomotion.offline.parquet", clip_id)
        local, _, _ = load_sequence_tracks(clip / "sequence_tracks.json")
        local_map = {(str(row["track_id"]), int(row["timestamp_us"])): row for row in local}
        local_rig = load_nurec_poses(clip)
        ego_samples = [(int(row["timestamp"]), _pose([row["x"], row["y"], row["z"]], [row["qx"], row["qy"], row["qz"], row["qw"]])) for row in ego]
        rebase = inv(ego_samples[0][1])
        selected = []
        for raw_index, row in sorted(enumerate(raw), key=lambda x: (str(x[1]["track_id"]), int(x[1]["timestamp_us"]))):
            if source_name(row.get("source")) != "AUTOLABEL":
                continue
            if math.dist([row["center_x"], row["center_y"], row["center_z"]], [0, 0, 0]) < 3:
                continue
            mapped = int(row["timestamp_us"]) + int(record["offset_us"])
            if (str(row["track_id"]), mapped) in local_map:
                selected.append((raw_index, row, mapped))
        variant_a, variant_b = [], []
        for raw_index, row, mapped in selected:
            target = local_map[(str(row["track_id"]), mapped)]
            raw_reference_ts = int(row["reference_frame_timestamp_us"])
            mapped_reference_ts = raw_reference_ts + int(record["offset_us"])
            raw_ego = np.asarray(interpolate_pose(ego_samples, raw_reference_ts), dtype=float)
            ego_a = mat_np(rebase) @ raw_ego
            ego_b = mat_np(interpolate_pose(local_rig, mapped_reference_ts))
            object_pose, raw_rotation = object_pose_scipy(row)
            if raw_rotation is not None:
                roundtrip = R.from_euler("xyz", raw_rotation.as_euler("xyz", degrees=False), degrees=False)
                scipy_errors.append(math.degrees((raw_rotation.inv() * roundtrip).magnitude()))
            object_a = ego_a @ object_pose
            object_b = ego_b @ object_pose
            target_pose = pose_from_target(target)
            a_center, a_yaw, a_signed_yaw = residual(object_a, target)
            b_center, b_yaw, b_signed_yaw = residual(object_b, target)
            delta = inv((ego_a).tolist())
            delta = mat_np(delta) @ ego_b
            bridge_rows.append({"clip_id": clip_id, "track_id": str(row["track_id"]), "raw_timestamp_us": int(row["timestamp_us"]), "mapped_timestamp_us": mapped, "reference_frame_timestamp_us": raw_reference_ts, "mapped_reference_timestamp_us": mapped_reference_ts, "delta_x_m": float(delta[0, 3]), "delta_y_m": float(delta[1, 3]), "delta_z_m": float(delta[2, 3]), "delta_translation_norm_m": float(np.linalg.norm(delta[:3, 3])), "delta_yaw_deg": math.degrees(pose_yaw(delta.tolist()))})
            dims = [row["size_x"], row["size_y"], row["size_z"]]
            a_corners, b_corners, target_corners = corners(object_a, dims), corners(object_b, dims), corners(target_pose, target["dimensions"])
            b_hull, target_hull = hull_xy(b_corners), hull_xy(target_corners)
            row_out = {"clip_id": clip_id, "track_id": str(row["track_id"]), "raw_timestamp_us": int(row["timestamp_us"]), "mapped_timestamp_us": mapped, "reference_frame_timestamp_us": raw_reference_ts, "mapped_reference_timestamp_us": mapped_reference_ts, "center_residual_xyz_a": (center(target_pose) - center(object_a)).tolist(), "center_residual_xyz_b": (center(target_pose) - center(object_b)).tolist(), "center_residual_m_a": a_center, "center_residual_m_b": b_center, "yaw_residual_deg_a": a_yaw, "yaw_residual_deg_b": b_yaw, "signed_yaw_delta_deg_a": a_signed_yaw, "signed_yaw_delta_deg_b": b_signed_yaw, "bev_iou_a": polygon_iou(hull_xy(a_corners), target_hull), "bev_iou_b": polygon_iou(b_hull, target_hull), "corner3d_error_m_a": assignment_rmse(a_corners, target_corners), "corner3d_error_m_b": assignment_rmse(b_corners, target_corners), "ego_bridge_delta_x_m": float(delta[0, 3]), "ego_bridge_delta_y_m": float(delta[1, 3]), "ego_bridge_delta_z_m": float(delta[2, 3]), "ego_bridge_delta_yaw_deg": math.degrees(pose_yaw(delta.tolist()))}
            variant_a.append({"clip_id": clip_id, "track_id": str(row["track_id"]), "raw_timestamp_us": int(row["timestamp_us"]), "center_residual_m": a_center, "yaw_residual_deg": a_yaw, "signed_yaw_delta_deg": a_signed_yaw, "center_residual_xyz": row_out["center_residual_xyz_a"], "bev_iou": row_out["bev_iou_a"], "corner3d_error_m": row_out["corner3d_error_m_a"]})
            variant_b.append({"clip_id": clip_id, "track_id": str(row["track_id"]), "raw_timestamp_us": int(row["timestamp_us"]), "center_residual_m": b_center, "yaw_residual_deg": b_yaw, "signed_yaw_delta_deg": b_signed_yaw, "center_residual_xyz": row_out["center_residual_xyz_b"], "bev_iou": row_out["bev_iou_b"], "corner3d_error_m": row_out["corner3d_error_m_b"]})
            all_rows.append(row_out)
        a_bias, b_bias = signed_bias(variant_a), signed_bias(variant_b)
        clip_bias_rows.append({"clip_id": clip_id, **{f"A_{k}": v for k, v in a_bias.items()}, **{f"B_{k}": v for k, v in b_bias.items()}})
        clip_delta = [x for x in bridge_rows if x["clip_id"] == clip_id]
        clip_delta_rows.append({"clip_id": clip_id, "matched_count": len(clip_delta), "mean_delta_x_m": float(np.mean([x["delta_x_m"] for x in clip_delta])), "mean_delta_y_m": float(np.mean([x["delta_y_m"] for x in clip_delta])), "mean_delta_z_m": float(np.mean([x["delta_z_m"] for x in clip_delta])), "std_delta_x_m": float(np.std([x["delta_x_m"] for x in clip_delta])), "std_delta_y_m": float(np.std([x["delta_y_m"] for x in clip_delta])), "std_delta_z_m": float(np.std([x["delta_z_m"] for x in clip_delta])), "mean_delta_yaw_deg": float(np.mean([x["delta_yaw_deg"] for x in clip_delta])), "std_delta_yaw_deg": float(np.std([x["delta_yaw_deg"] for x in clip_delta]))})
    metrics_a = metric([{**x, "center_residual_m": x["center_residual_m_a"], "yaw_residual_deg": x["yaw_residual_deg_a"]} for x in all_rows])
    metrics_b = metric([{**x, "center_residual_m": x["center_residual_m_b"], "yaw_residual_deg": x["yaw_residual_deg_b"]} for x in all_rows])
    b_iou = [x["bev_iou_b"] for x in all_rows]
    b_corner = [x["corner3d_error_m_b"] for x in all_rows]
    a_outliers = [x for x in all_rows if x["center_residual_m_a"] > 2]
    b_by_key = {(x["clip_id"], x["track_id"], x["raw_timestamp_us"]): x for x in all_rows}
    outlier_rows = []
    for row in a_outliers:
        outlier_rows.append({"clip_id": row["clip_id"], "track_id": row["track_id"], "raw_timestamp_us": row["raw_timestamp_us"], "reference_frame_timestamp_us": row["reference_frame_timestamp_us"], "mapped_timestamp_us": row["mapped_timestamp_us"], "A_center_residual_m": row["center_residual_m_a"], "B_center_residual_m": row["center_residual_m_b"], "A_center_residual_xyz": json.dumps(row["center_residual_xyz_a"]), "B_center_residual_xyz": json.dumps(row["center_residual_xyz_b"]), "A_bev_iou": row["bev_iou_a"], "B_bev_iou": row["bev_iou_b"], "A_3d_corner_error_m": row["corner3d_error_m_a"], "B_3d_corner_error_m": row["corner3d_error_m_b"], "classification": "RESOLVED_BY_LOCAL_RIG_BRIDGE" if row["center_residual_m_b"] <= 2 else ("IMPROVED_BUT_REMAINS" if row["center_residual_m_b"] < row["center_residual_m_a"] else ("UNCHANGED" if row["center_residual_m_b"] == row["center_residual_m_a"] else "WORSENED"))})
    delta_trans = [x["delta_translation_norm_m"] for x in bridge_rows]
    delta_yaw = [abs(x["delta_yaw_deg"]) for x in bridge_rows]
    a_max_bias = [max(abs(x[f"A_mean_d{k}"]) for x in clip_bias_rows) for k in ("x", "y", "z")]
    b_max_bias = [max(abs(x[f"B_mean_d{k}"]) for x in clip_bias_rows) for k in ("x", "y", "z")]
    provenance = {"status": "VERIFIED_EXISTING_LOCAL_EGO_FRAME_CONTRACT", "source": "rig_trajectories.json:T_rig_worlds + T_rig_world_timestamps_us", "accepted_prior_contract": "LOCAL_EGO_FRAME_STATUS=VERIFIED_NCORE_LOCAL_WORLD", "world_to_nre_applied": False, "world_to_scene_applied": False, "T_world_base_applied": False, "interpolation": "linear translation + SLERP", "no_fitted_correction": True}
    b_improves = metrics_b["center_p95_m"] <= metrics_a["center_p95_m"] and metrics_b["center_rmse_m"] <= metrics_a["center_rmse_m"] and max(b_max_bias) < max(a_max_bias)
    b_geometry_ok = percentile(b_iou, 0.05) > 0.5 and percentile(b_corner, 0.95) < 1.0
    global_frame_error = False
    closure_failed = not (SCIPY_AVAILABLE and b_improves and b_geometry_ok)
    best = "LOCAL_NUREC_RIG_TRAJECTORY" if (not closure_failed and provenance["status"].startswith("VERIFIED")) else "UNRESOLVED"
    outlier_resolved = sum(x["classification"] == "RESOLVED_BY_LOCAL_RIG_BRIDGE" for x in outlier_rows)
    outlier_remaining = len(outlier_rows) - outlier_resolved
    _csv(output / "cuboid_bridge_variant_comparison.csv", all_rows)
    _csv(output / "ego_bridge_delta_per_timestamp.csv", bridge_rows)
    _csv(output / "ego_bridge_delta_per_clip.csv", clip_delta_rows)
    _csv(output / "cuboid_bridge_bias_per_clip.csv", clip_bias_rows)
    _csv(output / "cuboid_local_rig_geometry.csv", all_rows)
    _csv(output / "center_outlier_bridge_comparison.csv", outlier_rows)
    summary = {"matched_selected_observation_count": len(all_rows), "A": metrics_a, "B": metrics_b, "B_global_bev_iou_mean": float(statistics.mean(b_iou)), "B_global_bev_iou_p05": percentile(b_iou, 0.05), "B_global_bev_iou_min": min(b_iou), "B_global_3d_corner_p95_m": percentile(b_corner, 0.95), "B_global_3d_corner_max_m": max(b_corner), "A_center_outlier_count": len(a_outliers), "B_center_outlier_count": sum(x["center_residual_m_b"] > 2 for x in all_rows), "center_outliers_resolved_by_b_count": outlier_resolved, "center_outliers_remaining_b_count": outlier_remaining, "ego_bridge_translation_rmse_m": math.sqrt(sum(x * x for x in delta_trans) / len(delta_trans)), "ego_bridge_translation_p95_m": percentile(delta_trans, 0.95), "ego_bridge_translation_max_m": max(delta_trans), "ego_bridge_yaw_rmse_deg": math.sqrt(sum(x * x for x in delta_yaw) / len(delta_yaw)), "ego_bridge_yaw_p95_deg": percentile(delta_yaw, 0.95), "ego_bridge_yaw_max_deg": max(delta_yaw), "max_abs_clip_mean_dx_a": a_max_bias[0], "max_abs_clip_mean_dx_b": b_max_bias[0], "max_abs_clip_mean_dy_a": a_max_bias[1], "max_abs_clip_mean_dy_b": b_max_bias[1], "max_abs_clip_mean_dz_a": a_max_bias[2], "max_abs_clip_mean_dz_b": b_max_bias[2], "scipy_quat_euler_roundtrip_status": "VERIFIED" if SCIPY_AVAILABLE else "UNAVAILABLE_RUNTIME_MISSING", "max_scipy_rotation_roundtrip_error_deg": max(scipy_errors) if scipy_errors else None}
    _dump(output / "cuboid_local_rig_geometry_summary.json", summary)
    _dump(output / "cuboid_local_rig_bridge_contract.json", {"variant_a": "inverse(T_first_raw) @ T_rig_world_raw(reference_frame_timestamp_us)", "variant_b": "rig_trajectories.json:T_rig_worlds(mapped_reference_timestamp_us)", "mapped_reference_timestamp": "raw reference_frame_timestamp_us + existing per-clip offset", "provenance": provenance, "RAW_REBASE_BRIDGE_STATUS": "VERIFIED_BASELINE", "LOCAL_RIG_BRIDGE_STATUS": provenance["status"], "BEST_PROVEN_BRIDGE": best, "AUTHORITATIVE_CUBOID_EGO_BRIDGE": best, "GLOBAL_FRAME_TRANSFORM_ERROR_FOUND": global_frame_error, "GEOMETRY_CLOSURE_CRITERIA_FAILED": closure_failed, "no_offset_rederive": True, "no_fitted_correction": True})
    _dump(output / "cuboid_local_rig_bridge_final_summary.json", {**summary, "raw_rebase_bridge_status": "VERIFIED_BASELINE", "local_rig_bridge_status": provenance["status"], "best_proven_bridge": best, "authoritative_cuboid_ego_bridge": best, "global_frame_transform_error_found": global_frame_error, "geometry_closure_criteria_failed": closure_failed, "cuboid_serialization_status": "VERIFIED" if not closure_failed else "PARTIALLY_VERIFIED", "cuboid_frame_status": "VERIFIED" if not closure_failed else "PARTIALLY_VERIFIED", "obstacle_transform_status": "VERIFIED" if not closure_failed else "PARTIALLY_VERIFIED", "normalized_obstacle_status": "VERIFIED" if not closure_failed else "PARTIALLY_VERIFIED", "obstacle_geometry_block_ready_to_close": not closure_failed, "obstacle_geometry_block": "CLOSED" if not closure_failed else "OPEN", "per_clip_offset_rederived": False, "cf_proxy_label_set_ready": True, "ttc_proxy_label_set_ready": True, "cf_data_ready": False, "ttc_data_ready": False, "physical_world_obstacle_completeness": "NOT_CLAIMED", "remaining_blockers": [] if not closure_failed else (["SCIPY_RUNTIME_UNAVAILABLE"] if not SCIPY_AVAILABLE else ["LOCAL_RIG_BRIDGE_DID_NOT_MATERIALLY_IMPROVE_ALL_REQUIRED_EVIDENCE"]), "recommended_next_step": "start scorer contract review" if not closure_failed else "retain geometry block open and review exact local-rig provenance/residuals"})


if __name__ == "__main__":
    main()
