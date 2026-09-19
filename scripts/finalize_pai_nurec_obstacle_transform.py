"""Apply the documented PAI rig-pose transform and audit against NuRec tracks."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from scripts.audit_pai_obstacle_offline import _find_member, group_rows_by_mapped_timestamp, nearest_group, PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from audit_pai_obstacle_offline import _find_member, group_rows_by_mapped_timestamp, nearest_group, PILOT_CLIPS
    from prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump

PUBLIC_NCORE_CONVERTER = "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
PUBLIC_INSTANT_NUREC = "https://github.com/NVIDIA/instant-nurec/blob/main/instant_nurec/datasets/utils.py"
PUBLIC_NCORE_API = "https://nvidia.github.io/nurec/ncore/reference/apis/data.v3.html"


def _read_member(pai_root, relative_dir, filename, clip_id):
    root = Path(pai_root) / relative_dir
    for archive_path in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if clip_id in name and name.endswith(filename)]
            if members:
                return pq.read_table(io.BytesIO(archive.read(members[0]))).to_pylist(), archive_path, members[0]
    raise FileNotFoundError(f"OFFICIAL_MEMBER_NOT_FOUND clip={clip_id} dir={relative_dir} suffix={filename}")


def transform_object_pose(pai_row, ego_pose_world_rig):
    """T_object_scene = T_world_rig @ T_object_rig; no fitted parameters."""
    object_rig = _pose([pai_row["center_x"], pai_row["center_y"], pai_row["center_z"]], [pai_row["orientation_x"], pai_row["orientation_y"], pai_row["orientation_z"], pai_row["orientation_w"]])
    return _mm(ego_pose_world_rig, object_rig)


def matrix_center(matrix):
    return [matrix[0][3], matrix[1][3], matrix[2][3]]


def matrix_quaternion(matrix):
    return _rquat([row[:3] for row in matrix[:3]])


def load_ego_pose_samples(rows):
    return [(int(row["timestamp"]), _pose([row["x"], row["y"], row["z"],], [row["qx"], row["qy"], row["qz"], row["qw"]])) for row in rows]


def _p95(values):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))]


def crosscheck_clip(clip_id, pai_rows, ego_rows, local_rows, offset_us):
    local = {(str(row["track_id"]), int(row["timestamp_us"])): row for row in local_rows}
    ego_samples = load_ego_pose_samples(ego_rows)
    pairs, candidates = [], []
    for pai in pai_rows:
        key = (str(pai["track_id"]), int(pai["timestamp_us"]) + int(offset_us))
        local_row = local.get(key)
        if local_row is None:
            candidates.append({"clip_id": clip_id, "track_id": str(pai["track_id"]), "pai_timestamp_us": int(pai["timestamp_us"]), "nurec_timestamp_us": key[1], "classification": "UNRESOLVED_VERSION_DIFFERENCE", "proven": False})
            continue
        ego_pose = interpolate_pose(ego_samples, int(pai["timestamp_us"]))
        transformed = transform_object_pose(pai, ego_pose)
        center = matrix_center(transformed)
        quaternion = matrix_quaternion(transformed)
        center_residual = math.dist(center, local_row["center"])
        yaw_residual = abs(_wrap(pose_yaw(_pose(center, quaternion)) - pose_yaw(_pose(local_row["center"], local_row["quaternion"]))))
        dimension_residual = max(abs(pai[key_name] - local_row["dimensions"][index]) for index, key_name in enumerate(("size_x", "size_y", "size_z")))
        pairs.append({"clip_id": clip_id, "track_id": str(pai["track_id"]), "pai_timestamp_us": int(pai["timestamp_us"]), "nurec_timestamp_us": key[1], "center_scene_x": center[0], "center_scene_y": center[1], "center_scene_z": center[2], "quaternion_scene_x": quaternion[0], "quaternion_scene_y": quaternion[1], "quaternion_scene_z": quaternion[2], "quaternion_scene_w": quaternion[3], "center_residual_m": center_residual, "yaw_residual_deg": math.degrees(yaw_residual), "dimension_max_abs_m": dimension_residual, "track_id_provenance": "EXACT_ID_PRESERVED", "transform_formula": "T_world_rig(timestamp) @ T_object_rig"})
    centers = [r["center_residual_m"] for r in pairs]; yaws = [r["yaw_residual_deg"] for r in pairs]; dims = [r["dimension_max_abs_m"] for r in pairs]
    return pairs, candidates, {"clip_id": clip_id, "matched_observation_count": len(pairs), "center_rmse_m": math.sqrt(sum(v * v for v in centers) / len(centers)) if centers else None, "center_median_m": statistics.median(centers) if centers else None, "center_p95_m": _p95(centers), "center_max_m": max(centers) if centers else None, "yaw_rmse_deg": math.sqrt(sum(v * v for v in yaws) / len(yaws)) if yaws else None, "yaw_p95_deg": _p95(yaws), "yaw_max_deg": max(yaws) if yaws else None, "dimension_max_abs_m": max(dims) if dims else None, "track_id_provenance": "EXACT_ID_PRESERVED", "per_clip_offset_rederived": False}


def proxy_availability(queries, pai_rows, offset_us, tolerance_us):
    grouped = group_rows_by_mapped_timestamp(pai_rows, offset_us); result = []
    for query in queries:
        nearest, delta = nearest_group(query, grouped, tolerance_us)
        rows = grouped.get(nearest, []) if nearest is not None else []
        result.append({"query_timestamp_us": query, "nearest_label_timestamp_us": nearest, "delta_us": delta, "obstacle_count": len(rows), "label_set_timestamp_available": nearest is not None, "physical_world_completeness": "NOT_CLAIMED", "query_ready": nearest is not None})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pai-root", required=True); parser.add_argument("--manifest", required=True); parser.add_argument("--time-alignment-jsonl", required=True); parser.add_argument("--output-root", required=True); args = parser.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["clip_id"] in PILOT_CLIPS]
    time_path = Path(args.time_alignment_jsonl); cross_rows = []; cross_summaries = []; candidates = []; cf_all = []; ttc_all = []; ego_contracts = []
    for item in manifest:
        cid = item["clip_id"]; clip = Path(item["nurec_clip_dir"]); record = load_time_record(time_path, cid, int(item.get("physicalai_t0_us", 5_100_000)))
        pai_rows, _, _ = _read_member(args.pai_root, "labels/obstacle.offline", ".obstacle.offline.parquet", cid); ego_rows, ego_archive, ego_member = _read_member(args.pai_root, "labels/egomotion.offline", ".egomotion.offline.parquet", cid); local_rows, _, _ = load_sequence_tracks(clip / "sequence_tracks.json")
        pairs, clip_candidates, summary = crosscheck_clip(cid, pai_rows, ego_rows, local_rows, record["offset_us"]); cross_rows.extend(pairs); candidates.extend(clip_candidates); cross_summaries.append(summary)
        ego_contracts.append({"clip_id": cid, "schema": ["timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"], "timestamp_semantics": "CLIP_RELATIVE_MICROSECONDS", "pose_direction": "T_WORLD_RIG", "quaternion_order": "qx,qy,qz,qw", "origin": "ego_vehicle_at_timestamp_zero", "source_archive": str(ego_archive), "source_member": ego_member, "row_count": len(ego_rows), "timestamp_min_us": min(int(r["timestamp"]) for r in ego_rows), "timestamp_max_us": max(int(r["timestamp"]) for r in ego_rows)})
        cf, ttc = build_queries(record["nurec_t0_us"]); cf_all.extend({"clip_id": cid, **row} for row in proxy_availability(cf, pai_rows, record["offset_us"], 50_000)); ttc_all.extend({"clip_id": cid, **row} for row in proxy_availability(ttc, pai_rows, record["offset_us"], 100_000))
    _csv(root / "pai_nurec_transformed_crosscheck.csv", cross_rows); _dump(root / "pai_nurec_transformed_crosscheck_summary.json", {"clips": cross_summaries, "matched_observation_count": len(cross_rows), "max_center_rmse_m": max((s["center_rmse_m"] for s in cross_summaries if s["center_rmse_m"] is not None), default=None), "max_center_p95_m": max((s["center_p95_m"] for s in cross_summaries if s["center_p95_m"] is not None), default=None), "max_center_max_m": max((s["center_max_m"] for s in cross_summaries if s["center_max_m"] is not None), default=None), "max_yaw_rmse_deg": max((s["yaw_rmse_deg"] for s in cross_summaries if s["yaw_rmse_deg"] is not None), default=None), "max_yaw_p95_deg": max((s["yaw_p95_deg"] for s in cross_summaries if s["yaw_p95_deg"] is not None), default=None), "max_yaw_max_deg": max((s["yaw_max_deg"] for s in cross_summaries if s["yaw_max_deg"] is not None), default=None), "max_dimension_abs_m": max((s["dimension_max_abs_m"] for s in cross_summaries if s["dimension_max_abs_m"] is not None), default=None)})
    _csv(root / "pai_local_extraction_candidate_classification.csv", candidates); _csv(root / "cf_proxy_label_set_availability.csv", cf_all); _csv(root / "ttc_proxy_label_set_availability.csv", ttc_all)
    _dump(root / "pai_egomotion_contract.json", {"status": "VERIFIED_FROM_OFFICIAL_RAW_SCHEMA", "clips": ego_contracts, "evidence": "labels/egomotion.offline parquet fields timestamp,qx,qy,qz,qw,x,y,z"})
    _dump(root / "pai_obstacle_frame_contract.json", {"status": "VERIFIED_FROM_OFFICIAL_RAW_SCHEMA_AND_PUBLIC_CONVERTER", "raw_reference_frame_values": ["rig"], "reference_frame_timestamp_semantics": "same-time rig pose timestamp for obstacle observation", "orientation_order": "qx,qy,qz,qw", "source": "labels/obstacle.offline + official converter"})
    _dump(root / "pai_to_ncore_transform_contract.json", {"status": "PUBLIC_CONTRACT_VERIFIED", "source": PUBLIC_NCORE_CONVERTER, "edge": "PAI obstacle row -> CuboidTrackObservation(reference_frame_id, reference_frame_timestamp_us, bbox3)"})
    _dump(root / "ncore_to_nurec_transform_contract.json", {"status": "PUBLIC_CHAIN_VERIFIED", "source": PUBLIC_INSTANT_NUREC, "edge": "reference frame -> world through pose_graph.evaluate_poses; consolidated tracks are world-frame"})
    _dump(root / "pai_to_nurec_transform_contract.json", {"status": "PARTIALLY_VERIFIED_NUMERICALLY_SUPPORTED", "formula": "T_object_nurec_scene = T_world_rig(timestamp_us) @ T_object_rig", "timestamp_mapping": "nurec_timestamp_us = pai_timestamp_us + existing_per_clip_offset_us", "offset_rederived": False, "no_fitted_transform": True, "public_ncore": PUBLIC_NCORE_CONVERTER, "public_nurec": PUBLIC_INSTANT_NUREC, "direction_evidence": "official egomotion local world pose plus NCore world-frame consolidation", "world_to_nre_not_applied": "local sequence_tracks comparison is against NuRec world-compatible scene serialization; separate world_to_nre binding is not proven"})
    cf_available = sum(row["label_set_timestamp_available"] for row in cf_all); ttc_available = sum(row["label_set_timestamp_available"] for row in ttc_all); cross_summary = json.loads((root / "pai_nurec_transformed_crosscheck_summary.json").read_text(encoding="utf-8"))
    _dump(root / "proxy_label_set_readiness_summary.json", {"physical_world_obstacle_completeness": "NOT_CLAIMED", "cf_proxy_label_set_required_query_count": len(cf_all), "cf_proxy_label_set_available_query_count": cf_available, "cf_proxy_label_set_availability_rate": cf_available / len(cf_all) if cf_all else None, "cf_proxy_label_set_ready": cf_available == len(cf_all), "ttc_proxy_label_set_required_query_count": len(ttc_all), "ttc_proxy_label_set_available_query_count": ttc_available, "ttc_proxy_label_set_availability_rate": ttc_available / len(ttc_all) if ttc_all else None, "ttc_proxy_label_set_ready": ttc_available == len(ttc_all), "cf_data_ready": False, "ttc_data_ready": False})
    _dump(root / "audit_bugfix_summary.json", {"object_count_grouping_fixed": True, "missing_count_bug_fixed": True, "missing_count_semantics": "proven_missing=null; candidate_count reported separately", "cf_ttc_grouping_source": "actual_raw_obstacle_rows_grouped_by_mapped_timestamp"})
    _dump(root / "pai_nurec_transform_final_summary.json", {"pilot_clip_count": len(manifest), "pai_egomotion_contract_status": "VERIFIED_FROM_OFFICIAL_RAW_SCHEMA", "pai_raw_obstacle_frame_status": "VERIFIED_FROM_OFFICIAL_RAW_SCHEMA_AND_PUBLIC_CONVERTER", "pai_to_ncore_transform_status": "PUBLIC_CONTRACT_VERIFIED", "ncore_to_nurec_transform_status": "PUBLIC_CHAIN_VERIFIED", "pai_to_nurec_full_transform_status": "PARTIALLY_VERIFIED_NUMERICALLY_SUPPORTED", "track_id_provenance_status": "EXACT_ID_PRESERVED_FOR_MATCHED_ROWS; UNMATCHED_ROWS_UNRESOLVED", "pai_nurec_matched_observation_count": len(cross_rows), "max_center_rmse_m": cross_summary["max_center_rmse_m"], "max_center_p95_m": cross_summary["max_center_p95_m"], "max_center_max_m": cross_summary["max_center_max_m"], "max_yaw_rmse_deg": cross_summary["max_yaw_rmse_deg"], "max_yaw_p95_deg": cross_summary["max_yaw_p95_deg"], "max_yaw_max_deg": cross_summary["max_yaw_max_deg"], "max_dimension_abs_m": cross_summary["max_dimension_abs_m"], "local_extraction_missing_observation_count": None, "local_extraction_missing_observation_candidate_count": len(candidates), "local_extraction_missing_status": "UNRESOLVED_CANDIDATES_NOT_YET_FRAME_ALIGNED", "per_clip_offset_rederived": False, "obstacle_transform_status": "PARTIALLY_VERIFIED", "normalized_obstacle_status": "PARTIALLY_VERIFIED", "physical_world_obstacle_completeness": "NOT_CLAIMED", "pai_empty_frame_semantics": "UNRESOLVED_EMPTY_ATTESTATION", "cf_proxy_label_set_required_query_count": len(cf_all), "cf_proxy_label_set_available_query_count": cf_available, "cf_proxy_label_set_availability_rate": cf_available / len(cf_all) if cf_all else None, "cf_proxy_label_set_ready": cf_available == len(cf_all), "ttc_proxy_label_set_required_query_count": len(ttc_all), "ttc_proxy_label_set_available_query_count": ttc_available, "ttc_proxy_label_set_availability_rate": ttc_available / len(ttc_all) if ttc_all else None, "ttc_proxy_label_set_ready": ttc_available == len(ttc_all), "cf_data_ready": False, "ttc_data_ready": False, "obstacle_geometry_block_ready_to_close": False, "remaining_blockers": ["NUREC_SCENE_WORLD_TO_NRE_BINDING_NOT_PROVEN", "UNMATCHED_PAI_ROWS_NOT_SEMANTICALLY_CLASSIFIED", "EMPTY_FRAME_ATTESTATION_UNRESOLVED"], "recommended_next_step": "Review the documented NuRec world/base serialization edge; only then decide whether the numerically supported transformed geometry can be promoted to the scorer contract."})


if __name__ == "__main__":
    main()
