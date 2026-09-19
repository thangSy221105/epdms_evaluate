"""Phase 0 Audit: Environment dependencies and data contract validation."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import EvaluationConfig
from .io_jsonl import compute_file_sha256, iter_jsonl, read_jsonl_indexed
from .map_loader import inspect_clip_map_status, load_lane_polygons_for_clip
from .observation_contract import (
    build_ttc_projection_timestamps,
    evaluate_query_coverage,
    expand_confirmed_empty_timestamps,
    index_and_filter_obstacles,
)
from .score_record import _coordinate_contract, _observation_attestation
from .time_contract import (
    TimeContractError,
    normalize_trajectory_timeline,
    prepare_trajectory_window,
    resolve_time_origin,
)


def audit_environment() -> Dict[str, Any]:
    """Inspects Python version and available packages without modifying environment."""
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_available": False,
        "pandas_available": False,
        "scipy_available": False,
        "shapely_available": False,
        "navsim_available": False,
        "nuplan_available": False,
    }

    try:
        import numpy as np
        env["numpy_available"] = True
        env["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import pandas as pd
        env["pandas_available"] = True
        env["pandas_version"] = pd.__version__
    except ImportError:
        pass

    try:
        import scipy
        env["scipy_available"] = True
        env["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    try:
        import shapely
        env["shapely_available"] = True
        env["shapely_version"] = shapely.__version__
    except ImportError:
        pass

    try:
        import navsim
        env["navsim_available"] = True
        env["navsim_version"] = getattr(navsim, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import nuplan
        env["nuplan_available"] = True
        env["nuplan_version"] = getattr(nuplan, "__version__", "unknown")
    except ImportError:
        pass

    env["proxy_possible"] = bool(env["numpy_available"] and env["pandas_available"])
    env["official_stage1_possible"] = bool(env["navsim_available"] and env["nuplan_available"])
    env["official_full_possible"] = bool(env["navsim_available"] and env["nuplan_available"])

    return env


def _trajectory_for_row(row: Dict[str, Any], alpha: Optional[float] = None) -> Any:
    if not isinstance(row, dict):
        return None
    if alpha is not None and float(alpha) != 0.0 and row.get("guided_waypoints") is not None:
        return row.get("guided_waypoints")
    if row.get("clean_waypoints") is not None:
        return row.get("clean_waypoints")
    if row.get("guided_waypoints") is not None:
        return row.get("guided_waypoints")
    trajectories = row.get("trajectories")
    if isinstance(trajectories, dict):
        return trajectories.get("clean") or trajectories.get("guided")
    return None


def _normalized_window(row: Dict[str, Any], raw: Any, t0_us: int, config: EvaluationConfig, role: str) -> Any:
    if raw is None:
        raise TimeContractError(f"{role} trajectory is missing")
    items = raw.tolist() if hasattr(raw, "tolist") and not isinstance(raw, list) else list(raw)
    window = prepare_trajectory_window(items, t0_us, config.future_poses, role=role)
    return normalize_trajectory_timeline(
        window,
        t0_us=t0_us,
        expected_frequency_hz=config.frequency_hz,
        expected_horizon_s=config.horizon_s,
        role=role,
        strict_grid=config.strict_mode,
    )


def _future_and_full_timestamps(timeline: Any, t0_us: int) -> Any:
    if timeline.includes_t0:
        return timeline.timestamps_us.copy()
    return np.concatenate(([int(t0_us)], timeline.timestamps_us))


def audit_time_alignment(config: EvaluationConfig, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Audit clock domains using the same normalized timeline as scoring."""
    pred_map = read_jsonl_indexed(config.prediction_jsonl, lambda r: f"{r.get('clip_id')}|{r.get('mode')}|{float(r.get('alpha', 0.0))}") if config.prediction_jsonl.is_file() else {}
    ctx_map = read_jsonl_indexed(config.context_jsonl, lambda r: str(r.get("clip_id", ""))) if config.context_jsonl.is_file() else {}
    gt_map = read_jsonl_indexed(config.ground_truth_jsonl, lambda r: str(r.get("clip_id", ""))) if config.ground_truth_jsonl.is_file() else {}
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for pred in pred_map.values():
        clip_id = str(pred.get("clip_id", ""))
        if clip_id in seen:
            continue
        seen.add(clip_id)
        ctx = ctx_map.get(clip_id, {})
        gt = gt_map.get(clip_id, {})
        t0_pred = pred.get("t0_us")
        t0_gt = gt.get("t0_us")
        t0_ctx = ctx.get("t0_us")
        status = "ALIGNED_DIRECT"
        reason = ""
        pred_timeline = None
        gt_timeline = None
        try:
            t0_us, _ = resolve_time_origin(pred, ctx, gt, strict_mode=True)
            pred_timeline = _normalized_window(pred, _trajectory_for_row(pred, pred.get("alpha", 0.0)), t0_us, config, "prediction")
            graw = next((gt.get(k) for k in ("ego_future_xyz", "expert_future", "future_waypoints") if gt.get(k) is not None), None)
            gt_timeline = _normalized_window(gt, graw, t0_us, config, "ground_truth")
            if not np.array_equal(
                _future_and_full_timestamps(pred_timeline, t0_us)[1:],
                _future_and_full_timestamps(gt_timeline, t0_us)[1:],
            ):
                status, reason = "TIMELINE_GRID_MISMATCH", "prediction and GT normalized grids differ"
        except Exception as exc:
            status, reason = type(exc).__name__, str(exc)

        obstacles = (((ctx.get("semantic_context") or {}).get("obstacle") or {}).get("all_obstacles") or [])
        obs_ts = []
        for obs in obstacles:
            if isinstance(obs, dict):
                value = obs.get("timestamp_micros")
                if value is None and isinstance(obs.get("key"), dict):
                    value = obs["key"].get("timestamp_micros")
                try:
                    if value is not None:
                        obs_ts.append(int(value))
                except (TypeError, ValueError):
                    pass
        ego_values = []
        for container in (ctx, (ctx.get("semantic_context") or {})):
            if isinstance(container, dict):
                for key_name in ("egomotion", "ego_motion", "ego_pose"):
                    candidate = container.get(key_name)
                    if isinstance(candidate, list):
                        for item in candidate:
                            if isinstance(item, dict):
                                value = item.get("timestamp_micros", item.get("t_us"))
                                try:
                                    if value is not None:
                                        ego_values.append(int(value))
                                except (TypeError, ValueError):
                                    pass
        if t0_pred is None or t0_gt is None or t0_ctx is None:
            status, reason = "INSUFFICIENT_ALIGNMENT_METADATA", "one or more source t0_us values are missing"
        elif len({int(t0_pred), int(t0_gt), int(t0_ctx)}) > 1:
            status, reason = "CONFLICTING_TIME_ORIGIN", "prediction/context/GT t0_us values disagree"
        elif obs_ts and pred_timeline is not None:
            start_us = int(_future_and_full_timestamps(pred_timeline, int(t0_pred))[0])
            end_us = int(_future_and_full_timestamps(pred_timeline, int(t0_pred))[-1])
            if max(obs_ts) < start_us - 100_000 or min(obs_ts) > end_us + 1_000_000:
                status, reason = "CLOCK_DOMAIN_MISMATCH", "obstacle timestamps do not overlap prediction clip clock"
        rows.append({
            "clip_id": clip_id, "prediction_t0_us": t0_pred, "gt_t0_us": t0_gt, "context_t0_us": t0_ctx,
            "prediction_first_time_us": int(pred_timeline.timestamps_us[0]) if pred_timeline is not None else None,
            "prediction_last_time_us": int(pred_timeline.timestamps_us[-1]) if pred_timeline is not None else None,
            "gt_first_time_us": int(gt_timeline.timestamps_us[0]) if gt_timeline is not None else None,
            "gt_last_time_us": int(gt_timeline.timestamps_us[-1]) if gt_timeline is not None else None,
            "obstacle_min_us": min(obs_ts) if obs_ts else None, "obstacle_max_us": max(obs_ts) if obs_ts else None,
            "egomotion_min_us": min(ego_values) if ego_values else None, "egomotion_max_us": max(ego_values) if ego_values else None,
            "candidate_pred_to_obstacle_offset_us": (int(t0_pred) - min(obs_ts)) if obs_ts and t0_pred is not None else None,
            "candidate_pred_to_egomotion_offset_us": (int(t0_pred) - min(ego_values)) if ego_values and t0_pred is not None else None,
            "time_alignment_status": status, "time_alignment_source": "normalized_timeline" if status == "ALIGNED_DIRECT" else "none",
            "time_alignment_confidence": 1.0 if status == "ALIGNED_DIRECT" else 0.0, "failure_reason": reason,
        })
        if max_rows is not None and len(rows) >= max_rows:
            break
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["time_alignment_status"]] = counts.get(row["time_alignment_status"], 0) + 1
    return {"rows": rows, "status_counts": counts, "clips_checked": len(rows), "ready": bool(rows) and all(r["time_alignment_status"] == "ALIGNED_DIRECT" for r in rows)}


def audit_coordinate_alignment(config: EvaluationConfig, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Audit frame and reference-point declarations without guessing transforms."""
    pred_map = read_jsonl_indexed(config.prediction_jsonl, lambda r: f"{r.get('clip_id')}|{r.get('mode')}|{float(r.get('alpha', 0.0))}") if config.prediction_jsonl.is_file() else {}
    ctx_map = read_jsonl_indexed(config.context_jsonl, lambda r: str(r.get("clip_id", ""))) if config.context_jsonl.is_file() else {}
    gt_map = read_jsonl_indexed(config.ground_truth_jsonl, lambda r: str(r.get("clip_id", ""))) if config.ground_truth_jsonl.is_file() else {}
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for pred in pred_map.values():
        clip_id = str(pred.get("clip_id", ""))
        if clip_id in seen:
            continue
        seen.add(clip_id)
        ctx = ctx_map.get(clip_id, {})
        gt = gt_map.get(clip_id, {})
        sc = ctx.get("semantic_context") if isinstance(ctx, dict) else {}
        sc = sc if isinstance(sc, dict) else {}
        def first(*containers: Any, keys: tuple[str, ...]) -> Optional[str]:
            for container in containers:
                if isinstance(container, dict):
                    for key in keys:
                        value = container.get(key)
                        if value is not None:
                            return str(value)
            return None
        pframe = first(pred, keys=("coordinate_frame", "frame", "prediction_frame"))
        gframe = first(gt, keys=("coordinate_frame", "frame", "gt_frame"))
        oframe = first(ctx, sc, keys=("obstacle_frame", "coordinate_frame", "frame"))
        mframe = first(ctx, sc, keys=("map_frame",))
        panchor = first(pred, keys=("reference_point", "anchor", "prediction_anchor"))
        ganchor = first(gt, keys=("reference_point", "anchor", "gt_anchor"))
        oanchor = first(ctx, sc, keys=("obstacle_anchor", "reference_point", "anchor"))
        manchor = first(ctx, sc, keys=("map_anchor", "reference_point", "anchor"))
        verified = bool(
            pframe and gframe and oframe and mframe and panchor and ganchor and oanchor and manchor
            and pframe == gframe == oframe == mframe and panchor == ganchor == oanchor == manchor
        )
        rows.append({"clip_id": clip_id, "prediction_frame": pframe, "gt_frame": gframe, "obstacle_frame": oframe, "map_frame": mframe, "prediction_anchor": panchor, "gt_anchor": ganchor, "obstacle_anchor": oanchor, "map_anchor": manchor, "transform_required": False if verified else None, "transform_source": "declared_metadata" if verified else None, "coordinate_alignment_verified": verified, "failure_reason": "" if verified else "COORDINATE_CONTRACT_UNRESOLVED"})
        if max_rows is not None and len(rows) >= max_rows:
            break
    return {"rows": rows, "clips_checked": len(rows), "ready": bool(rows) and all(row["coordinate_alignment_verified"] for row in rows)}


def audit_data_contracts(config: EvaluationConfig) -> Dict[str, Any]:
    """Thoroughly checks the 3 input files against schema contracts."""
    report: Dict[str, Any] = {
        "files_checked": {},
        "prediction_stats": {},
        "context_stats": {},
        "ground_truth_stats": {},
        "missing_records": [],
        "duplicate_records": [],
        "readiness": {},
    }

    # 1. Prediction inspection
    pred_path = config.prediction_jsonl
    report["files_checked"]["prediction"] = {
        "path": str(pred_path),
        "exists": pred_path.is_file(),
        "sha256": compute_file_sha256(pred_path) if pred_path.is_file() else "",
    }

    pred_conditions: Set[str] = set()
    pred_clips: Set[str] = set()
    pred_modes: Set[str] = set()
    pred_alphas: Set[float] = set()
    pred_dup_count = 0
    total_pred_records = 0

    if pred_path.is_file():
        for row in iter_jsonl(pred_path):
            total_pred_records += 1
            cid = str(row.get("clip_id", ""))
            mode = str(row.get("mode", ""))
            alpha = float(row.get("alpha", 0.0))
            key = f"{cid}|{mode}|{alpha}"
            if key in pred_conditions:
                pred_dup_count += 1
                report["duplicate_records"].append({"file": "prediction", "key": key})
            pred_conditions.add(key)
            pred_clips.add(cid)
            pred_modes.add(mode)
            pred_alphas.add(alpha)

    report["prediction_stats"] = {
        "total_rows": total_pred_records,
        "unique_conditions": len(pred_conditions),
        "unique_clips": len(pred_clips),
        "modes_found": sorted(list(pred_modes)),
        "alphas_found": sorted(list(pred_alphas)),
        "duplicate_count": pred_dup_count,
    }

    # 2. Context inspection
    ctx_path = config.context_jsonl
    report["files_checked"]["context"] = {
        "path": str(ctx_path),
        "exists": ctx_path.is_file(),
        "sha256": compute_file_sha256(ctx_path) if ctx_path.is_file() else "",
    }

    ctx_clips: Set[str] = set()
    ctx_dup_count = 0
    total_ctx_records = 0
    ctx_obstacles_summary: Dict[str, int] = {"total_obs": 0, "clips_with_obs": 0}
    ctx_road_summary: Dict[str, int] = {"total_boundaries": 0, "clips_with_boundaries": 0}

    if ctx_path.is_file():
        for row in iter_jsonl(ctx_path):
            total_ctx_records += 1
            cid = str(row.get("clip_id", ""))
            if cid in ctx_clips:
                ctx_dup_count += 1
                report["duplicate_records"].append({"file": "context", "clip_id": cid})
            ctx_clips.add(cid)

            sc = row.get("semantic_context", {})
            obs_list = sc.get("obstacle", {}).get("all_obstacles", [])
            if obs_list:
                ctx_obstacles_summary["total_obs"] += len(obs_list)
                ctx_obstacles_summary["clips_with_obs"] += 1

            road_list = sc.get("road_boundary", {}).get("all_geometry_records", [])
            if road_list:
                ctx_road_summary["total_boundaries"] += len(road_list)
                ctx_road_summary["clips_with_boundaries"] += 1

    report["context_stats"] = {
        "total_rows": total_ctx_records,
        "unique_clips": len(ctx_clips),
        "duplicate_count": ctx_dup_count,
        "obstacles_summary": ctx_obstacles_summary,
        "road_summary": ctx_road_summary,
    }

    # 3. Ground truth inspection
    gt_path = config.ground_truth_jsonl
    report["files_checked"]["ground_truth"] = {
        "path": str(gt_path),
        "exists": gt_path.is_file(),
        "sha256": compute_file_sha256(gt_path) if gt_path.is_file() else "",
    }

    gt_clips: Set[str] = set()
    gt_dup_count = 0
    total_gt_records = 0
    if gt_path.is_file():
        for row in iter_jsonl(gt_path):
            total_gt_records += 1
            cid = str(row.get("clip_id", ""))
            if cid in gt_clips:
                gt_dup_count += 1
                report["duplicate_records"].append({"file": "ground_truth", "clip_id": cid})
            gt_clips.add(cid)

    report["ground_truth_stats"] = {
        "total_rows": total_gt_records,
        "unique_clips": len(gt_clips),
        "duplicate_count": gt_dup_count,
    }

    ctx_map = read_jsonl_indexed(ctx_path, lambda r: str(r.get("clip_id", ""))) if ctx_path.is_file() else {}
    gt_map = read_jsonl_indexed(gt_path, lambda r: str(r.get("clip_id", ""))) if gt_path.is_file() else {}

    # 4. Trajectory contract & finite verification on predictions
    trajectory_validation_issues = []
    trajectory_validation_issue_count = 0
    checked_pred_count = 0
    if pred_path.is_file():
        for row in iter_jsonl(pred_path):
            checked_pred_count += 1
            cid = str(row.get("clip_id", ""))
            mode = str(row.get("mode", ""))
            raw_alpha = row.get("alpha", 0.0)
            try:
                alpha = float(raw_alpha)
            except (TypeError, ValueError):
                if len(trajectory_validation_issues) < 20:
                    trajectory_validation_issues.append({
                        "clip_id": cid, "mode": mode, "alpha": raw_alpha,
                        "issue": f"Non-numeric alpha value: {raw_alpha}"
                    })
                continue

            try:
                cid_ctx = ctx_map.get(cid, {})
                cid_gt = gt_map.get(cid, {})
                t0_us, _ = resolve_time_origin(row, cid_ctx, cid_gt, strict_mode=config.strict_mode)
                _normalized_window(row, _trajectory_for_row(row, alpha), t0_us, config, "prediction")
            except Exception as err:
                trajectory_validation_issue_count += 1
                if len(trajectory_validation_issues) < 20:
                    trajectory_validation_issues.append({
                        "clip_id": cid, "mode": mode, "alpha": alpha,
                        "issue": str(err)
                    })

    report["trajectory_contract_stats"] = {
        "checked_predictions": checked_pred_count,
        "total_issues": trajectory_validation_issue_count,
        "sample_issues": trajectory_validation_issues,
    }

    # 4b. Ground-truth timeline validation uses the exact scorer window and
    # normalizer, rather than treating record presence as readiness.
    pred_by_clip: Dict[str, Dict[str, Any]] = {}
    if pred_path.is_file():
        for row in iter_jsonl(pred_path):
            pred_by_clip.setdefault(str(row.get("clip_id", "")), row)
    gt_timeline_issues: List[Dict[str, Any]] = []
    gt_checked = 0
    gt_valid = 0
    for cid in sorted(pred_clips):
        gt_checked += 1
        pred_row = pred_by_clip.get(cid, {})
        gt_row = gt_map.get(cid)
        issue: Optional[Dict[str, Any]] = None
        try:
            if not gt_row:
                raise TimeContractError("missing ground-truth record")
            ctx_row = ctx_map.get(cid, {})
            t0_us, _ = resolve_time_origin(pred_row, ctx_row, gt_row, strict_mode=config.strict_mode)
            graw = next((gt_row.get(k) for k in ("ego_future_xyz", "expert_future", "future_waypoints") if gt_row.get(k) is not None), None)
            gt_timeline = _normalized_window(gt_row, graw, t0_us, config, "ground_truth")
            pred_timeline = _normalized_window(pred_row, _trajectory_for_row(pred_row, pred_row.get("alpha", 0.0)), t0_us, config, "prediction")
            if not np.array_equal(
                _future_and_full_timestamps(pred_timeline, t0_us)[1:],
                _future_and_full_timestamps(gt_timeline, t0_us)[1:],
            ):
                raise TimeContractError("ground-truth and prediction normalized grids differ")
            gt_valid += 1
        except Exception as exc:
            issue = {"clip_id": cid, "failure_type": type(exc).__name__, "failure_reason": str(exc)}
            gt_timeline_issues.append(issue)
    report["gt_timeline_stats"] = {
        "checked_clips": gt_checked,
        "valid_clips": gt_valid,
        "invalid_clips": len(gt_timeline_issues),
        "issue_count": len(gt_timeline_issues),
        "issues": gt_timeline_issues[:50],
    }

    # 4c. Observation readiness uses the same CF/TTC coverage primitives as
    # scoring, including per-frame confirmed-empty evidence.
    observation_rows: List[Dict[str, Any]] = []
    for cid in sorted(pred_clips):
        row: Dict[str, Any] = {"clip_id": cid}
        try:
            pred_row = pred_by_clip[cid]
            ctx_row = ctx_map.get(cid)
            if not ctx_row:
                raise TimeContractError("missing context record")
            t0_us, _ = resolve_time_origin(pred_row, ctx_row, gt_map.get(cid), strict_mode=config.strict_mode)
            pred_timeline = _normalized_window(pred_row, _trajectory_for_row(pred_row, pred_row.get("alpha", 0.0)), t0_us, config, "prediction")
            cf_ts = _future_and_full_timestamps(pred_timeline, t0_us)
            obstacle_block = ((ctx_row.get("semantic_context") or {}).get("obstacle") or {})
            obstacles = obstacle_block.get("all_obstacles")
            if obstacles is None:
                raise TimeContractError("missing all_obstacles")
            confirmed_scene, confirmed_ts = _observation_attestation(ctx_row)
            obs_by_time, obs_all_ts, invalid_count, missing_ts_count = index_and_filter_obstacles(obstacles, raise_on_corrupt=False)
            cf_empty = expand_confirmed_empty_timestamps(cf_ts, confirmed_ts, confirmed_scene)
            _, cf_observed, cf_confirmed, cf_missing, cf_ratio = evaluate_query_coverage(cf_ts, obs_by_time, obs_all_ts, cf_empty)
            ttc_ts = build_ttc_projection_timestamps(cf_ts, config.ttc_horizon_s)
            ttc_empty = expand_confirmed_empty_timestamps(ttc_ts, confirmed_ts, confirmed_scene)
            _, ttc_observed, ttc_confirmed, ttc_missing, ttc_ratio = evaluate_query_coverage(ttc_ts, obs_by_time, obs_all_ts, ttc_empty, half_step_us=100_000)
            row.update({
                "cf_required_frames": len(cf_ts), "cf_observed_frames": cf_observed,
                "cf_confirmed_empty_frames": cf_confirmed, "cf_missing_frames": cf_missing,
                "cf_coverage_ratio": cf_ratio,
                "ttc_required_observations": len(ttc_ts), "ttc_observed_observations": ttc_observed,
                "ttc_confirmed_empty_observations": ttc_confirmed, "ttc_missing_observations": ttc_missing,
                "ttc_coverage_ratio": ttc_ratio,
                "invalid_obstacle_count": invalid_count, "missing_timestamp_obstacle_count": missing_ts_count,
                "observation_contract_ready": bool(cf_missing == 0 and ttc_missing == 0 and invalid_count == 0),
                "failure_reason": "" if (cf_missing == 0 and ttc_missing == 0 and invalid_count == 0) else "observation coverage or obstacle integrity is incomplete",
            })
        except Exception as exc:
            row.update({"observation_contract_ready": False, "failure_type": type(exc).__name__, "failure_reason": str(exc)})
        observation_rows.append(row)
    report["observation_coverage_stats"] = {
        "checked_clips": len(observation_rows),
        "ready_clips": sum(1 for row in observation_rows if row.get("observation_contract_ready")),
        "invalid_clips": sum(1 for row in observation_rows if not row.get("observation_contract_ready")),
        "rows": observation_rows,
    }

    # 5. Grid completeness check (clip x mode x alpha)
    expected_modes = config.modes or ["cross_scene", "no_reasoning", "noisy", "opposite_action"]
    expected_alphas = config.alphas or [0.0, 0.5, 1.0, 2.0]
    expected_grid_per_clip = len(expected_modes) * len(expected_alphas)
    expected_total_grid = len(pred_clips) * expected_grid_per_clip
    missing_grid_conditions = []

    for cid in sorted(pred_clips):
        for m in expected_modes:
            for a in expected_alphas:
                k = f"{cid}|{m}|{a}"
                if k not in pred_conditions:
                    if len(missing_grid_conditions) < 50:
                        missing_grid_conditions.append({"clip_id": cid, "mode": m, "alpha": a})

    report["grid_stats"] = {
        "expected_modes": expected_modes,
        "expected_alphas": expected_alphas,
        "expected_total": expected_total_grid,
        "actual_total": len(pred_conditions),
        "missing_grid_count": len(missing_grid_conditions),
        "missing_sample": missing_grid_conditions[:10],
    }

    # 6. Parquet loader verification
    parquet_loader_status = "UNKNOWN"
    parquet_error = None
    if config.context_filtered_dir.is_dir():
        try:
            import pandas as pd
            # Find first available clip directory with parquet
            test_clip_dirs = list(config.context_filtered_dir.iterdir())[:5]
            found_test_file = None
            for cd in test_clip_dirs:
                clipgt = cd / "clipgt"
                if clipgt.is_dir():
                    for fn in ["lane.parquet", "intersection_area.parquet", "obstacle.parquet"]:
                        candidate = clipgt / fn
                        if candidate.is_file():
                            found_test_file = candidate
                            break
                if found_test_file:
                    break
            if found_test_file:
                test_df = pd.read_parquet(found_test_file)
                parquet_loader_status = f"OK (tested {found_test_file.name}, {len(test_df)} rows)"
            else:
                parquet_loader_status = "NO_PARQUET_FOUND"
        except Exception as exc:
            parquet_loader_status = f"FAILED: {exc}"
            parquet_error = str(exc)
    else:
        parquet_loader_status = "DIR_NOT_FOUND"

    report["parquet_loader"] = {
        "status": parquet_loader_status,
        "error": parquet_error,
    }

    # 7. Map & Drivable Polygon verification per clip
    map_stats = {
        "clips_checked": 0,
        "clips_with_map": 0,
        "clips_missing_map": 0,
        "map_polygon_issues": 0,
        "status_counts": {
            "OK": 0,
            "PARTIAL": 0,
            "FILE_NOT_FOUND": 0,
            "NO_DRIVABLE_POLYGON": 0,
            "PARQUET_READ_ERROR": 0,
            "UNSUPPORTED_SCHEMA": 0,
            "INVALID_GEOMETRY": 0,
        },
        "inventory": [],
        "sample_missing_map": [],
    }
    if config.context_filtered_dir.is_dir() and pred_clips:
        for cid in sorted(pred_clips):
            map_stats["clips_checked"] += 1
            info = inspect_clip_map_status(config.context_filtered_dir, cid)
            st = info.get("status", "UNKNOWN")
            map_stats["status_counts"][st] = map_stats["status_counts"].get(st, 0) + 1

            map_stats["inventory"].append({
                "clip_id": cid,
                "status": st,
                "lane_polygons": info.get("lane_polygon_count", 0),
                "intersection_polygons": info.get("intersection_polygon_count", 0),
                "total_polygons": info.get("total_polygons", 0),
                "valid_polygon_count": info.get("valid_polygon_count", 0),
                "invalid_polygon_count": info.get("invalid_polygon_count", 0),
                "skipped_row_count": info.get("skipped_row_count", 0),
                "usable_for_strict_scoring": info.get("usable_for_strict_scoring", False),
                "detail": info.get("detail", ""),
            })

            if st == "OK":
                map_stats["clips_with_map"] += 1
            else:
                map_stats["clips_missing_map"] += 1
                if st in ("INVALID_GEOMETRY", "PARTIAL"):
                    map_stats["map_polygon_issues"] += 1
                if len(map_stats["sample_missing_map"]) < 10:
                    map_stats["sample_missing_map"].append(f"{cid} ({st})")
    report["map_stats"] = map_stats

    # Round-5/6 readiness blockers. These diagnostics are observational only;
    # no clock offset or coordinate transform is applied here.
    report["time_alignment"] = audit_time_alignment(config)
    report["coordinate_alignment"] = audit_coordinate_alignment(config)

    # Check intersection & missing
    all_clips = pred_clips.union(ctx_clips).union(gt_clips)
    for cid in sorted(all_clips):
        missing = []
        if cid not in pred_clips:
            missing.append("prediction")
        if cid not in ctx_clips:
            missing.append("context")
        if cid not in gt_clips:
            missing.append("ground_truth")
        if missing:
            report["missing_records"].append({"clip_id": cid, "missing": missing})

    # Profile readiness
    env = audit_environment()
    has_inputs = bool(pred_path.is_file() and ctx_path.is_file() and gt_path.is_file())
    input_grid_ready = (
        env["proxy_possible"]
        and has_inputs
        and len(report["missing_records"]) == 0
        and len(report["duplicate_records"]) == 0
        and report["grid_stats"]["missing_grid_count"] == 0
        and trajectory_validation_issue_count == 0
        and parquet_loader_status.startswith("OK")
    )
    map_ready = bool(map_stats["clips_checked"] > 0 and map_stats["clips_missing_map"] == 0 and map_stats["map_polygon_issues"] == 0)
    gt_timeline_ready = bool(
        len(pred_clips) > 0
        and total_gt_records == len(gt_clips)
        and gt_clips.issuperset(pred_clips)
        and report["gt_timeline_stats"]["invalid_clips"] == 0
    )
    timeline_ready = bool(trajectory_validation_issue_count == 0 and report["gt_timeline_stats"]["invalid_clips"] == 0 and report["time_alignment"]["ready"])
    coordinate_ready = bool(report["coordinate_alignment"]["ready"])
    observation_ready = bool(
        report["observation_coverage_stats"]["checked_clips"] == len(pred_clips)
        and report["observation_coverage_stats"]["invalid_clips"] == 0
    )
    proxy_ready = bool(input_grid_ready and map_ready and timeline_ready and coordinate_ready and observation_ready)

    report["readiness"] = {
        "nurec_safety_proxy_v1": "READY" if proxy_ready else "NOT READY",
        "navsim_v2_stage1": "READY" if (env["official_stage1_possible"] and has_inputs) else "NOT READY",
        "navsim_v2_full": "READY" if (env["official_full_possible"] and has_inputs) else "NOT READY",
        "INPUT_GRID_READY": input_grid_ready,
        "MAP_READY": map_ready,
        "PREDICTION_TIMELINE_READY": bool(trajectory_validation_issue_count == 0),
        "GT_TIMELINE_READY": gt_timeline_ready,
        "TIME_ALIGNMENT_READY": timeline_ready,
        "COORDINATE_ALIGNMENT_READY": coordinate_ready,
        "OBSERVATION_COVERAGE_CONTRACT_READY": observation_ready,
        "blockers": [
            name for name, ok in (("INPUT_GRID_READY", input_grid_ready), ("MAP_READY", map_ready), ("PREDICTION_TIMELINE_READY", trajectory_validation_issue_count == 0), ("GT_TIMELINE_READY", gt_timeline_ready), ("TIME_ALIGNMENT_READY", timeline_ready), ("COORDINATE_ALIGNMENT_READY", coordinate_ready), ("OBSERVATION_COVERAGE_CONTRACT_READY", observation_ready)) if not ok
        ],
        "dataset_status": "DATASET_READY" if proxy_ready else "DATASET_NOT_READY",
    }

    return report


def run_and_save_audit(config: EvaluationConfig, output_dir: Path) -> Dict[str, Any]:
    """Runs Phase 0 audit and saves reports to the analysis directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = audit_environment()
    contracts = audit_data_contracts(config)

    # Save environment manifest
    env_file = output_dir / "environment_manifest.json"
    with env_file.open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    # Save data contract report JSON
    report_json_file = output_dir / "data_contract_report.json"
    with report_json_file.open("w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2)

    # Save missing & duplicates CSV
    missing_file = output_dir / "missing_records.csv"
    with missing_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "missing_components"])
        for item in contracts["missing_records"]:
            writer.writerow([item["clip_id"], ";".join(item["missing"])])

    dup_file = output_dir / "duplicate_records.csv"
    with dup_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "key_or_clip_id"])
        for item in contracts["duplicate_records"]:
            writer.writerow([item.get("file", ""), item.get("key", item.get("clip_id", ""))])

    # Save Map Inventory CSV & Markdown
    map_inv = contracts.get("map_stats", {}).get("inventory", [])
    map_csv_file = output_dir / "map_inventory_300.csv"
    with map_csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "status", "lane_polygons", "intersection_polygons", "total_polygons", "valid_polygons", "invalid_polygons", "skipped_rows", "usable_for_strict_scoring", "detail"])
        for item in map_inv:
            writer.writerow([
                item.get("clip_id", ""),
                item.get("status", ""),
                item.get("lane_polygons", 0),
                item.get("intersection_polygons", 0),
                item.get("total_polygons", 0),
                item.get("valid_polygon_count", 0),
                item.get("invalid_polygon_count", 0),
                item.get("skipped_row_count", 0),
                item.get("usable_for_strict_scoring", False),
                item.get("detail", ""),
            ])

    status_counts = contracts.get("map_stats", {}).get("status_counts", {})
    map_md_file = output_dir / "map_inventory_300.md"
    map_lines = [
        "# Drivable Map Polygons Inventory (300 Clips)",
        "",
        "This audit checks drivable map availability (`lane.parquet` and `intersection_area.parquet`) in each clip's `clipgt` directory.",
        "",
        "## Status Summary",
        "",
        "| Map Status Category | Clip Count | Evaluation Impact |",
        "| :--- | :---: | :--- |",
        f"| **`OK`** | **{status_counts.get('OK', 0)}** | Drivable area polygons available; DAC metric fully supported. |",
        f"| **`PARTIAL`** | **{status_counts.get('PARTIAL', 0)}** | Partially corrupted polygons present; strictly rejected in strict mode. |",
        f"| **`FILE_NOT_FOUND`** | **{status_counts.get('FILE_NOT_FOUND', 0)}** | Missing map files in `clipgt`; cannot compute true DAC. |",
        f"| **`NO_DRIVABLE_POLYGON`** | **{status_counts.get('NO_DRIVABLE_POLYGON', 0)}** | Map files present but contains 0 polygons (empty location arrays). |",
        f"| **`PARQUET_READ_ERROR`** | **{status_counts.get('PARQUET_READ_ERROR', 0)}** | Read failure / corruption during parquet deserialization. |",
        f"| **`UNSUPPORTED_SCHEMA`** | **{status_counts.get('UNSUPPORTED_SCHEMA', 0)}** | Parquet file missing expected schema or columns. |",
        f"| **`INVALID_GEOMETRY`** | **{status_counts.get('INVALID_GEOMETRY', 0)}** | Polygon vertices contain non-finite numbers (NaN/Inf). |",
        f"| **Total Checked** | **{contracts.get('map_stats', {}).get('clips_checked', 0)}** | |",
        "",
        "## Key Findings",
        f"- Exactly **{status_counts.get('OK', 0)} clips** contain fully valid drivable surface geometry.",
        f"- Exactly **{status_counts.get('PARTIAL', 0)} clips** contain partial/corrupted polygons requiring review.",
        f"- Exactly **{status_counts.get('FILE_NOT_FOUND', 0)} clips** have no map parquet files in `clipgt`.",
        f"- Exactly **{status_counts.get('NO_DRIVABLE_POLYGON', 0)} clip** has map parquet files present but with 0 vertices (e.g. empty location arrays).",
        "",
        "## Detailed Inventory (Non-OK Clips)",
        "",
        "| Clip ID | Status | Detail |",
        "| :--- | :--- | :--- |",
    ]
    for item in map_inv:
        if item.get("status") != "OK":
            map_lines.append(f"| `{item.get('clip_id')}` | `{item.get('status')}` | {item.get('detail')} |")

    with map_md_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(map_lines) + "\n")

    # Round-5 alignment reports are intentionally separate from the map
    # inventory so a reviewer can distinguish code readiness from dataset
    # readiness.
    time_report = contracts.get("time_alignment", {"rows": []})
    time_csv_file = output_dir / "time_alignment_report.csv"
    time_fields = [
        "clip_id", "prediction_t0_us", "gt_t0_us", "context_t0_us",
        "prediction_first_time_us", "prediction_last_time_us", "gt_first_time_us", "gt_last_time_us",
        "obstacle_min_us", "obstacle_max_us", "egomotion_min_us", "egomotion_max_us",
        "candidate_pred_to_obstacle_offset_us", "candidate_pred_to_egomotion_offset_us",
        "time_alignment_status", "time_alignment_source", "time_alignment_confidence", "failure_reason",
    ]
    with time_csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=time_fields)
        writer.writeheader()
        for row in time_report.get("rows", []):
            writer.writerow({key: row.get(key) for key in time_fields})
    time_md_file = output_dir / "time_alignment_report.md"
    time_lines = ["# Time Alignment Report", "", f"Status counts: `{time_report.get('status_counts', {})}`", "", "| Clip | Status | Reason |", "|---|---|---|"]
    time_lines.extend(f"| `{row.get('clip_id')}` | `{row.get('time_alignment_status')}` | {row.get('failure_reason', '')} |" for row in time_report.get("rows", []))
    time_md_file.write_text("\n".join(time_lines) + "\n", encoding="utf-8")

    coord_report = contracts.get("coordinate_alignment", {"rows": []})
    coord_fields = ["clip_id", "prediction_frame", "gt_frame", "obstacle_frame", "map_frame", "prediction_anchor", "gt_anchor", "obstacle_anchor", "map_anchor", "transform_required", "transform_source", "coordinate_alignment_verified", "failure_reason"]
    coord_csv_file = output_dir / "coordinate_alignment_report.csv"
    with coord_csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=coord_fields)
        writer.writeheader()
        for row in coord_report.get("rows", []):
            writer.writerow({key: row.get(key) for key in coord_fields})
    coord_md_file = output_dir / "coordinate_alignment_report.md"
    coord_lines = ["# Coordinate Alignment Report", "", f"Ready: `{coord_report.get('ready', False)}`", "", "| Clip | Verified | Frame/anchor reason |", "|---|---|---|"]
    coord_lines.extend(f"| `{row.get('clip_id')}` | `{row.get('coordinate_alignment_verified')}` | {row.get('failure_reason', '')} |" for row in coord_report.get("rows", []))
    coord_md_file.write_text("\n".join(coord_lines) + "\n", encoding="utf-8")

    observation_report = contracts.get("observation_coverage_stats", {})
    observation_csv_file = output_dir / "observation_coverage_report.csv"
    observation_fields = [
        "clip_id", "cf_required_frames", "cf_observed_frames", "cf_confirmed_empty_frames",
        "cf_missing_frames", "cf_coverage_ratio", "ttc_required_observations",
        "ttc_observed_observations", "ttc_confirmed_empty_observations", "ttc_missing_observations",
        "ttc_coverage_ratio", "invalid_obstacle_count", "missing_timestamp_obstacle_count",
        "observation_contract_ready", "failure_reason",
    ]
    with observation_csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=observation_fields)
        writer.writeheader()
        for row in observation_report.get("rows", []):
            writer.writerow({key: row.get(key) for key in observation_fields})

    # Save Markdown report
    md_file = output_dir / "data_contract_report.md"
    lines = [
        "# Phase 0 — Data Contract & Environment Audit Report",
        "",
        f"**Date / Time:** Generated automatically",
        f"**Metric Profile Selected:** `{config.metric_profile}`",
        "",
        "## 1. Readiness Conclusion",
        "",
        f"| Evaluation Profile | Status | Reason |",
        f"| :--- | :---: | :--- |",
        f"| **`navsim_v2_full`** | **{contracts['readiness']['navsim_v2_full']}** | NAVSIM / nuPlan package installed: {env['navsim_available']} |",
        f"| **`navsim_v2_stage1`** | **{contracts['readiness']['navsim_v2_stage1']}** | NAVSIM / nuPlan package installed: {env['navsim_available']} |",
        f"| **`nurec_safety_proxy_v1`** | **{contracts['readiness']['nurec_safety_proxy_v1']}** | Pure Python + NumPy ({env.get('numpy_version')}) + Pandas ({env.get('pandas_version')}) |",
        "",
        "## 2. Input Datasets Integrity",
        "",
        f"* **Predictions (`ar1_output`):** {contracts['prediction_stats']['total_rows']} rows, {contracts['prediction_stats']['unique_clips']} unique clips, {contracts['prediction_stats']['unique_conditions']} conditions.",
        f"  * Modes: `{contracts['prediction_stats']['modes_found']}`",
        f"  * Alphas: `{contracts['prediction_stats']['alphas_found']}`",
        f"* **Context (`full_context`):** {contracts['context_stats']['total_rows']} rows, {contracts['context_stats']['unique_clips']} unique clips.",
        f"  * Total obstacle instances across clips: {contracts['context_stats']['obstacles_summary']['total_obs']}",
        f"  * Total road boundary polygons across clips: {contracts['context_stats']['road_summary']['total_boundaries']}",
        f"* **Ground Truth (`future_gt`):** {contracts['ground_truth_stats']['total_rows']} rows, {contracts['ground_truth_stats']['unique_clips']} unique clips.",
        f"* **Missing / Incomplete Clips:** {len(contracts['missing_records'])} missing records identified.",
        f"* **Duplicate Records:** {len(contracts['duplicate_records'])} duplicates found.",
        "",
        "## 3. Data Contracts & Parquet Engine Validation",
        "",
        f"* **Trajectory Horizon & Finite Check:** {contracts['trajectory_contract_stats']['checked_predictions']} trajectories inspected. Issues found: {contracts['trajectory_contract_stats']['total_issues']}.",
        f"* **Clip x Mode x Alpha Grid Completeness:** {contracts['grid_stats']['actual_total']} / {contracts['grid_stats']['expected_total']} expected conditions present ({contracts['grid_stats']['missing_grid_count']} missing).",
        f"* **Parquet Engine & Map Loader:** `{contracts['parquet_loader']['status']}`",
        f"* **Per-Clip Drivable Map Polygons:** {contracts.get('map_stats', {}).get('clips_with_map', 0)} / {contracts.get('map_stats', {}).get('clips_checked', 0)} clips have valid map polygons.",
        "",
        "### Drivable Map Polygons Breakdown",
        "",
        f"| Category | Count | Description |",
        f"| :--- | :---: | :--- |",
        f"| `OK` | {status_counts.get('OK', 0)} | Clips with valid drivable polygons. |",
        f"| `FILE_NOT_FOUND` | {status_counts.get('FILE_NOT_FOUND', 0)} | Clips with neither lane nor intersection parquet in clipgt. |",
        f"| `NO_DRIVABLE_POLYGON` | {status_counts.get('NO_DRIVABLE_POLYGON', 0)} | Parquet present but 0 valid polygon vertices. |",
        f"| `PARQUET_READ_ERROR` | {status_counts.get('PARQUET_READ_ERROR', 0)} | Deserialization errors. |",
        f"| `UNSUPPORTED_SCHEMA` | {status_counts.get('UNSUPPORTED_SCHEMA', 0)} | Missing required columns. |",
        f"| `INVALID_GEOMETRY` | {status_counts.get('INVALID_GEOMETRY', 0)} | Non-finite vertex coordinates. |",
        "",
        f"See `map_inventory_300.csv` and `map_inventory_300.md` for per-clip details.",
        "",
    ]
    with md_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "environment": env,
        "contracts": contracts,
        "files_written": [
            str(env_file),
            str(report_json_file),
            str(md_file),
            str(missing_file),
            str(dup_file),
            str(map_csv_file),
            str(map_md_file),
            str(time_csv_file),
            str(time_md_file),
            str(coord_csv_file),
            str(coord_md_file),
            str(observation_csv_file),
        ],
    }
