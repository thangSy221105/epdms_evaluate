"""Phase 0 Audit: Environment dependencies and data contract validation."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from .config import EvaluationConfig
from .io_jsonl import compute_file_sha256, iter_jsonl


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

    # 4. Trajectory contract & finite verification on predictions
    trajectory_validation_issues = []
    checked_pred_count = 0
    if pred_path.is_file():
        for row in iter_jsonl(pred_path):
            checked_pred_count += 1
            cid = str(row.get("clip_id", ""))
            mode = str(row.get("mode", ""))
            if alpha > 0.0:
                traj = row.get("guided_waypoints")
                if traj is None and "trajectories" in row:
                    traj = row["trajectories"].get("guided")
            else:
                traj = row.get("clean_waypoints")
                if traj is None:
                    traj = row.get("guided_waypoints")
                if traj is None and "trajectories" in row:
                    traj = row["trajectories"].get("clean") or row["trajectories"].get("guided")

            if not isinstance(traj, list) or len(traj) < 40:
                if len(trajectory_validation_issues) < 20:
                    trajectory_validation_issues.append({
                        "clip_id": cid, "mode": mode, "alpha": alpha,
                        "issue": f"Insufficient waypoints: {len(traj) if isinstance(traj, list) else 'None'} < 40"
                    })
            else:
                for pt_idx, pt in enumerate(traj[:40]):
                    if not isinstance(pt, dict):
                        if len(trajectory_validation_issues) < 20:
                            trajectory_validation_issues.append({
                                "clip_id": cid, "mode": mode, "alpha": alpha,
                                "issue": f"Malformed waypoint at index {pt_idx}: {pt}"
                            })
                        break
                    x = pt.get("x_m", pt.get("x"))
                    y = pt.get("y_m", pt.get("y"))
                    if x is None or y is None:
                        if len(trajectory_validation_issues) < 20:
                            trajectory_validation_issues.append({
                                "clip_id": cid, "mode": mode, "alpha": alpha,
                                "issue": f"Missing coordinates at index {pt_idx}: {pt}"
                            })
                        break
                    try:
                        xf, yf = float(x), float(y)
                        import math
                        if math.isnan(xf) or math.isinf(xf) or math.isnan(yf) or math.isinf(yf):
                            if len(trajectory_validation_issues) < 20:
                                trajectory_validation_issues.append({
                                    "clip_id": cid, "mode": mode, "alpha": alpha,
                                    "issue": f"Non-finite waypoint at index {pt_idx}: ({xf}, {yf})"
                                })
                            break
                    except (TypeError, ValueError):
                        if len(trajectory_validation_issues) < 20:
                            trajectory_validation_issues.append({
                                "clip_id": cid, "mode": mode, "alpha": alpha,
                                "issue": f"Non-numeric coordinates at index {pt_idx}: ({x}, {y})"
                            })
                        break

    report["trajectory_contract_stats"] = {
        "checked_predictions": checked_pred_count,
        "total_issues": len(trajectory_validation_issues),
        "sample_issues": trajectory_validation_issues,
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

    # Check intersection & missing
    all_clips = pred_clips.union(ctx_clips).union(gt_clips)
    for cid in all_clips:
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
    proxy_ready = (
        env["proxy_possible"]
        and has_inputs
        and len(trajectory_validation_issues) == 0
        and ("FAILED" not in parquet_loader_status)
    )

    report["readiness"] = {
        "nurec_safety_proxy_v1": "READY" if proxy_ready else "NOT READY",
        "navsim_v2_stage1": "READY" if (env["official_stage1_possible"] and has_inputs) else "NOT READY",
        "navsim_v2_full": "READY" if (env["official_full_possible"] and has_inputs) else "NOT READY",
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
        ],
    }
