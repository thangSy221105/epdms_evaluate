#!/usr/bin/env python3
"""CLI runner to evaluate conditions using EPDMS / NuRec Safety Proxy v1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.epdms.audit import audit_environment
from tools.epdms.config import EvaluationConfig
from tools.epdms.io_jsonl import AtomicJsonlWriter, compute_file_sha256, iter_jsonl, read_jsonl_indexed
from tools.epdms.reporting import export_table_to_csv
from tools.epdms.schemas import EvaluationScoreRecord
from tools.epdms.score_record import evaluate_single_condition


def load_lane_polygons_for_clip(filtered_dir: Path, clip_id: str) -> List[Any]:
    """Loads lane and road boundaries for a clip if available."""
    clip_dir = filtered_dir / clip_id / "clipgt"
    polygons = []
    if not clip_dir.is_dir():
        return polygons

    # Check lane.parquet
    lane_pq = clip_dir / "lane.parquet"
    if lane_pq.is_file():
        try:
            import numpy as np
            import pandas as pd
            df = pd.read_parquet(lane_pq)
            for _, row in df.iterrows():
                lane_data = row.get("lane", {})
                if isinstance(lane_data, dict):
                    left = lane_data.get("left_rail")
                    right = lane_data.get("right_rail")
                    if left is not None and right is not None and len(left) > 1 and len(right) > 1:
                        pts_left = [[p["x"], p["y"]] for p in left]
                        pts_right = [[p["x"], p["y"]] for p in reversed(right)]
                        poly = np.array(pts_left + pts_right, dtype=float)
                        polygons.append(poly)
        except Exception:
            pass

    return polygons


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory conditions using EPDMS / NuRec Safety Proxy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "epdms_300.json",
        help="Path to evaluation config JSON.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["nurec_safety_proxy_v1", "navsim_v2_stage1", "navsim_v2_full"],
        default=None,
        help="Evaluation profile override.",
    )
    parser.add_argument("--horizon", type=float, default=None, help="Evaluation horizon in seconds (default: 4.0).")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from existing score file.")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips to evaluate.")
    args = parser.parse_args()

    print(f"[*] Loading config from: {args.config}")
    config = EvaluationConfig.from_file(args.config)

    profile = args.profile or config.metric_profile
    horizon_s = args.horizon or config.horizon_s
    resume = args.resume if args.resume is not None else config.resume

    # Enforce profile contract
    env = audit_environment()
    if profile in ("navsim_v2_full", "navsim_v2_stage1"):
        if not (env["navsim_available"] and env["nuplan_available"]):
            print(f"[!] Error: Profile '{profile}' requires NAVSIM and nuPlan packages.")
            print("[!] Per project rules, we do not silently fallback to proxy when official profile is requested.")
            sys.exit(1)

    print(f"[*] Profile:           {profile}")
    print(f"[*] Horizon:           {horizon_s}s (@ {config.frequency_hz} Hz)")
    print(f"[*] Strict Mode:       {config.strict_mode}")
    print(f"[*] Resume Mode:       {resume}")

    # Output paths
    score_dir = config.score_dir
    score_dir.mkdir(parents=True, exist_ok=True)
    score_jsonl = score_dir / "epdms_scores_300.jsonl"
    score_csv = score_dir / "epdms_scores_300.csv"
    error_jsonl = score_dir / "epdms_errors_300.jsonl"
    manifest_json = score_dir / "run_manifest.json"

    # Read existing records if resume
    completed_keys = set()
    if resume and score_jsonl.is_file():
        for r in iter_jsonl(score_jsonl):
            k = r.get("record_key")
            if k:
                completed_keys.add(k)
        print(f"[*] Resuming: found {len(completed_keys)} already evaluated conditions in {score_jsonl.name}")

    # Load context & ground truth indexed by clip_id
    print(f"[*] Loading context index from: {config.context_jsonl}")
    context_map = read_jsonl_indexed(config.context_jsonl, lambda r: str(r.get("clip_id", "")))

    print(f"[*] Loading ground truth index from: {config.ground_truth_jsonl}")
    gt_map = read_jsonl_indexed(config.ground_truth_jsonl, lambda r: str(r.get("clip_id", "")))

    # Optional rule group mapping
    rule_group_file = config.analysis_dir.parent / "sensitivity" / "sensitivity_rule_groups_fixed_alpha05_4800.jsonl"
    rule_group_map = {}
    if rule_group_file.is_file():
        print(f"[*] Loading rule groups mapping from: {rule_group_file.name}")
        for r in iter_jsonl(rule_group_file):
            k = f"{r.get('clip_id')}|{r.get('mode')}|{float(r.get('alpha', 0.0)):.3f}".rstrip("0").rstrip(".") if float(r.get("alpha", 0.0)) != 0 else f"{r.get('clip_id')}|{r.get('mode')}|0"
            rule_group_map[k] = r.get("group")

    # Load predictions
    print(f"[*] Reading predictions from: {config.prediction_jsonl}")
    all_pred_rows = list(iter_jsonl(config.prediction_jsonl))

    if args.max_clips is not None:
        target_clips = sorted(list(dict.fromkeys(r.get("clip_id") for r in all_pred_rows)))[:args.max_clips]
        all_pred_rows = [r for r in all_pred_rows if r.get("clip_id") in target_clips]
        print(f"[*] Filtered to first {args.max_clips} clips ({len(all_pred_rows)} conditions)")

    total_conditions = len(all_pred_rows)
    print(f"[*] Total conditions to process: {total_conditions}")

    all_score_dicts: List[Dict[str, Any]] = []
    # If resuming, load existing records into all_score_dicts for CSV export
    if resume and score_jsonl.is_file():
        for r in iter_jsonl(score_jsonl):
            all_score_dicts.append(r)

    score_writer = AtomicJsonlWriter(score_jsonl, append_if_exists=resume)
    error_writer = AtomicJsonlWriter(error_jsonl, append_if_exists=resume)

    cached_polygons: Dict[str, List[Any]] = {}

    start_time = time.time()
    processed_this_run = 0
    skipped_count = 0

    try:
        for idx, pred_row in enumerate(all_pred_rows, start=1):
            clip_id = str(pred_row.get("clip_id", ""))
            mode = str(pred_row.get("mode", ""))
            alpha = float(pred_row.get("alpha", 0.0))
            record_key = f"{clip_id}|{mode}|{alpha:.3f}".rstrip("0").rstrip(".") if alpha != 0 else f"{clip_id}|{mode}|0"

            if record_key in completed_keys:
                skipped_count += 1
                continue

            ctx_row = context_map.get(clip_id)
            gt_row = gt_map.get(clip_id)
            rg = rule_group_map.get(record_key)

            if clip_id not in cached_polygons:
                cached_polygons[clip_id] = load_lane_polygons_for_clip(config.context_filtered_dir, clip_id)
            polygons = cached_polygons[clip_id]

            # Evaluate condition
            eval_record: EvaluationScoreRecord = evaluate_single_condition(
                pred_row=pred_row,
                context_row=ctx_row,
                gt_row=gt_row,
                vehicle=config.vehicle,
                horizon_s=horizon_s,
                frequency_hz=config.frequency_hz,
                rule_group=rg,
                lane_polygons=polygons,
            )

            eval_record.config_sha256 = config.sha256
            rec_dict = eval_record.to_dict()

            if eval_record.valid:
                score_writer.write(rec_dict)
                all_score_dicts.append(rec_dict)
            else:
                error_writer.write(rec_dict)

            processed_this_run += 1
            completed_keys.add(record_key)

            if processed_this_run % 50 == 0:
                score_writer.flush()
                error_writer.flush()
                elapsed = time.time() - start_time
                rate = processed_this_run / max(1e-3, elapsed)
                print(f"[{idx}/{total_conditions}] Evaluated {processed_this_run} conditions ({rate:.1f} cond/s) - current: {record_key}")

    finally:
        score_writer.close()
        error_writer.close()

    # Export complete CSV
    if all_score_dicts:
        print(f"[*] Writing complete CSV table to: {score_csv}")
        export_table_to_csv(all_score_dicts, score_csv)

    # Save run manifest
    total_time = time.time() - start_time
    manifest_data = {
        "profile": profile,
        "horizon_s": horizon_s,
        "frequency_hz": config.frequency_hz,
        "total_conditions": total_conditions,
        "processed_this_run": processed_this_run,
        "skipped_resumed": skipped_count,
        "total_completed": len(all_score_dicts),
        "total_runtime_s": round(total_time, 2),
        "config_sha256": config.sha256,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[+] Finished Evaluation:")
    print(f"    Total Completed: {len(all_score_dicts)}")
    print(f"    Processed now:   {processed_this_run}")
    print(f"    Skipped resumed: {skipped_count}")
    print(f"    Elapsed time:    {total_time:.1f}s")
    print(f"    Scores JSONL:    {score_jsonl}")
    print(f"    Scores CSV:      {score_csv}")
    print(f"    Run Manifest:    {manifest_json}")


if __name__ == "__main__":
    main()
