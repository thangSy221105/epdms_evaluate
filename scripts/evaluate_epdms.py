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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from tools.epdms.audit import audit_environment
from tools.epdms.config import EvaluationConfig
from tools.epdms.io_jsonl import AtomicJsonlWriter, compute_file_sha256, iter_jsonl, read_jsonl_indexed
from tools.epdms.map_loader import inspect_clip_map_status, load_lane_polygons_for_clip
from tools.epdms.reporting import export_table_to_csv
from tools.epdms.run_identity import (
    compute_map_directory_content_sha256,
    verify_resume_safety_before_recovery,
    write_manifest_atomic,
)
from tools.epdms.schemas import EvaluationScoreRecord
from tools.epdms.score_record import evaluate_single_condition



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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="Resume from existing score file.")
    parser.add_argument("--score-dir", type=Path, default=None, help="Output directory for scores.")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips to evaluate.")
    args = parser.parse_args()

    print(f"[*] Loading config from: {args.config}")
    config = EvaluationConfig.from_file(args.config)

    profile = args.profile or config.metric_profile
    horizon_s = args.horizon or config.horizon_s
    resume = args.resume if args.resume is not None else config.resume

    # Enforce profile contract: official navsim profiles must raise NotImplementedError
    if profile in ("navsim_v2_full", "navsim_v2_stage1"):
        raise NotImplementedError("OFFICIAL_PROFILE_NOT_IMPLEMENTED")

    # Compute effective fingerprint including sources and map
    source_hashes = {}
    if config.prediction_jsonl.is_file():
        source_hashes["prediction_jsonl"] = compute_file_sha256(config.prediction_jsonl)
    if config.context_jsonl.is_file():
        source_hashes["context_jsonl"] = compute_file_sha256(config.context_jsonl)
    if config.ground_truth_jsonl.is_file():
        source_hashes["ground_truth_jsonl"] = compute_file_sha256(config.ground_truth_jsonl)

    if config.context_filtered_dir.is_dir():
        map_sha = compute_map_directory_content_sha256(config.context_filtered_dir)
        if map_sha:
            source_hashes["context_filtered_map"] = map_sha

    current_effective_fingerprint = config.compute_effective_fingerprint(
        source_hashes=source_hashes,
        runtime_overrides={"horizon_s": horizon_s, "metric_profile": profile},
    )

    print(f"[*] Profile:               {profile}")
    print(f"[*] Horizon:               {horizon_s}s (@ {config.frequency_hz} Hz)")
    print(f"[*] Strict Mode:           {config.strict_mode}")
    print(f"[*] Resume Mode:           {resume}")
    print(f"[*] Effective Fingerprint: {current_effective_fingerprint[:16]}...")

    # Output paths
    score_dir = args.score_dir or config.score_dir
    score_dir.mkdir(parents=True, exist_ok=True)
    score_jsonl = score_dir / "epdms_scores_300.jsonl"
    score_csv = score_dir / "epdms_scores_300.csv"
    error_jsonl = score_dir / "epdms_errors_300.jsonl"
    manifest_json = score_dir / "run_manifest.json"

    # 1. Validate identity on resume BEFORE any recovery or file modification
    if resume:
        verify_resume_safety_before_recovery(
            score_dir=score_dir,
            expected_fingerprint=current_effective_fingerprint,
            artifact_names=[
                score_jsonl.name,
                f"{score_jsonl.name}.tmp",
                error_jsonl.name,
                f"{error_jsonl.name}.tmp",
            ],
        )

        # 2. Recover .tmp and repair truncated lines AFTER verifying safety
        AtomicJsonlWriter.prepare_file_for_resume(score_jsonl)
        AtomicJsonlWriter.prepare_file_for_resume(error_jsonl)

    # 3. Open writers
    score_writer = AtomicJsonlWriter(score_jsonl, append_if_exists=resume)
    error_writer = AtomicJsonlWriter(error_jsonl, append_if_exists=resume)

    completed_keys = set()
    all_score_dicts: List[Dict[str, Any]] = []

    # 4. Read cleanly recovered records from score_jsonl
    if resume and score_jsonl.is_file():
        for r in iter_jsonl(score_jsonl):
            k = r.get("record_key")
            if k:
                if k in completed_keys:
                    raise ValueError(f"Corrupted score file: duplicate record key found on resume: {k}")
                completed_keys.add(k)
                all_score_dicts.append(r)
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

    cached_polygons: Dict[str, List[Any]] = {}

    start_time = time.time()
    processed_this_run = 0
    skipped_count = 0

    # Write pre-run manifest with RUNNING status before loop
    manifest_data = {
        "status": "RUNNING",
        "profile": profile,
        "horizon_s": horizon_s,
        "frequency_hz": config.frequency_hz,
        "total_conditions": total_conditions,
        "processed_this_run": 0,
        "skipped_resumed": len(completed_keys),
        "total_completed": len(all_score_dicts),
        "total_runtime_s": 0.0,
        "config_sha256": config.sha256,
        "effective_fingerprint": current_effective_fingerprint,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_manifest_atomic(manifest_json, manifest_data)

    try:
        for idx, pred_row in enumerate(all_pred_rows, start=1):
            eval_record: Optional[EvaluationScoreRecord] = None
            record_key: Optional[str] = None
            try:
                raw_alpha = pred_row.get("alpha")
                alpha = float(raw_alpha) if raw_alpha is not None else 0.0
                clip_id = str(pred_row.get("clip_id", "unknown"))
                mode = str(pred_row.get("mode", "unknown"))
                record_key = f"{clip_id}|{mode}|{alpha:.3f}".rstrip("0").rstrip(".") if alpha != 0 else f"{clip_id}|{mode}|0"

                if record_key in completed_keys:
                    skipped_count += 1
                    continue

                ctx_row = context_map.get(clip_id)
                gt_row = gt_map.get(clip_id)
                rg = rule_group_map.get(record_key)

                if clip_id not in cached_polygons:
                    cached_polygons[clip_id] = inspect_clip_map_status(config.context_filtered_dir, clip_id)
                map_res = cached_polygons[clip_id]
                polygons = map_res.get("lane_polygons", []) + map_res.get("intersection_polygons", [])
                clip_map_status = map_res.get("status")

                # Evaluate condition with full parameter propagation
                eval_record = evaluate_single_condition(
                    pred_row=pred_row,
                    context_row=ctx_row,
                    gt_row=gt_row,
                    vehicle=config.vehicle,
                    horizon_s=horizon_s,
                    frequency_hz=config.frequency_hz,
                    rule_group=rg,
                    lane_polygons=polygons,
                    map_status=clip_map_status,
                    touch_is_collision=config.touch_is_collision,
                    ttc_horizon_s=config.ttc_horizon_s,
                    progress_stationary_threshold_m=config.progress_stationary_threshold_m,
                    strict_mode=config.strict_mode,
                    metric_profile=profile,
                )
            except Exception as exc:
                cid = str(pred_row.get("clip_id", "unknown")) if isinstance(pred_row, dict) else "unknown"
                m = str(pred_row.get("mode", "unknown")) if isinstance(pred_row, dict) else "unknown"
                record_key = record_key or f"{cid}|{m}|err"
                eval_record = EvaluationScoreRecord(
                    record_key=record_key,
                    clip_id=cid,
                    mode=m,
                    alpha=0.0,
                    valid=False,
                    failure_stage="cli_condition_loop",
                    failure_type=type(exc).__name__,
                    failure_reason=str(exc),
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

    # Save final run manifest with COMPLETED status
    total_time = time.time() - start_time
    manifest_data.update({
        "status": "COMPLETED",
        "processed_this_run": processed_this_run,
        "skipped_resumed": skipped_count,
        "total_completed": len(all_score_dicts),
        "total_runtime_s": round(total_time, 2),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_manifest_atomic(manifest_json, manifest_data)

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
