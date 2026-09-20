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

from tools.epdms.audit import audit_data_contracts, audit_environment
from tools.epdms.condition_identity import parse_condition_identity, record_key_from_prediction
from tools.epdms.config import EvaluationConfig
from tools.epdms.io_jsonl import AtomicJsonlWriter, compute_file_sha256, iter_jsonl, read_jsonl_indexed
from tools.epdms.map_loader import inspect_clip_map_status, load_lane_polygons_for_clip
from tools.epdms.map_transform import load_transformed_map_for_clip
from tools.epdms.nurec_inputs import (
    decorate_context,
    decorate_prediction_gt,
    decorate_inputs,
    load_coordinate_contract,
    load_dac_readiness,
    load_observation_readiness,
    load_time_mapping,
)
from tools.epdms.reporting import export_table_to_csv
from tools.epdms.run_identity import (
    METRIC_IMPLEMENTATION_VERSION,
    assert_fresh_score_dir_safe,
    compute_map_directory_content_sha256,
    generate_run_id,
    load_latest_checkpoint_states,
    RunIdentityError,
    verify_resume_safety_before_recovery,
    write_manifest_atomic,
)
from tools.epdms.schemas import EvaluationScoreRecord
from tools.epdms.score_record import evaluate_single_condition


def _record_key_from_prediction(row: Dict[str, Any]) -> str:
    return record_key_from_prediction(row)


def derive_scoring_status(n_expected: int, n_valid: int, n_invalid: int, n_unprocessed: int) -> str:
    """Separate score completeness from whether the runner loop finished."""
    if n_valid == 0:
        return "BLOCKED"
    if n_valid == n_expected and n_invalid == 0 and n_unprocessed == 0:
        return "COMPLETE"
    return "PARTIAL"


def state_counts(expected_keys: set[str], states: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    valid = {key for key, row in states.items() if key in expected_keys and row.get("valid") is True}
    invalid = {key for key, row in states.items() if key in expected_keys and row.get("valid") is not True}
    return {
        "n_expected": len(expected_keys),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_unprocessed": len(expected_keys - valid - invalid),
    }



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
    parser.add_argument("--retry-invalid", action="store_true", help="Re-evaluate keys found in the previous errors JSONL.")
    parser.add_argument("--score-dir", type=Path, default=None, help="Output directory for scores.")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips to evaluate.")
    parser.add_argument("--clip-ids-file", type=Path, default=None, help="Optional newline-delimited clip IDs to evaluate.")
    parser.add_argument("--overwrite-new-run", action="store_true", help="Allow a fresh --no-resume run to replace canonical output in a non-empty score directory.")
    parser.add_argument("--nurec-coordinate-contract", type=Path, default=None, help="Accepted frozen NuRec coordinate contract JSON.")
    parser.add_argument("--nurec-observation-readiness", type=Path, default=None, help="Full-300 observation readiness CSV.")
    parser.add_argument("--nurec-time-mapping", type=Path, default=None, help="Verified per-clip NuRec/PAI time mapping JSONL.")
    parser.add_argument("--nurec-dac-readiness", type=Path, default=None, help="Production DAC readiness CSV; absent means DAC is fail-closed.")
    parser.add_argument("--nurec-map-transform-contract", type=Path, default=None, help="Verified NuRec map→EGO_AT_T0 transform contract JSON.")
    args = parser.parse_args()

    print(f"[*] Loading config from: {args.config}")
    config = EvaluationConfig.from_file(args.config)

    profile = args.profile or config.metric_profile
    horizon_s = args.horizon or config.horizon_s
    resume = args.resume if args.resume is not None else config.resume

    # Enforce profile contract: official navsim profiles must raise NotImplementedError
    if profile in ("navsim_v2_full", "navsim_v2_stage1"):
        raise NotImplementedError("OFFICIAL_PROFILE_NOT_IMPLEMENTED")

    nurec_adapter = None
    if profile == "nurec_safety_proxy_v1" and all((args.nurec_coordinate_contract, args.nurec_observation_readiness, args.nurec_time_mapping)):
        nurec_adapter = {
            "contract": load_coordinate_contract(args.nurec_coordinate_contract),
            "readiness": load_observation_readiness(args.nurec_observation_readiness),
            "time_mapping": load_time_mapping(args.nurec_time_mapping),
            "dac": load_dac_readiness(args.nurec_dac_readiness) if args.nurec_dac_readiness else {},
            "map_transform_contract": json.loads(args.nurec_map_transform_contract.read_text(encoding="utf-8")) if args.nurec_map_transform_contract else {},
        }
        print("[*] NuRec frozen input adapter: enabled")
        print(f"[*] NuRec DAC readiness manifest: {'loaded' if args.nurec_dac_readiness else 'absent (fail-closed)'}")

    # Resolve the exact clip scope before computing run identity. The scope is
    # the sorted set of clip IDs, not merely the requested count.
    print(f"[*] Reading predictions from: {config.prediction_jsonl}")
    all_pred_rows = list(iter_jsonl(config.prediction_jsonl))
    all_clip_ids = sorted({str(r.get("clip_id", "")) for r in all_pred_rows})
    requested_clip_ids = None
    if args.clip_ids_file is not None:
        requested_clip_ids = {line.strip() for line in args.clip_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        target_clips = [cid for cid in all_clip_ids if cid in requested_clip_ids]
    else:
        target_clips = all_clip_ids[:args.max_clips] if args.max_clips is not None else all_clip_ids
    all_pred_rows = [r for r in all_pred_rows if str(r.get("clip_id", "")) in set(target_clips)]
    clip_scope = sorted(set(target_clips))
    if args.max_clips is not None and args.clip_ids_file is None:
        print(f"[*] Filtered to first {args.max_clips} clips ({len(all_pred_rows)} conditions)")
    if args.clip_ids_file is not None:
        print(f"[*] Filtered to requested clip IDs: {len(target_clips)} clips ({len(all_pred_rows)} conditions)")

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
    if args.nurec_map_transform_contract is not None and args.nurec_map_transform_contract.is_file():
        source_hashes["nurec_map_transform_contract"] = compute_file_sha256(args.nurec_map_transform_contract)

    current_effective_fingerprint = config.compute_effective_fingerprint(
        source_hashes=source_hashes,
        runtime_overrides={"horizon_s": horizon_s, "metric_profile": profile},
        clip_scope=clip_scope,
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
    error_output_jsonl = error_jsonl

    # Resolve execution identity independently from the deterministic
    # effective fingerprint. A fresh execution always gets a new run_id;
    # resume reuses the manifest's run_id.
    previous_manifest: Optional[Dict[str, Any]] = None
    if resume:
        previous_manifest = verify_resume_safety_before_recovery(
            score_dir=score_dir,
            expected_fingerprint=current_effective_fingerprint,
            strict_identity=True,
        )
        run_id = str(previous_manifest.get("run_id")) if previous_manifest and previous_manifest.get("run_id") else generate_run_id()
        if previous_manifest:
            verify_resume_safety_before_recovery(
                score_dir=score_dir,
                expected_fingerprint=current_effective_fingerprint,
                expected_run_id=run_id,
                strict_identity=True,
            )
    else:
        assert_fresh_score_dir_safe(score_dir, overwrite_new_run=args.overwrite_new_run)
        run_id = generate_run_id()

    if args.retry_invalid and resume and error_jsonl.is_file():
        # Keep retry attempts append-safe and auditable instead of creating
        # duplicate latest-state rows in the canonical errors file.
        error_output_jsonl = score_dir / f"epdms_errors_300_attempt_{run_id}_{time.time_ns()}.jsonl"
    manifest_json = score_dir / "run_manifest.json"

    # 1. Validate identity on resume BEFORE any recovery or file modification
    if resume:
        # 2. Recover .tmp and repair truncated lines AFTER verifying safety
        AtomicJsonlWriter.prepare_file_for_resume(score_jsonl)
        AtomicJsonlWriter.prepare_file_for_resume(error_jsonl)
        for attempt_path in sorted(score_dir.glob("epdms_errors_300_attempt_*.jsonl")):
            AtomicJsonlWriter.prepare_file_for_resume(attempt_path)

    # 3. Open writers
    score_writer = AtomicJsonlWriter(score_jsonl, append_if_exists=resume)
    error_writer = AtomicJsonlWriter(error_output_jsonl, append_if_exists=resume and error_output_jsonl == error_jsonl)

    # 4. Read the latest state per key across canonical and retry attempt
    # files. Historical errors remain available for audit but do not affect
    # current-state counts.
    states = load_latest_checkpoint_states(
        score_dir,
        expected_run_id=run_id,
        expected_fingerprint=current_effective_fingerprint,
        strict_identity=True,
    ) if resume else {}
    valid_keys = {key for key, row in states.items() if row.get("valid") is True}
    invalid_keys = {key for key, row in states.items() if row.get("valid") is not True}
    previous_invalid_types: Dict[str, str] = {
        key: str(row.get("failure_type") or "unknown")
        for key, row in states.items() if row.get("valid") is not True
    }
    all_score_dicts: List[Dict[str, Any]] = [row for row in states.values() if row.get("valid") is True]
    print(f"[*] Resuming: valid={len(valid_keys)}, invalid={len(invalid_keys)}")

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
            k = parse_condition_identity(r).record_key
            rule_group_map[k] = r.get("group")

    total_conditions = len(all_pred_rows)
    print(f"[*] Total conditions to process: {total_conditions}")

    cached_polygons: Dict[str, List[Any]] = {}
    cached_decorated_context: Dict[str, Dict[str, Any]] = {}

    start_time = time.time()
    processed_this_run = 0
    skipped_count = 0
    skipped_valid_count = 0
    skipped_invalid_count = 0
    retried_invalid_count = 0
    identities = [parse_condition_identity(row, row_index=idx) for idx, row in enumerate(all_pred_rows)]
    expected_keys = {identity.record_key for identity in identities}

    try:
        phase0_report = audit_data_contracts(config)
        dataset_status = phase0_report.get("readiness", {}).get("dataset_status", "DATASET_NOT_READY")
    except Exception as audit_error:
        dataset_status = "DATASET_NOT_READY"
        print(f"[!] Phase 0 dataset-status audit unavailable: {audit_error}")

    # Write pre-run manifest with RUNNING status before loop
    manifest_data = {
        "status": "RUNNING",
        "profile": profile,
        "horizon_s": horizon_s,
        "frequency_hz": config.frequency_hz,
        "total_conditions": total_conditions,
        "n_expected": len(expected_keys),
        "n_valid": len(valid_keys & expected_keys),
        "n_invalid": len(invalid_keys & expected_keys),
        "n_unprocessed": len(expected_keys - valid_keys - invalid_keys),
        "n_skipped_valid": 0,
        "n_skipped_invalid": 0,
        "n_retried_invalid": 0,
        "execution_status": "RUNNING",
        "scoring_status": derive_scoring_status(len(expected_keys), len(valid_keys & expected_keys), len(invalid_keys & expected_keys), len(expected_keys - valid_keys - invalid_keys)),
        "dataset_status": dataset_status,
        "clip_scope": clip_scope,
        "processed_this_run": 0,
        "skipped_resumed": len(valid_keys) + (0 if args.retry_invalid else len(invalid_keys)),
        "retry_invalid": bool(args.retry_invalid),
        "total_completed": len(all_score_dicts),
        "total_runtime_s": 0.0,
        "config_sha256": config.sha256,
        "effective_fingerprint": current_effective_fingerprint,
        "run_id": run_id,
        "implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_manifest_atomic(manifest_json, manifest_data)

    execution_error: Optional[BaseException] = None
    try:
        for idx, (pred_row, identity) in enumerate(zip(all_pred_rows, identities), start=1):
            eval_record: Optional[EvaluationScoreRecord] = None
            record_key: Optional[str] = None
            try:
                alpha = identity.alpha if identity.alpha is not None else 0.0
                clip_id = identity.clip_id
                mode = identity.mode
                record_key = identity.record_key

                if record_key in valid_keys:
                    skipped_count += 1
                    skipped_valid_count += 1
                    continue
                if record_key in invalid_keys and not args.retry_invalid:
                    skipped_count += 1
                    skipped_invalid_count += 1
                    continue
                if args.retry_invalid and record_key in invalid_keys:
                    retried_invalid_count += 1

                if not identity.valid:
                    eval_record = EvaluationScoreRecord(
                        record_key=record_key,
                        clip_id=clip_id,
                        mode=mode,
                        alpha=0.0,
                        valid=False,
                        failure_stage="condition_identity",
                        failure_type=identity.failure_type or "InvalidAlphaError",
                        failure_reason=identity.failure_reason or "Invalid condition identity",
                    )
                    raise StopIteration

                ctx_row = context_map.get(clip_id)
                gt_row = gt_map.get(clip_id)
                rg = rule_group_map.get(record_key)

                dac_ready = False
                time_record = None
                if nurec_adapter is not None and ctx_row is not None and gt_row is not None:
                    readiness = nurec_adapter["readiness"].get(clip_id)
                    time_record = nurec_adapter["time_mapping"].get(clip_id)
                    if readiness is not None and time_record is not None and time_record.get("verified"):
                        if clip_id not in cached_decorated_context:
                            cached_decorated_context[clip_id] = decorate_context(
                                ctx_row, nurec_adapter["contract"], time_record, readiness,
                            )
                        pred_row = decorate_prediction_gt(pred_row, nurec_adapter["contract"], "prediction")
                        gt_row = decorate_prediction_gt(gt_row, nurec_adapter["contract"], "ground_truth")
                        ctx_row = cached_decorated_context[clip_id]
                        dac_row = nurec_adapter["dac"].get(clip_id, {})
                        dac_ready = str(dac_row.get("dac_ready", "")).lower() == "true"

                if clip_id not in cached_polygons:
                    if (
                        nurec_adapter is not None
                        and nurec_adapter.get("map_transform_contract", {}).get("status") == "VERIFIED"
                        and time_record is not None
                    ):
                        cached_polygons[clip_id] = load_transformed_map_for_clip(
                            config.context_filtered_dir,
                            clip_id,
                            int(time_record["nurec_t0_us"]),
                            strict=True,
                        )
                    else:
                        cached_polygons[clip_id] = inspect_clip_map_status(config.context_filtered_dir, clip_id)
                map_res = cached_polygons[clip_id]
                polygons = (
                    map_res.get("transformed_lane_polygons", [])
                    + map_res.get("transformed_intersection_polygons", [])
                ) if map_res.get("transform_status") == "VERIFIED" else (
                    map_res.get("lane_polygons", []) + map_res.get("intersection_polygons", [])
                )
                clip_map_status = map_res.get("status")
                if map_res.get("transform_status") == "VERIFIED":
                    clip_map_status = "OK"
                if nurec_adapter is not None and not dac_ready:
                    polygons = []
                    clip_map_status = "MAP_TRANSFORM_MISSING" if map_res.get("status") in {"OK", "PARTIAL", "NO_DRIVABLE_POLYGON"} else "MAP_MISSING"

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
            except StopIteration:
                pass
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
            previous_state = states.get(record_key or "", {})
            if args.retry_invalid and record_key in previous_invalid_types:
                rec_dict["attempt_number"] = int(previous_state.get("attempt_number", 1)) + 1
                rec_dict["previous_failure_type"] = previous_invalid_types[record_key]
            else:
                rec_dict["attempt_number"] = 1
            rec_dict["run_id"] = run_id
            rec_dict["effective_fingerprint"] = current_effective_fingerprint
            rec_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if eval_record.valid:
                score_writer.write(rec_dict)
                all_score_dicts.append(rec_dict)
            else:
                error_writer.write(rec_dict)

            processed_this_run += 1
            if eval_record.valid:
                valid_keys.add(record_key)
                invalid_keys.discard(record_key)
            else:
                invalid_keys.add(record_key)
                valid_keys.discard(record_key)
            states[record_key] = rec_dict

            if processed_this_run % 50 == 0:
                score_writer.flush()
                error_writer.flush()
                elapsed = time.time() - start_time
                rate = processed_this_run / max(1e-3, elapsed)
                print(f"[{idx}/{total_conditions}] Evaluated {processed_this_run} conditions ({rate:.1f} cond/s) - current: {record_key}")

    except BaseException as exc:
        execution_error = exc
        print(f"[!] Evaluation loop failed: {type(exc).__name__}: {exc}")
    finally:
        score_writer.close()
        error_writer.close()

    # Export complete CSV
    if all_score_dicts:
        print(f"[*] Writing complete CSV table to: {score_csv}")
        export_table_to_csv(all_score_dicts, score_csv)

    # Save final manifest from latest unique state, separating execution from
    # score completeness and dataset readiness.
    total_time = time.time() - start_time
    counts = state_counts(expected_keys, states)
    scoring_status = derive_scoring_status(
        counts["n_expected"], counts["n_valid"], counts["n_invalid"], counts["n_unprocessed"]
    )
    execution_status = "FAILED" if execution_error is not None else "COMPLETED"
    pilot_gating_only = args.max_clips is not None and counts["n_valid"] == 0
    if pilot_gating_only:
        pilot_status = "REAL_SCORING_PILOT_BLOCKED"
    elif args.max_clips is not None and scoring_status == "COMPLETE":
        pilot_status = "REAL_SCORING_PILOT_COMPLETED"
    elif args.max_clips is not None:
        pilot_status = "REAL_SCORING_PILOT_PARTIAL"
    else:
        pilot_status = "SCORING_COMPLETED" if scoring_status == "COMPLETE" else f"SCORING_{scoring_status}"
    manifest_data.update({
        "status": execution_status,
        "execution_status": execution_status,
        "scoring_status": scoring_status,
        "pilot_status": pilot_status,
        **counts,
        "n_skipped_valid": skipped_valid_count,
        "n_skipped_invalid": skipped_invalid_count,
        "n_retried_invalid": retried_invalid_count,
        "processed_this_run": processed_this_run,
        "skipped_resumed": skipped_count,
        "total_completed": counts["n_valid"],
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
    print(f"    Execution Status: {execution_status}")
    print(f"    Scoring Status:   {scoring_status}")
    print(f"    Pilot Status:     {pilot_status}")

    if execution_error is not None:
        raise execution_error


if __name__ == "__main__":
    main()
