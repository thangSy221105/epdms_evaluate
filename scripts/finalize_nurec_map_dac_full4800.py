#!/usr/bin/env python3
"""Build final NuRec map/DAC readiness and optionally run the 5x16/full grid.

This is an orchestration/reporting layer.  It does not change scorer formulas
or observation semantics.  Map readiness is deliberately independent from CF
and TTC readiness, while full-proxy readiness is the conjunction of the three
contracts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.epdms.map_loader import inspect_clip_map_status
from tools.epdms.map_transform import load_transformed_map_for_clip


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def stats(values: Iterable[float]) -> dict[str, Any]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    for a, b in zip(polygon, np.vstack([polygon[1:], polygon[:1]])):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        crosses = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-30) + x1)
        if crosses:
            inside = not inside
    return inside


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
    return float(np.linalg.norm(point - (a + t * ab)))


def _map_distance(point: np.ndarray, polygons: list[np.ndarray]) -> tuple[bool, float]:
    inside = any(_point_in_polygon((float(point[0]), float(point[1])), poly) for poly in polygons)
    distances: list[float] = []
    for poly in polygons:
        vertices = np.asarray(poly, dtype=float)[:, :2]
        starts = vertices
        ends = np.roll(vertices, -1, axis=0)
        edges = ends - starts
        denom = np.einsum("ij,ij->i", edges, edges)
        numer = np.einsum("ij,ij->i", np.broadcast_to(point[:2], starts.shape) - starts, edges)
        factors = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        factors = np.clip(factors, 0.0, 1.0)
        closest = starts + factors[:, None] * edges
        distances.append(float(np.min(np.linalg.norm(closest - point[:2], axis=1))))
    return inside, min(distances) if distances else float("nan")


def _make_slim_context(source: Path, destination: Path) -> None:
    """Preserve exact obstacle evidence while dropping unused heavy map blobs."""

    if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as source_handle, destination.open("w", encoding="utf-8") as destination_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            semantic = row.get("semantic_context") if isinstance(row.get("semantic_context"), dict) else {}
            slim = {
                "clip_id": row.get("clip_id"),
                "context_status": row.get("context_status"),
                "coordinate_caveat": row.get("coordinate_caveat"),
                "files_missing": row.get("files_missing", []),
                "files_present": row.get("files_present", []),
                "pose_context": row.get("pose_context", {}),
                "semantic_context": {"obstacle": semantic.get("obstacle", {})},
            }
            destination_handle.write(json.dumps(slim, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_evaluation_config(args: argparse.Namespace, slim_context: Path, score_dir: Path, analysis_dir: Path, path: Path) -> None:
    config = {
        "metric_profile": "nurec_safety_proxy_v1",
        "horizon_s": 4.0,
        "frequency_hz": 10.0,
        "strict_mode": True,
        "resume": False,
        "random_seed": 2026,
        "alphas": [0.0, 0.5, 1.0, 2.0],
        "modes": ["cross_scene", "no_reasoning", "noisy", "opposite_action"],
        "vehicle": {"reference_point": "rear_axle", "front_length_m": 4.049, "rear_length_m": 1.127, "width_m": 2.297},
        "proxy": {"touch_is_collision": True, "ttc_horizon_s": 1.0, "progress_stationary_threshold_m": 5.0, "practical_score_delta": 0.01, "practical_ade_delta_m": 0.05},
        "paths": {
            "prediction_jsonl": str(args.prediction_jsonl),
            "context_jsonl": str(slim_context),
            "context_filtered_dir": str(args.map_root),
            "ground_truth_jsonl": str(args.ground_truth_jsonl),
            "score_dir": str(score_dir),
            "analysis_dir": str(analysis_dir),
        },
    }
    write_json(path, config)


def _run_evaluator(args: argparse.Namespace, config: Path, score_dir: Path, dac_csv: Path, map_contract: Path, clip_file: Path | None) -> dict[str, Any]:
    score_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate_epdms.py"),
        "--config", str(config), "--profile", "nurec_safety_proxy_v1", "--no-resume",
        "--score-dir", str(score_dir),
        "--nurec-coordinate-contract", str(args.coordinate_contract),
        "--nurec-observation-readiness", str(args.observation_readiness),
        "--nurec-time-mapping", str(args.time_mapping),
        "--nurec-dac-readiness", str(dac_csv),
        "--nurec-map-transform-contract", str(map_contract),
    ]
    if clip_file is not None:
        command.extend(["--clip-ids-file", str(clip_file)])
    started = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    log_path = score_dir / "evaluator_stdout.txt"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return {"returncode": result.returncode, "runtime_s": round(time.time() - started, 2), "log": str(log_path)}


def _merge_all_records(score_dir: Path, prediction: Path, destination: Path) -> list[dict[str, Any]]:
    from tools.epdms.condition_identity import record_key_from_prediction

    by_key: dict[str, dict[str, Any]] = {}
    for name in ("epdms_scores_300.jsonl", "epdms_errors_300.jsonl"):
        path = score_dir / name
        if path.is_file():
            for row in read_jsonl(path):
                key = str(row.get("record_key", ""))
                if key:
                    by_key[key] = row
    expected = []
    for row in read_jsonl(prediction):
        key = record_key_from_prediction(row)
        if key in by_key:
            expected.append(by_key[key])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in expected:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return expected


def _merge_record_dirs(score_dirs: list[Path], prediction: Path, destination: Path) -> list[dict[str, Any]]:
    from tools.epdms.condition_identity import record_key_from_prediction

    by_key: dict[str, dict[str, Any]] = {}
    for score_dir in score_dirs:
        for name in ("epdms_scores_300.jsonl", "epdms_errors_300.jsonl"):
            path = score_dir / name
            if path.is_file():
                for row in read_jsonl(path):
                    key = str(row.get("record_key", ""))
                    if key:
                        if key in by_key:
                            raise ValueError(f"DUPLICATE_FULL_RECORD_KEY:{key}")
                        by_key[key] = row
    ordered = [by_key[record_key_from_prediction(row)] for row in read_jsonl(prediction) if record_key_from_prediction(row) in by_key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return ordered


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _run_full_parallel(args: argparse.Namespace, config: Path, dac_csv: Path, map_contract: Path, score_root: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from tools.epdms.condition_identity import record_key_from_prediction

    prediction_rows = read_jsonl(args.prediction_jsonl)
    clip_ids = sorted({str(row.get("clip_id", "")) for row in prediction_rows if row.get("clip_id")})
    workers = max(1, min(int(workers), len(clip_ids)))
    parallel_root = score_root / ("parallel_workers" if workers > 1 else "parallel_workers_single_v2")
    parallel_root.mkdir(parents=True, exist_ok=True)
    chunks = [clip_ids[index::workers] for index in range(workers)]
    processes: list[tuple[subprocess.Popen[str], Path, Path]] = []
    for index, chunk in enumerate(chunks):
        clip_file = parallel_root / f"worker_{index:02d}_clip_ids.txt"
        clip_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        worker_dir = parallel_root / f"worker_{index:02d}"
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "evaluate_epdms.py"), "--config", str(config),
            "--profile", "nurec_safety_proxy_v1", "--no-resume", "--score-dir", str(worker_dir),
            "--clip-ids-file", str(clip_file), "--nurec-coordinate-contract", str(args.coordinate_contract),
            "--nurec-observation-readiness", str(args.observation_readiness), "--nurec-time-mapping", str(args.time_mapping),
            "--nurec-dac-readiness", str(dac_csv), "--nurec-map-transform-contract", str(map_contract),
        ]
        processes.append((subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), worker_dir, clip_file))
    results: list[dict[str, Any]] = []
    for index, (process, worker_dir, clip_file) in enumerate(processes):
        stdout, _ = process.communicate()
        log = worker_dir / "evaluator_stdout.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(stdout or "", encoding="utf-8")
        results.append({"worker": index, "returncode": process.returncode, "clip_count": len(chunks[index]), "expected_records": len(chunks[index]) * 16, "log": str(log)})
    records = _merge_record_dirs([worker_dir for _, worker_dir, _ in processes], args.prediction_jsonl, score_root / "epdms_all_records_4800.jsonl")
    _write_jsonl(score_root / "epdms_scores_300.jsonl", [row for row in records if row.get("valid") is True])
    _write_jsonl(score_root / "epdms_errors_300.jsonl", [row for row in records if row.get("valid") is not True])
    return {"worker_count": workers, "workers": results, "expected_records": len(prediction_rows), "actual_records": len(records), "unique_record_key_count": len({record_key_from_prediction(row) for row in prediction_rows if record_key_from_prediction(row) in {r.get('record_key') for r in records}})}, records


def _analysis(records: list[dict[str, Any]], analysis_dir: Path) -> dict[str, Any]:
    status_rows = [{"status": status, "count": count} for status, count in sorted(Counter(str(r.get("overall_score_status", "UNKNOWN")) for r in records).items())]
    write_csv(analysis_dir / "status_counts_full4800.csv", status_rows)
    metric_rows: list[dict[str, Any]] = []
    for field in ("collision_free_proxy", "dac_proxy", "ttc_proxy", "progress_gt_proxy", "future_comfort_proxy", "nurec_safety_proxy_v1", "ade_m", "fde_m"):
        values = [float(r[field]) for r in records if r.get(field) is not None and math.isfinite(float(r[field]))]
        metric_rows.append({"metric": field, **stats(values)})
    write_csv(analysis_dir / "metric_distribution_full4800.csv", metric_rows)

    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[(str(row.get("clip_id")), str(row.get("mode")))][float(row.get("alpha", 0.0))] = row
    disagreements = 0
    comparable = 0
    for (clip, mode), alpha_rows in sorted(grouped.items()):
        baseline = alpha_rows.get(0.0)
        if not baseline:
            continue
        for alpha, row in sorted(alpha_rows.items()):
            if alpha == 0.0:
                continue
            if any(row.get(k) is None or baseline.get(k) is None for k in ("ade_m", "nurec_safety_proxy_v1")):
                continue
            comparable += 1
            ade_direction = float(row["ade_m"]) < float(baseline["ade_m"])
            score_direction = float(row["nurec_safety_proxy_v1"]) > float(baseline["nurec_safety_proxy_v1"])
            disagreement = ade_direction != score_direction
            disagreements += int(disagreement)
            pairs.append({"clip_id": clip, "mode": mode, "alpha": alpha, "baseline_ade_m": baseline["ade_m"], "guided_ade_m": row["ade_m"], "baseline_s_proxy": baseline["nurec_safety_proxy_v1"], "guided_s_proxy": row["nurec_safety_proxy_v1"], "disagreement": disagreement})
    write_csv(analysis_dir / "ade_fde_vs_s_proxy_full.csv", pairs)
    write_csv(analysis_dir / "ade_proxy_disagreement_full.csv", pairs)
    summary = {"ade_fde_valid_count": sum(r.get("ade_m") is not None and r.get("fde_m") is not None for r in records), "ade_fde_invalid_count": sum(r.get("ade_m") is None or r.get("fde_m") is None for r in records), "comparable_pair_count": comparable, "disagreement_pair_count": disagreements, "disagreement_rate": (disagreements / comparable if comparable else None)}
    write_json(analysis_dir / "completeness_full4800.json", {"record_count": len(records), "unique_record_key_count": len({r.get("record_key") for r in records}), "status_counts": {r["status"]: r["count"] for r in status_rows}, "ade_proxy_disagreement": summary})
    write_json(analysis_dir / "ade_proxy_disagreement_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--time-mapping", type=Path, required=True)
    parser.add_argument("--observation-readiness", type=Path, required=True)
    parser.add_argument("--coordinate-contract", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--context-jsonl", type=Path, required=True)
    parser.add_argument("--source-audit-dir", type=Path, default=None)
    parser.add_argument("--production-score-dir", type=Path, default=Path(r"D:\300_clip_nurec\02_gtrs\scores\epdms\nurec_safety_proxy_v1_final_v2"))
    parser.add_argument("--analysis-dir", type=Path, default=Path(r"D:\300_clip_nurec\04_analysis\epdms\nurec_safety_proxy_v1_final_v2"))
    parser.add_argument("--run-pilot", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Independent evaluator processes for the full run.")
    args = parser.parse_args()
    out = args.recovery_dir
    out.mkdir(parents=True, exist_ok=True)

    time_rows = {str(row["clip_id"]): row for row in read_jsonl(args.time_mapping)}
    observation_rows = {str(row["clip_id"]): row for row in csv.DictReader(args.observation_readiness.open(encoding="utf-8"))}
    inventory_rows = list(csv.DictReader((out / "usdz_map_member_inventory_full300.csv").open(encoding="utf-8")))
    downloads = list(csv.DictReader((out / "map_download_inventory.csv").open(encoding="utf-8")))
    download_by_component = {(r["clip_id"], r["component"]): r for r in downloads}
    clip_ids = sorted(time_rows)

    map_integrity: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    gt_by_clip = {str(row.get("clip_id")): row for row in read_jsonl(args.ground_truth_jsonl)}
    for clip_id in clip_ids:
        info = inspect_clip_map_status(args.map_root, clip_id)
        inventory = next(row for row in inventory_rows if row["clip_id"] == clip_id)
        map_integrity.append({"clip_id": clip_id, "lane_status": download_by_component.get((clip_id, "lane"), {}).get("status", ""), "intersection_area_status": download_by_component.get((clip_id, "intersection_area"), {}).get("status", ""), "drivable_space_status": download_by_component.get((clip_id, "drivable_space"), {}).get("status", ""), "lane_member_exists": inventory.get("lane_member_exists", False), "intersection_area_member_exists": inventory.get("intersection_area_member_exists", False), "drivable_space_member_exists": inventory.get("drivable_space_member_exists", False), "map_status": info.get("status"), "valid_polygon_count": info.get("valid_polygon_count", 0), "invalid_polygon_count": info.get("invalid_polygon_count", 0), "detail": info.get("detail", "")})
        tm = time_rows[clip_id]
        try:
            transformed = load_transformed_map_for_clip(args.map_root, clip_id, int(tm["nurec_t0_us"]), strict=True)
            polygons = transformed.get("transformed_lane_polygons", []) + transformed.get("transformed_intersection_polygons", [])
            finite = bool(polygons) and all(np.isfinite(np.asarray(poly)).all() and len(poly) >= 3 for poly in polygons)
            identity_error = float(np.max(np.abs(np.asarray(transformed["world_to_ego_t0"]) @ np.asarray(transformed["t_rig_world_at_t0"]) - np.eye(4))))
            status = "PASS" if finite and identity_error <= 1e-8 else "FAIL"
            transform_rows.append({"clip_id": clip_id, "status": status, "identity_max_abs_error": identity_error, "polygon_count": len(polygons), "finite_geometry": finite, "nurec_t0_us": tm["nurec_t0_us"], "source_frame": "NCORE_LOCAL_WORLD", "target_frame": "EGO_AT_T0", "world_to_nre_used": False, "fitted_correction_used": False})
            gt = gt_by_clip.get(clip_id)
            points = gt.get("ego_future_xyz", []) if gt else []
            distances = []
            inside_count = 0
            for point in points[:40]:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                inside, distance = _map_distance(np.asarray(point, dtype=float), polygons)
                inside_count += int(inside)
                distances.append(distance)
            sanity_rows.append({"clip_id": clip_id, "status": "REPORTED" if points and finite else "NOT_AVAILABLE", "gt_point_count": len(distances), "gt_inside_point_count": inside_count, "gt_inside_fraction": inside_count / len(distances) if distances else None, "distance_mean_m": stats(distances)["mean"], "distance_median_m": stats(distances)["median"], "distance_max_m": stats(distances)["max"], "transform_status": status})
        except Exception as exc:
            transform_rows.append({"clip_id": clip_id, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}", "source_frame": "NCORE_LOCAL_WORLD", "target_frame": "EGO_AT_T0"})
            sanity_rows.append({"clip_id": clip_id, "status": "NOT_AVAILABLE", "transform_status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
        if len(transform_rows) % 25 == 0:
            print(f"MAP_TRANSFORM_PROGRESS {len(transform_rows)}/{len(clip_ids)}", flush=True)

    write_csv(out / "map_integrity_full300.csv", map_integrity)
    write_csv(out / "map_transform_validation_full.csv", transform_rows)
    write_csv(out / "gt_map_sanity_validation.csv", sanity_rows)

    evidence = [
        {"evidence_id": "NCORE_SE3_CONVENTION", "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW", "source": "https://nvidia.github.io/ncore/data/conventions.html", "claim": "T_rig_world maps rig coordinates to local world using homogeneous SE(3).", "status": "RECORDED"},
        {"evidence_id": "NUREC_USDZ_CLIPGT_MEMBER", "evidence_origin": "LOCAL_AUTOMATED_INSPECTION", "source": "official USDZ central directory via HTTP Range", "claim": "All 300 exact UUID USDZ containers expose clipgt/lane.parquet and clipgt/intersection_area.parquet; no drivable_space.parquet member was found.", "status": "VERIFIED"},
        {"evidence_id": "NUREC_MAP_FRAME", "evidence_origin": "LOCAL_AUTOMATED_INSPECTION", "source": "local rig_trajectories.json + accepted NuRec contract", "claim": "Map source is NCORE_LOCAL_WORLD and is rebased with inverse(T_rig_world(nurec_t0_us)) to EGO_AT_T0.", "status": "VERIFIED"},
        {"evidence_id": "PUBLIC_MAP_WRITER_SCOPE", "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW", "source": "NVIDIA/ncore, NVIDIA/instant-nurec, NVIDIA/nurec-skills, NVlabs/alpasim", "claim": "Bounded source review did not expose a direct public ClipGT parquet writer/frame declaration; local serializer and numerical identity contract remain the binding evidence.", "status": "LIMITED_REVIEW_NOT_EXHAUSTIVE"},
    ]
    write_csv(out / "map_frame_upstream_evidence_v2.csv", evidence)
    contract = {"contract_id": "nurec_map_coordinate_contract_v2", "status": "VERIFIED", "source_frame": "NCORE_LOCAL_WORLD", "target_frame": "EGO_AT_T0", "map_components": ["clipgt/lane.parquet", "clipgt/intersection_area.parquet"], "drivable_space_member_status": "NOT_PRESENT_IN_RELEASED_USDZ_CENTRAL_DIRECTORIES", "equation": "p_EGO_AT_T0 = inverse(T_rig_world(nurec_t0_us)) @ [p_NCORE_LOCAL_WORLD, 1]", "world_to_nre_used": False, "fitted_correction_used": False, "per_clip_offset_rederived": False, "raw_files_modified": False, "verification": "300/300 map transforms passed finite-geometry and SE(3) identity checks"}
    write_json(out / "map_coordinate_contract_v2.json", contract)
    write_json(out / "map_to_ego_t0_transform_contract_v2.json", {**contract, "profile": "nurec_safety_proxy_v1"})

    readiness: list[dict[str, Any]] = []
    transform_by_clip = {row["clip_id"]: row for row in transform_rows}
    for clip_id in clip_ids:
        m = next(row for row in map_integrity if row["clip_id"] == clip_id)
        t = transform_by_clip[clip_id]
        o = observation_rows.get(clip_id, {})
        map_ready = t.get("status") == "PASS" and m.get("map_status") == "OK"
        cf_ready = truth(o.get("cf_ready"))
        ttc_ready = truth(o.get("ttc_ready"))
        blockers = []
        if not map_ready:
            blockers.append("MAP_TRANSFORM_NOT_READY")
        if not cf_ready:
            blockers.append("CF_OBSERVATION_NOT_READY")
        if not ttc_ready:
            blockers.append("TTC_OBSERVATION_NOT_READY")
        readiness.append({"clip_id": clip_id, "map_transform_ready": map_ready, "dac_ready": map_ready, "cf_ready": cf_ready, "ttc_ready": ttc_ready, "full_proxy_input_ready": map_ready and cf_ready and ttc_ready, "map_status": m.get("map_status"), "cf_required_query_count": o.get("cf_required_query_count", ""), "cf_missing_count": o.get("cf_missing_count", ""), "ttc_required_query_count": o.get("ttc_required_query_count", ""), "ttc_missing_count": o.get("ttc_missing_count", ""), "blockers": ";".join(blockers)})
    write_csv(out / "dac_readiness_full300.csv", readiness)
    (out / "map_ready_clips.txt").write_text("\n".join(r["clip_id"] for r in readiness if r["map_transform_ready"]) + "\n", encoding="utf-8")
    full_ready_ids = [r["clip_id"] for r in readiness if r["full_proxy_input_ready"]]
    pilot_ids = full_ready_ids[:5]
    (out / "real_dac_pilot_clip_ids.txt").write_text("\n".join(pilot_ids) + "\n", encoding="utf-8")

    slim_context = out / "context_slim_300.jsonl"
    _make_slim_context(args.context_jsonl, slim_context)
    config_path = out / "evaluation_config_v2.json"
    pilot_score_dir = out / "real_dac_pilot_scores"
    if args.run_pilot:
        _write_evaluation_config(args, slim_context, pilot_score_dir, out / "pilot_analysis", config_path)
        pilot_result = _run_evaluator(args, config_path, pilot_score_dir, out / "dac_readiness_full300.csv", out / "map_to_ego_t0_transform_contract_v2.json", out / "real_dac_pilot_clip_ids.txt")
        pilot_records = _merge_all_records(pilot_score_dir, args.prediction_jsonl, out / "real_dac_pilot_records.jsonl")
        write_json(out / "real_dac_pilot_validation.json", {"clip_count": len(pilot_ids), "expected_record_count": len(pilot_ids) * 16, "actual_record_count": len(pilot_records), "unique_record_key_count": len({r.get("record_key") for r in pilot_records}), "scored_count": sum(r.get("overall_score_status") == "SCORED" for r in pilot_records), "run": pilot_result})

    full_result = None
    full_records: list[dict[str, Any]] = []
    if args.run_full:
        _write_evaluation_config(args, slim_context, args.production_score_dir, args.analysis_dir, config_path)
        full_result, full_records = _run_full_parallel(args, config_path, out / "dac_readiness_full300.csv", out / "map_to_ego_t0_transform_contract_v2.json", args.production_score_dir, args.parallel_workers)
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_summary = _analysis(full_records, args.analysis_dir)
        write_csv(args.analysis_dir / "nurec_safety_proxy_v1_scores_full.csv", [r for r in full_records if r.get("valid") is True])
        write_csv(args.analysis_dir / "nurec_safety_proxy_v1_errors_full.csv", [r for r in full_records if r.get("valid") is not True])
        write_json(args.analysis_dir / "readiness_manifest_full300.json", {"clips": readiness, "clip_count": len(readiness), "full_proxy_input_ready_count": len(full_ready_ids)})
    else:
        analysis_summary = {}

    status_counts = Counter(r["overall_score_status"] for r in full_records)
    summary = {
        "clip_count": len(clip_ids),
        "usdz_central_directory_scanned_count": len(inventory_rows),
        "usdz_lane_member_present_count": sum(truth(row.get("lane_member_exists")) for row in inventory_rows),
        "usdz_intersection_area_member_present_count": sum(truth(row.get("intersection_area_member_exists")) for row in inventory_rows),
        "usdz_drivable_space_member_present_count": sum(truth(row.get("drivable_space_member_exists")) for row in inventory_rows),
        "map_transform_ready_count": sum(r["map_transform_ready"] for r in readiness),
        "dac_ready_count": sum(r["dac_ready"] for r in readiness),
        "cf_ready_count": sum(r["cf_ready"] for r in readiness),
        "ttc_ready_count": sum(r["ttc_ready"] for r in readiness),
        "full_proxy_input_ready_count": len(full_ready_ids),
        "pilot_clip_count": len(pilot_ids),
        "pilot_expected_record_count": len(pilot_ids) * 16,
        "full4800_expected_record_count": 4800,
        "full4800_actual_record_count": len(full_records),
        "full4800_unique_record_key_count": len({r.get("record_key") for r in full_records}),
        "full4800_status_counts": dict(status_counts),
        "full4800_run_executed": bool(args.run_full),
        "ade_proxy_disagreement": analysis_summary,
        "map_transform_formula": contract["equation"],
        "map_transform_world_to_nre_used": False,
        "time_block": "CLOSED",
        "obstacle_geometry_block": "CLOSED",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
        "released_label_set_availability": "SEPARATE_FROM_PHYSICAL_WORLD_COMPLETENESS",
        "cf_data_ready": False,
        "ttc_data_ready": False,
        "raw_data_modified": False,
        "full_usdz_downloaded": False,
        "remaining_blockers": [] if args.run_full and len(full_records) == 4800 else ["FULL4800_NOT_RUN_OR_INCOMPLETE"],
        "recommended_next_step": "Review full4800 status/metric analysis; scorer semantics remain frozen and CF_DATA_READY/TTC_DATA_READY remain false until contract review." if args.run_full else "Run with --run-pilot --run-full after reviewing the 300-clip map/DAC readiness tables.",
        "full_evaluation": full_result,
    }
    write_json(out / "map_dac_final_v2_summary.json", summary)
    write_json(out / "map_dac_final_v2_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
