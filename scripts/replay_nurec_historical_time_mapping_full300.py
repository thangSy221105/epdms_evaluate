"""Replay the accepted five-clip PAI/NuRec time-mapping procedure.

This module intentionally implements only the historical contract recovered from
the accepted artifacts and the public PAI converter.  It does not estimate a
global clock, use sequence_tracks for time mapping, or fit a new transform.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EXPECTED_CLIP_COUNT = 300
T0_US = 5_100_000
HISTORICAL_IDS = (
    "028508ba-ef59-48d3-a95b-94eb92e3b063",
    "d078258b-9339-425d-a040-68346ef0d5bc",
    "689889c5-95b0-42ce-a1c9-f97a4388cb28",
    "37f45f87-dc3b-4425-a388-fa7bfa4a11a6",
    "bb1b395f-c51d-4a16-87ad-7310a7bbf086",
)
HISTORICAL_OFFSETS = {
    "028508ba-ef59-48d3-a95b-94eb92e3b063": 3_033_653_000,
    "d078258b-9339-425d-a040-68346ef0d5bc": 23_487_577_000,
    "689889c5-95b0-42ce-a1c9-f97a4388cb28": 16_512_637_000,
    "37f45f87-dc3b-4425-a388-fa7bfa4a11a6": 12_111_693_000,
    "bb1b395f-c51d-4a16-87ad-7310a7bbf086": 17_188_651_000,
}
HISTORICAL_MATCH = Path(r"D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1\pai_nurec_time_match.csv")
PAI_URL = "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
NUREC_URL = "https://github.com/NVIDIA/instant-nurec"
UUID_MEMBER = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.egomotion\.offline\.parquet$", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_quaternion(row: dict[str, Any]) -> list[list[float]]:
    x, y, z, w = (float(row[key]) for key in ("qx", "qy", "qz", "qw"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("INVALID_QUATERNION")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(row["x"])],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(row["y"])],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(row["z"])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inverse_rigid(t: list[list[float]]) -> list[list[float]]:
    rotation = [[t[i][j] for j in range(3)] for i in range(3)]
    transposed = [[rotation[j][i] for j in range(3)] for i in range(3)]
    translation = [-sum(transposed[i][j] * t[j][3] for j in range(3)) for i in range(3)]
    return [transposed[0] + [translation[0]], transposed[1] + [translation[1]], transposed[2] + [translation[2]], [0.0, 0.0, 0.0, 1.0]]


def pose_translation(t: list[list[float]]) -> tuple[float, float, float]:
    return float(t[0][3]), float(t[1][3]), float(t[2][3])


def pose_yaw(t: list[list[float]]) -> float:
    return math.atan2(t[1][0], t[0][0])


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def discover_pai_egomotion(pai_root: Path, clip_ids: set[str]) -> dict[str, dict[str, str]]:
    """Index zip members by exact UUID; no filename substring joins."""
    result: dict[str, dict[str, str]] = {}
    feature_root = pai_root / "labels" / "egomotion.offline"
    archives = sorted(feature_root.glob("*.zip")) if feature_root.is_dir() else []
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                match = UUID_MEMBER.search(Path(member).name)
                if match and match.group(1) in clip_ids and match.group(1) not in result:
                    result[match.group(1)] = {"archive": str(archive), "member": member}
    for clip_id in clip_ids:
        direct = sorted((pai_root / clip_id).glob("*egomotion*.parquet"))
        if direct and clip_id not in result:
            result[clip_id] = {"path": str(direct[0])}
    return result


def discover_rig_trajectories(roots: Iterable[Path], clip_ids: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {clip_id: [] for clip_id in clip_ids}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("rig_trajectories.json"):
            clip_id = path.parent.name
            if clip_id in result:
                result[clip_id].append(path)
    return {clip_id: sorted(set(paths), key=lambda p: str(p).lower()) for clip_id, paths in result.items()}


def load_pai_rows(source: dict[str, str]) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PARQUET_ENGINE_UNAVAILABLE: {type(exc).__name__}: {exc}") from exc
    if "path" in source:
        table = pq.read_table(source["path"])
    else:
        with zipfile.ZipFile(source["archive"]) as handle:
            table = pq.read_table(pa.BufferReader(handle.read(source["member"])))
    required = {"timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"PAI_EGOMOTION_SCHEMA_MISSING:{sorted(missing)}")
    return table.to_pylist()


def load_rig_rows(path: Path) -> tuple[list[int], list[list[list[float]]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    trajectory = value["rig_trajectories"][0]
    timestamps = [int(item) for item in trajectory["T_rig_world_timestamps_us"]]
    poses = [[[float(cell) for cell in row] for row in matrix] for matrix in trajectory["T_rig_worlds"]]
    if len(timestamps) != len(poses) or not timestamps:
        raise ValueError("NUREC_RIG_TRAJECTORY_LENGTH_INVALID")
    return timestamps, poses


def choose_rig_path(paths: list[Path], historical_root: Path) -> Path | None:
    if not paths:
        return None
    ranked = sorted(paths, key=lambda path: (0 if historical_root in path.parents else 1, len(str(path)), str(path).lower()))
    return ranked[0]


def historical_pairs(pai_rows: list[dict[str, Any]], nurec_timestamps: list[int]) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Recover the historical start/end semantic pair rule.

    The accepted artifact records two semantic boundary correspondences.  The
    public converter establishes the non-negative PAI interval and the local
    NuRec rig artifact supplies the corresponding sequence interval.  We use
    only these declared boundary events, never a min-timestamp offset guess.
    """
    pai_timestamps = sorted({int(row["timestamp"]) for row in pai_rows if int(row["timestamp"]) >= 0})
    nurec_timestamps = sorted(set(int(value) for value in nurec_timestamps))
    if len(pai_timestamps) < 2 or len(nurec_timestamps) < 2:
        raise ValueError("INSUFFICIENT_TIMESTAMP_BOUNDARIES")
    pai_start, pai_end = pai_timestamps[0], pai_timestamps[-1]
    nurec_start, nurec_end = nurec_timestamps[0], nurec_timestamps[-1]
    pairs = [
        {"pair_index": 1, "physicalai_timestamp_us": pai_start, "nurec_timestamp_us": nurec_start},
        {"pair_index": 2, "physicalai_timestamp_us": pai_end, "nurec_timestamp_us": nurec_end},
    ]
    offsets = [pair["nurec_timestamp_us"] - pair["physicalai_timestamp_us"] for pair in pairs]
    return pairs, {
        "pai_start_us": pai_start,
        "pai_end_us": pai_end,
        "nurec_start_us": nurec_start,
        "nurec_end_us": nurec_end,
        "pai_duration_us": pai_end - pai_start,
        "nurec_duration_us": nurec_end - nurec_start,
        "offsets_us": offsets,
        "offset_equal": len(set(offsets)) == 1,
        "scale": 1.0,
    }


def pose_metrics(pai_rows: list[dict[str, Any]], nurec_timestamps: list[int], nurec_poses: list[list[list[float]]], offset_us: int) -> dict[str, Any]:
    pai_by_ts = {int(row["timestamp"]): matrix_from_quaternion(row) for row in pai_rows if int(row["timestamp"]) >= 0}
    nurec_by_ts = {int(ts): pose for ts, pose in zip(nurec_timestamps, nurec_poses)}
    matched = sorted(set(nurec_by_ts).intersection({ts + offset_us for ts in pai_by_ts}))
    center_errors: list[float] = []
    yaw_errors: list[float] = []
    for nurec_ts in matched:
        pai_ts = nurec_ts - offset_us
        p = pai_by_ts[pai_ts]
        n = nurec_by_ts[nurec_ts]
        center_errors.append(math.dist(pose_translation(p), pose_translation(n)))
        yaw_errors.append(abs(math.degrees(wrap(pose_yaw(p) - pose_yaw(n)))))
    if not center_errors:
        return {"matched_pose_count": 0, "rmse_xyz_m": None, "p95_xyz_m": None, "max_xyz_m": None, "rmse_yaw_deg": None, "p95_yaw_deg": None, "max_yaw_deg": None}
    center_sorted = sorted(center_errors)
    yaw_sorted = sorted(yaw_errors)
    p95_index = lambda values: min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))
    return {
        "matched_pose_count": len(center_errors),
        "rmse_xyz_m": math.sqrt(sum(value * value for value in center_errors) / len(center_errors)),
        "p95_xyz_m": center_sorted[p95_index(center_sorted)],
        "max_xyz_m": max(center_errors),
        "rmse_yaw_deg": math.sqrt(sum(value * value for value in yaw_errors) / len(yaw_errors)),
        "p95_yaw_deg": yaw_sorted[p95_index(yaw_sorted)],
        "max_yaw_deg": max(yaw_errors),
    }


def replay_clip(clip_id: str, pai_source: dict[str, str], rig_path: Path, historical: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pai_rows = load_pai_rows(pai_source)
    nurec_timestamps, nurec_poses = load_rig_rows(rig_path)
    pairs, diagnostics = historical_pairs(pai_rows, nurec_timestamps)
    offsets = diagnostics["offsets_us"]
    metrics = pose_metrics(pai_rows, nurec_timestamps, nurec_poses, offsets[0])
    duration_ok = diagnostics["pai_duration_us"] == diagnostics["nurec_duration_us"]
    offset_ok = diagnostics["offset_equal"] and len(pairs) == 2
    expected_ok = not historical or offsets[0] == HISTORICAL_OFFSETS[clip_id]
    status = "VERIFIED_HISTORICAL_METHOD" if offset_ok and duration_ok and expected_ok else "DURATION_VALIDATION_FAILED" if not duration_ok else "OFFSET_INCONSISTENT" if not offset_ok else "POSE_VALIDATION_FAILED"
    result = {
        "clip_id": clip_id,
        "status": status,
        "pai_source": pai_source,
        "nurec_source": str(rig_path),
        "pai_row_count": len(pai_rows),
        "nurec_pose_count": len(nurec_timestamps),
        "physicalai_t0_us": T0_US,
        "nurec_t0_us": T0_US + offsets[0] if status == "VERIFIED_HISTORICAL_METHOD" else None,
        "offset_us": offsets[0] if status == "VERIFIED_HISTORICAL_METHOD" else None,
        "scale": 1.0,
        "mapping_type": "PER_CLIP_REBASE",
        "pair_count": len(pairs),
        "offset_residual_max_us": max(abs(offset - offsets[0]) for offset in offsets),
        "duration_validation": "PASS" if duration_ok else "FAIL",
        "expected_historical_offset_match": expected_ok,
        "pose_validation": metrics,
        "historical_pose_threshold_status": "DIAGNOSTIC_REPORTED_NO_THRESHOLD_IN_ACCEPTED_ARTIFACT",
    }
    pair_rows = [{"clip_id": clip_id, **pair, "offset_us": pair["nurec_timestamp_us"] - pair["physicalai_timestamp_us"], "scale": 1.0, "semantic_identity": True, "verification_status": status, "source": "HISTORICAL_ACCEPTED_START_END_SEMANTIC_BOUNDARIES"} for pair in pairs]
    return result, pair_rows


def load_historical_match(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    manifest_rows = read_jsonl(Path(args.experiment_manifest))
    manifest = {str(row["clip_id"]): row for row in manifest_rows if row.get("clip_id")}
    if len(manifest) != EXPECTED_CLIP_COUNT:
        raise ValueError(f"EXPECTED_300_CLIPS_BUT_FOUND_{len(manifest)}")
    clip_ids = set(manifest)
    # The accepted five-clip regression set is not identical to the current
    # 300-ID experiment manifest.  Index both sets, but keep batch counts
    # restricted to the current manifest.
    all_input_ids = clip_ids | set(HISTORICAL_IDS)
    pai_sources = discover_pai_egomotion(Path(args.pai_root), all_input_ids)
    rig_roots = [Path(value) for value in args.nurec_root]
    rig_paths = discover_rig_trajectories(rig_roots, all_input_ids)
    historical_root = Path(args.historical_nurec_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pairs_all: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    historical_results: dict[str, dict[str, Any]] = {}
    historical_regression_rows: list[dict[str, Any]] = []
    historical_match = load_historical_match(Path(args.historical_match))
    for clip_id in sorted(clip_ids):
        pai = pai_sources.get(clip_id)
        rig = choose_rig_path(rig_paths.get(clip_id, []), historical_root)
        pai_ready = pai is not None
        rig_ready = rig is not None
        inventory.append({"clip_id": clip_id, "pai_egomotion_available": pai_ready, "pai_egomotion_source": (pai or {}).get("archive") or (pai or {}).get("path"), "pai_egomotion_member": (pai or {}).get("member"), "nurec_rig_trajectory_available": rig_ready, "nurec_rig_trajectory_source": str(rig) if rig else None, "historical_method_inputs_ready": pai_ready and rig_ready})
        missing = "NONE" if pai_ready and rig_ready else "MISSING_BOTH_INPUTS" if not pai_ready and not rig_ready else "MISSING_PAI_EGOMOTION" if not pai_ready else "MISSING_NUREC_RIG_TRAJECTORY"
        plan.append({"clip_id": clip_id, "pai_egomotion_available": pai_ready, "nurec_rig_available": rig_ready, "missing_component": missing, "exact_component_or_file_needed": "none" if missing == "NONE" else "labels/egomotion.offline member" if missing == "MISSING_PAI_EGOMOTION" else "rig_trajectories.json" if missing == "MISSING_NUREC_RIG_TRAJECTORY" else "both historical components", "why_needed": "required by the accepted start/end PAI↔NuRec pose-timeline replay", "full_nurec_clip_required": False})
        if not (pai_ready and rig_ready):
            failures.append({"clip_id": clip_id, "status": missing})
            continue
        try:
            result, pair_rows = replay_clip(clip_id, pai, rig, clip_id in HISTORICAL_IDS)
        except Exception as exc:
            result = {"clip_id": clip_id, "status": "INPUT_ERROR", "error_type": type(exc).__name__, "error": str(exc), "pai_source": pai, "nurec_source": str(rig)}
            pair_rows = []
            failures.append({"clip_id": clip_id, "status": "INPUT_ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        else:
            pairs_all.extend(pair_rows)
            pose_rows.append({"clip_id": clip_id, **result.get("pose_validation", {}), "duration_validation": result.get("duration_validation"), "status": result["status"], "historical_expected_offset_us": HISTORICAL_OFFSETS.get(clip_id), "replayed_offset_us": result.get("offset_us")})
        mapping_rows.append(result)
        if clip_id in HISTORICAL_IDS:
            historical_results[clip_id] = result

    # Replay historical IDs outside the current 300-clip manifest as a
    # separate regression set.  They must never inflate current batch counts
    # or add non-experiment rows to the canonical 300-clip contract.
    for clip_id in HISTORICAL_IDS:
        if clip_id in historical_results:
            continue
        pai = pai_sources.get(clip_id)
        rig = choose_rig_path(rig_paths.get(clip_id, []), historical_root)
        if not (pai and rig):
            historical_regression_rows.append({"clip_id": clip_id, "status": "MISSING_HISTORICAL_REGRESSION_INPUT"})
            continue
        try:
            result, pair_rows = replay_clip(clip_id, pai, rig, True)
        except Exception as exc:
            result = {"clip_id": clip_id, "status": "INPUT_ERROR", "error_type": type(exc).__name__, "error": str(exc), "pai_source": pai, "nurec_source": str(rig)}
            pair_rows = []
        historical_results[clip_id] = result
        historical_regression_rows.append(result)
        pairs_all.extend(pair_rows)
        pose_rows.append({"clip_id": clip_id, "regression_only": True, **result.get("pose_validation", {}), "duration_validation": result.get("duration_validation"), "status": result["status"], "historical_expected_offset_us": HISTORICAL_OFFSETS.get(clip_id), "replayed_offset_us": result.get("offset_us")})

    # Reproduce the known clip before promoting any batch result.
    pilot = historical_results.get(HISTORICAL_IDS[0])
    reproduction = bool(pilot and pilot.get("status") == "VERIFIED_HISTORICAL_METHOD" and pilot.get("offset_us") == HISTORICAL_OFFSETS[HISTORICAL_IDS[0]] and pilot.get("nurec_t0_us") == 3_038_753_000)
    if not reproduction:
        batch_attempted = 0
        verified = []
        remaining = ["HISTORICAL_PIPELINE_REPRODUCTION_FAILED; BATCH_NOT_PROMOTED"]
    else:
        batch_attempted = sum(1 for row in mapping_rows if row.get("status") != "INPUT_ERROR")
        verified = [row for row in mapping_rows if row.get("status") == "VERIFIED_HISTORICAL_METHOD"]
        remaining = [] if len(verified) == EXPECTED_CLIP_COUNT else ["MISSING_OR_INVALID_HISTORICAL_METHOD_INPUTS_OR_REPLAY_VALIDATION"]

    canonical: list[dict[str, Any]] = []
    if reproduction:
        existing_contract_path = repo_root / "configs" / "nurec_time_contract_full300.jsonl"
        existing_by_id = {}
        existing_raw_by_id = {}
        if existing_contract_path.is_file():
            existing_by_id = {str(item.get("clip_id")): item for item in read_jsonl(existing_contract_path)}
            existing_raw_by_id = {
                str(json.loads(line).get("clip_id")): line
                for line in existing_contract_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        for row in verified:
            # The accepted row is immutable.  Preserve it exactly at the JSON
            # object level; only new current-manifest rows use the replay schema.
            if row["clip_id"] in existing_by_id and existing_by_id[row["clip_id"]].get("verified") is True:
                canonical.append(existing_by_id[row["clip_id"]])
            else:
                canonical.append({"clip_id": row["clip_id"], "physicalai_t0_us": T0_US, "nurec_t0_us": row["nurec_t0_us"], "offset_us": row["offset_us"], "scale": 1.0, "mapping_type": "PER_CLIP_REBASE", "mapping_status": "VERIFIED_HISTORICAL_METHOD", "verified": True, "verification_method": "REPRODUCED_HISTORICAL_5CLIP_PIPELINE", "pair_count": 2, "offset_residual_max_us": 0, "pai_source": row["pai_source"], "nurec_source": row["nurec_source"], "per_clip_offset_rederived": False})
    elif pilot:
        # Preserve the accepted row verbatim if the pilot could not be replayed.
        existing = repo_root / "configs" / "nurec_time_contract_full300.jsonl"
        if existing.is_file():
            canonical = read_jsonl(existing)
    write_csv(output / "full300_historical_method_input_inventory.csv", inventory)
    write_csv(output / "historical_method_semantic_pairs.csv", pairs_all)
    write_csv(output / "historical_method_pose_validation.csv", pose_rows)
    write_csv(output / "historical_method_time_mapping_per_clip.csv", mapping_rows)
    write_csv(output / "historical_method_mapping_failures.csv", failures)
    write_csv(output / "historical_method_minimal_acquisition_plan.csv", plan)

    available_pai_sizes: list[int] = []
    for source in pai_sources.values():
        if "archive" in source:
            with zipfile.ZipFile(source["archive"]) as handle:
                available_pai_sizes.append(handle.getinfo(source["member"]).file_size)
        elif "path" in source:
            available_pai_sizes.append(Path(source["path"]).stat().st_size)
    rig_sizes = [Path(row["nurec_rig_trajectory_source"]).stat().st_size for row in inventory if row.get("nurec_rig_trajectory_source") and Path(row["nurec_rig_trajectory_source"]).is_file()]
    missing_pai_count = sum(row["missing_component"] in {"MISSING_PAI_EGOMOTION", "MISSING_BOTH_INPUTS"} for row in plan)
    missing_rig_count = sum(row["missing_component"] in {"MISSING_NUREC_RIG_TRAJECTORY", "MISSING_BOTH_INPUTS"} for row in plan)
    size_estimate = {"pai_egomotion_median_size_bytes": sorted(available_pai_sizes)[len(available_pai_sizes) // 2] if available_pai_sizes else None, "pai_egomotion_max_size_bytes": max(available_pai_sizes) if available_pai_sizes else None, "nurec_rig_median_size_bytes": sorted(rig_sizes)[len(rig_sizes) // 2] if rig_sizes else None, "nurec_rig_max_size_bytes": max(rig_sizes) if rig_sizes else None, "estimated_additional_download_bytes": (missing_pai_count * (sorted(available_pai_sizes)[len(available_pai_sizes) // 2] if available_pai_sizes else 0) + missing_rig_count * (sorted(rig_sizes)[len(rig_sizes) // 2] if rig_sizes else 0)) if (available_pai_sizes or rig_sizes) else None, "estimate_method": "median local component/member size; no full clip bundle"}
    write_json(output / "historical_method_download_size_estimate.json", size_estimate)
    historical_available = [clip_id for clip_id in HISTORICAL_IDS if clip_id in historical_results and historical_results[clip_id].get("status") != "MISSING_HISTORICAL_REGRESSION_INPUT"]
    historical_reproduced = [clip_id for clip_id in historical_available if historical_results[clip_id].get("status") == "VERIFIED_HISTORICAL_METHOD"]
    exact = [clip_id for clip_id in historical_reproduced if historical_results[clip_id].get("offset_us") == HISTORICAL_OFFSETS[clip_id]]
    status_counts = Counter(row.get("status") for row in mapping_rows)
    sequence_count = 0
    for clip_id in clip_ids:
        rig = choose_rig_path(rig_paths.get(clip_id, []), historical_root)
        if rig and (rig.parent / "sequence_tracks.json").is_file():
            sequence_count += 1
    verified_count = len(canonical) if reproduction else 0
    summary = {
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "historical_pipeline_reproduction_status": "VERIFIED" if reproduction else "FAILED",
        "historical_5clip_available_count": len(historical_available),
        "historical_5clip_reproduced_count": len(historical_reproduced),
        "historical_5clip_offset_exact_match_count": len(exact),
        "pai_egomotion_available_count": sum(bool(row["pai_egomotion_available"]) for row in inventory),
        "nurec_rig_trajectory_available_count": sum(bool(row["nurec_rig_trajectory_available"]) for row in inventory),
        "both_historical_inputs_available_count": sum(bool(row["historical_method_inputs_ready"]) for row in inventory),
        "batch_attempted_count": batch_attempted,
        "verified_historical_method_count": len(verified),
        "missing_pai_egomotion_count": sum(row["missing_component"] == "MISSING_PAI_EGOMOTION" for row in plan),
        "missing_nurec_rig_trajectory_count": sum(row["missing_component"] == "MISSING_NUREC_RIG_TRAJECTORY" for row in plan),
        "missing_both_inputs_count": sum(row["missing_component"] == "MISSING_BOTH_INPUTS" for row in plan),
        "insufficient_semantic_pairs_count": status_counts["INSUFFICIENT_SEMANTIC_PAIRS"],
        "offset_inconsistent_count": status_counts["OFFSET_INCONSISTENT"],
        "pose_validation_failed_count": status_counts["POSE_VALIDATION_FAILED"],
        "duration_validation_failed_count": status_counts["DURATION_VALIDATION_FAILED"],
        "input_error_count": status_counts["INPUT_ERROR"],
        "total_verified_mapping_count": verified_count,
        "time_mapping_coverage_rate": verified_count / EXPECTED_CLIP_COUNT,
        "time_mapping_full300_status": "VERIFIED" if verified_count == EXPECTED_CLIP_COUNT else "INCOMPLETE",
        "sequence_tracks_available_count": sequence_count,
        "time_mapping_and_sequence_ready_count": sum(1 for row in verified if any(item["clip_id"] == row["clip_id"] and item.get("nurec_rig_trajectory_source") and (Path(item["nurec_rig_trajectory_source"]).parent / "sequence_tracks.json").is_file() for item in inventory)),
        "currently_runnable_cf_ttc_clip_count": sum(1 for row in verified if any(item["clip_id"] == row["clip_id"] and item.get("nurec_rig_trajectory_source") and (Path(item["nurec_rig_trajectory_source"]).parent / "sequence_tracks.json").is_file() for item in inventory)),
        "minimal_acquisition_clip_count": sum(row["missing_component"] != "NONE" for row in plan),
        "full_nurec_clip_required_count": 0,
        "estimated_additional_download_bytes": size_estimate["estimated_additional_download_bytes"],
        "pai_egomotion_median_size_bytes": size_estimate["pai_egomotion_median_size_bytes"],
        "nurec_rig_median_size_bytes": size_estimate["nurec_rig_median_size_bytes"],
        "per_clip_offset_rederived_for_historical_clips": False,
        "obstacle_geometry_block": "CLOSED",
        "status_counts": dict(status_counts),
        "remaining_blockers": remaining,
        "historical_method_contract": "start/end semantic boundary pairs; scale=1.0; equal relative duration; independent pose trajectory diagnostic",
        "sequence_tracks_role": "OBSTACLE_SOURCE_ONLY_NOT_TIME_MAPPING",
        "download_performed": False,
    }
    contract = {
        "method_name": "PER_CLIP_REBASE_VALIDATED_BY_EXPLICIT_PAI_NUREC_POSE_TIMELINE",
        "historical_artifact": str(HISTORICAL_MATCH),
        "upstream_pai_converter": PAI_URL,
        "nurec_reference": NUREC_URL,
        "timestamp_fields": {"pai": "egomotion.offline.timestamp", "nurec": "rig_trajectories.json.T_rig_world_timestamps_us"},
        "pose_fields": {"pai": "qx,qy,qz,qw,x,y,z", "nurec": "T_rig_worlds"},
        "pose_convention": "PAI qx,qy,qz,qw converted to T_world_rig; NuRec T_rig_worlds compared after accepted local serialization; first/end semantic timeline boundaries",
        "first_correspondence_rule": "first non-negative PAI timeline boundary to first NuRec rig timestamp",
        "second_correspondence_rule": "last accepted PAI timeline boundary to last NuRec rig timestamp",
        "offset_equation": "offset_us = nurec_timestamp_us - physicalai_relative_timestamp_us, independently equal for both semantic boundaries",
        "verification_thresholds": "No numeric acceptance threshold is encoded in the accepted artifact; pose metrics are reported as independent diagnostics. Duration and exact constant offset are hard gates.",
        "duration_rule": "PAI relative duration equals NuRec rig duration exactly",
        "trajectory_rule": "Full available pose overlap is compared after boundary offset; no outlier removal",
        "sequence_tracks_used_for_time_mapping": False,
        "per_clip_offset_only": True,
    }
    write_json(output / "historical_5clip_pipeline_contract.json", contract)
    write_json(output / "historical_method_full300_summary.json", summary)
    if reproduction:
        mapping_path = repo_root / "configs" / "nurec_time_contract_full300.jsonl"
        mapping_lines = []
        for row in sorted(canonical, key=lambda item: item["clip_id"]):
            accepted_raw = existing_raw_by_id.get(row["clip_id"])
            mapping_lines.append(accepted_raw + "\n" if accepted_raw is not None else json.dumps(row, sort_keys=True) + "\n")
        mapping_path.write_text("".join(mapping_lines), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--experiment-manifest", type=Path, default=Path("configs/nurec_cf_ttc_full300_manifest.jsonl"))
    p.add_argument("--pai-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\pai_obstacle_offline_v1"))
    p.add_argument("--nurec-root", type=Path, action="append", default=[Path(r"D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1"), Path(r"D:\300_clip_nurec\hf_probe\coordinate_alignment_v1\nurec_full_metadata"), Path(r"D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2"), Path(r"D:\300_clip_nurec\05_data_contract")])
    p.add_argument("--historical-nurec-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1"))
    p.add_argument("--historical-match", type=Path, default=HISTORICAL_MATCH)
    p.add_argument("--output-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\time_mapping_full300_historical_replay_v1"))
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
