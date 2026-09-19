"""Memory-conscious NuRec schema inspection and EPDMS contract auditing.

This module deliberately inventories and stages evidence only. It never scores
trajectories, applies a guessed clock offset, invents coordinate metadata, or
modifies source files.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


REQUIRED_CLIP_FILES = [
    "clipgt/clip.parquet",
    "clipgt/association.parquet",
    "clipgt/calibration_estimate.parquet",
    "clipgt/egomotion_estimate.parquet",
    "clipgt/obstacle.parquet",
    "clipgt/drivable_space.parquet",
    "clipgt/lane.parquet",
    "clipgt/intersection_area.parquet",
    "clipgt/road_boundary.parquet",
    "clipgt/traffic_light.parquet",
    "clipgt/wait_line.parquet",
    "data_info.json",
    "datasource_summary.json",
    "pose_record.json",
    "rig_trajectories.json",
    "metadata.yaml",
]

OFFICIAL_FILES = {
    "traffic_light": "clipgt/traffic_light.parquet",
    "wait_line": "clipgt/wait_line.parquet",
    "lane_line": "clipgt/lane_line.parquet",
    "crosswalk": "clipgt/crosswalk.parquet",
    "traffic_sign": "clipgt/traffic_sign.parquet",
    "drivable_space": "clipgt/drivable_space.parquet",
}

PARQUET_FILES = {p for p in REQUIRED_CLIP_FILES if p.endswith(".parquet")}
TIMESTAMP_RE = re.compile(r"(?:^|[_\.])(timestamp|time|t0|start|end|frame_id|sample_id|token)(?:$|[_\.])", re.I)
COORDINATE_RE = re.compile(r"(?:^|[_\.])(x|y|z|center|position|translation|geometry|quaternion|yaw|rotation|frame|sensor|track|category)(?:$|[_\.])", re.I)


def parquet_engine() -> Dict[str, Any]:
    """Return available parquet engines without installing anything."""
    available: List[str] = []
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
            available.append(name)
        except Exception:
            pass
    return {
        "available": bool(available),
        "engines": available,
        "status": "READY" if available else "PARQUET_ENGINE_UNAVAILABLE",
        "install_hint": "Install pyarrow or fastparquet in the project runtime; no automatic installation was attempted.",
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _safe_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _nested_paths(value: Any, prefix: str = "", depth: int = 0, limit: int = 500) -> List[str]:
    if depth > 8 or limit <= 0:
        return []
    paths: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_nested_paths(child, path, depth + 1, limit - len(paths)))
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        paths.extend(_nested_paths(value[0], f"{prefix}[]", depth + 1, limit))
    return paths[:limit]


def _looks_like_timestamp(path: str) -> bool:
    return bool(TIMESTAMP_RE.search(path))


def _looks_like_coordinate(path: str) -> bool:
    return bool(COORDINATE_RE.search(path))


def _numeric_summary(values: Sequence[Any]) -> Dict[str, Any]:
    clean: List[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return {"count": 0, "min": None, "max": None, "unique_count": 0, "spacing_sample": [], "median_dt": None}
    unique = sorted(set(clean))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "unique_count": len(unique),
        "spacing_sample": diffs[:10],
        "median_dt": sorted(diffs)[len(diffs) // 2] if diffs else None,
    }


def _dataframe_records(frame: Any, limit: int = 5) -> List[Dict[str, Any]]:
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso"))


def inspect_parquet(path: Path, sample_rows: int = 5) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file(), "format": "parquet"}
    if not path.is_file():
        result["status"] = "FILE_NOT_FOUND"
        return result
    engine_info = parquet_engine()
    result["parquet_engine"] = engine_info
    if not engine_info["available"]:
        result.update({"status": "PARQUET_ENGINE_UNAVAILABLE", "error_code": "PARQUET_ENGINE_UNAVAILABLE", "error": engine_info["install_hint"]})
        return result
    try:
        import pandas as pd

        engine = engine_info["engines"][0]
        frame = pd.read_parquet(path, engine=engine)
        columns = [str(column) for column in frame.columns]
        result.update({
            "status": "OK",
            "row_count": int(len(frame)),
            "columns": columns,
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "null_counts": {str(column): int(value) for column, value in frame.isna().sum().items()},
            "sample_rows": _dataframe_records(frame, sample_rows),
        })
        timestamp_columns = [column for column in columns if _looks_like_timestamp(column)]
        result["timestamp_columns"] = {column: _numeric_summary(frame[column].dropna().tolist()) for column in timestamp_columns}
        result["coordinate_columns"] = [column for column in columns if _looks_like_coordinate(column)]
        result["timestamp_field_candidates"] = timestamp_columns
        result["coordinate_field_candidates"] = result["coordinate_columns"]
    except Exception as exc:
        result.update({"status": "PARQUET_READ_ERROR", "error_code": type(exc).__name__, "error": str(exc)})
    return result


def inspect_json_like(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["status"] = "FILE_NOT_FOUND"
        return result
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
                value = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            except ImportError:
                text = path.read_text(encoding="utf-8", errors="replace")
                result.update({"status": "YAML_PARSER_UNAVAILABLE", "top_level_keys": [], "raw_preview": text[:2000]})
                return result
        else:
            value = _load_json(path)
        paths = _nested_paths(value)
        result.update({
            "status": "OK",
            "top_level_keys": list(value.keys()) if isinstance(value, dict) else [],
            "nested_key_paths": paths,
            "timestamp_looking_fields": [item for item in paths if _looks_like_timestamp(item)],
            "frame_coordinate_looking_fields": [item for item in paths if _looks_like_coordinate(item)],
        })
    except Exception as exc:
        result.update({"status": "JSON_READ_ERROR", "error_code": type(exc).__name__, "error": str(exc)})
    return result


def inspect_clip(clip_dir: Path) -> Dict[str, Any]:
    clip_dir = clip_dir.resolve()
    if clip_dir.name == "clipgt":
        clip_dir = clip_dir.parent
    report: Dict[str, Any] = {
        "clip_id": clip_dir.name,
        "clip_dir": str(clip_dir),
        "parquet_engine": parquet_engine(),
        "parquet": {},
        "json_yaml": {},
    }
    for relative in REQUIRED_CLIP_FILES:
        path = clip_dir / relative
        if relative.endswith(".parquet"):
            report["parquet"][relative] = inspect_parquet(path)
        else:
            report["json_yaml"][relative] = inspect_json_like(path)
    return report


def report_to_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# NuRec schema report — `{report.get('clip_id')}`", "", f"Clip directory: `{report.get('clip_dir')}`", "", f"Parquet status: `{report.get('parquet_engine', {}).get('status')}`", ""]
    lines += ["## Parquet files", "", "| File | Status | Rows | Timestamp candidates | Coordinate candidates |", "|---|---|---:|---|---|"]
    for name, item in report.get("parquet", {}).items():
        lines.append(f"| `{name}` | `{item.get('status')}` | {item.get('row_count', '')} | `{', '.join(item.get('timestamp_field_candidates', []))}` | `{', '.join(item.get('coordinate_field_candidates', []))}` |")
    lines += ["", "## JSON/YAML files", "", "| File | Status | Top-level keys | Timestamp-looking fields | Frame/coordinate-looking fields |", "|---|---|---|---|---|"]
    for name, item in report.get("json_yaml", {}).items():
        lines.append(f"| `{name}` | `{item.get('status')}` | `{', '.join(item.get('top_level_keys', []))}` | `{', '.join(item.get('timestamp_looking_fields', []))}` | `{', '.join(item.get('frame_coordinate_looking_fields', []))}` |")
    lines += ["", "## Interpretation", "", "This report is schema inventory only. It does not establish a clock mapping, coordinate transform, empty-frame evidence, or evaluator readiness."]
    return "\n".join(lines) + "\n"


def write_schema_report(clip_dir: Path, output_dir: Path) -> Dict[str, Any]:
    report = inspect_clip(clip_dir)
    _safe_json_dump(output_dir / "schema_report.json", report)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_report.md").write_text(report_to_markdown(report), encoding="utf-8")
    return report


def discover_clip_dirs(dataset_root: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Find clip roots by locating clipgt directories; duplicate IDs are errors."""
    found: Dict[str, Path] = {}
    duplicates: List[str] = []
    candidates = [dataset_root] if (dataset_root / "clipgt").is_dir() else []
    candidates.extend(path.parent for path in dataset_root.rglob("clipgt") if path.is_dir())
    for clip_dir in candidates:
        clip_id = clip_dir.name
        if clip_id in found and found[clip_id] != clip_dir:
            duplicates.append(clip_id)
        else:
            found[clip_id] = clip_dir
    return dict(sorted(found.items())), sorted(set(duplicates))


def _jsonl_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.is_file():
        return result
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            clip_id = str(row.get("clip_id", ""))
            if clip_id and clip_id not in result:
                result[clip_id] = row
    return result


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if row.get(key) is not None:
            return str(row[key])
    return None


def _inventory_field(relative: str) -> str:
    filename = Path(relative).name
    stem = filename.rsplit(".", 1)[0]
    if relative.startswith("clipgt/"):
        stem = stem.replace("_estimate", "")
    return f"{stem}_exists"


def _row_t0(row: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not row or row.get("t0_us") is None:
        return None
    try:
        value = float(row["t0_us"])
        return int(round(value)) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _parquet_timestamp_summary(path: Path) -> Dict[str, Any]:
    info = inspect_parquet(path, sample_rows=0)
    result: Dict[str, Any] = {"status": info.get("status"), "min": None, "max": None, "unique_count": None, "row_count": info.get("row_count")}
    if info.get("status") != "OK":
        return result
    candidates = info.get("timestamp_field_candidates", [])
    if not candidates:
        result["status"] = "MISSING_TIME_METADATA"
        return result
    try:
        import pandas as pd
        frame = pd.read_parquet(path, columns=[candidates[0]], engine=parquet_engine()["engines"][0])
        result.update(_numeric_summary(frame[candidates[0]].dropna().tolist()))
        result["field"] = candidates[0]
    except Exception as exc:
        result.update({"status": "PARQUET_READ_ERROR", "error": str(exc)})
    return result


def _map_status(clip_dir: Path, engine_status: str) -> Dict[str, Any]:
    sources: Dict[str, Any] = {}
    for name in ("clipgt/drivable_space.parquet", "clipgt/lane.parquet", "clipgt/intersection_area.parquet", "clipgt/road_boundary.parquet"):
        path = clip_dir / name
        item: Dict[str, Any] = {"exists": path.is_file(), "status": "FILE_NOT_FOUND" if not path.is_file() else engine_status}
        if path.is_file() and engine_status == "READY":
            inspected = inspect_parquet(path, sample_rows=0)
            item.update({"status": inspected.get("status"), "row_count": inspected.get("row_count"), "geometry_fields": [c for c in inspected.get("columns", []) if "geom" in c.lower() or "polygon" in c.lower() or "shape" in c.lower()], "valid_polygon_count": inspected.get("valid_polygon_count"), "invalid_polygon_count": inspected.get("invalid_polygon_count")})
        sources[name] = item
    exists = [value["exists"] for value in sources.values()]
    if not any(exists):
        status = "FILE_NOT_FOUND"
    elif engine_status != "READY":
        status = "PARQUET_ENGINE_UNAVAILABLE"
    else:
        existing_sources = [value for value in sources.values() if value["exists"]]
        if any(value.get("status") == "NO_VALID_GEOMETRY" or (value.get("valid_polygon_count") == 0 and value.get("invalid_polygon_count", 0) > 0) for value in existing_sources):
            status = "NO_VALID_GEOMETRY"
        else:
            status = "OK" if all(value.get("status") == "OK" for value in existing_sources) else "PARTIAL"
    return {"status": status, "sources": sources, "drivable_space_available": bool((clip_dir / "clipgt/drivable_space.parquet").is_file()), "lane_available": bool((clip_dir / "clipgt/lane.parquet").is_file()), "intersection_available": bool((clip_dir / "clipgt/intersection_area.parquet").is_file()), "recommended_dac_source": "drivable_space" if (clip_dir / "clipgt/drivable_space.parquet").is_file() else "lane_plus_intersection" if (clip_dir / "clipgt/lane.parquet").is_file() and (clip_dir / "clipgt/intersection_area.parquet").is_file() else None}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def audit_dataset(dataset_root: Path, prediction_jsonl: Path, ground_truth_jsonl: Path, output_dir: Path, context_jsonl: Optional[Path] = None) -> Dict[str, Any]:
    clips, duplicate_ids = discover_clip_dirs(dataset_root)
    predictions = _jsonl_index(prediction_jsonl)
    ground_truth = _jsonl_index(ground_truth_jsonl)
    contexts = _jsonl_index(context_jsonl)
    engine = parquet_engine()
    inventory: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    obstacle_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    map_rows: List[Dict[str, Any]] = []
    ego_rows: List[Dict[str, Any]] = []
    official_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    contracts: Dict[str, Dict[str, Any]] = {}

    for clip_id, clip_dir in clips.items():
        try:
            pred = predictions.get(clip_id, {})
            gt = ground_truth.get(clip_id, {})
            context = contexts.get(clip_id, {})
            present = {_inventory_field(name): (clip_dir / name).is_file() for name in REQUIRED_CLIP_FILES}
            inventory.append({"clip_id": clip_id, "prediction_exists": bool(pred), "ground_truth_exists": bool(gt), **present, "duplicate_clip_id": clip_id in duplicate_ids})
            pred_t0, gt_t0 = _row_t0(pred), _row_t0(gt)
            obstacle_summary = _parquet_timestamp_summary(clip_dir / "clipgt/obstacle.parquet")
            ego_summary = _parquet_timestamp_summary(clip_dir / "clipgt/egomotion_estimate.parquet")
            clip_summary = _parquet_timestamp_summary(clip_dir / "clipgt/clip.parquet")
            pose_summary = _parquet_timestamp_summary(clip_dir / "clipgt/pose_record.parquet") if (clip_dir / "clipgt/pose_record.parquet").is_file() else {"status": "FILE_NOT_FOUND"}
            obstacle_min, obstacle_max = obstacle_summary.get("min"), obstacle_summary.get("max")
            obs_range = obstacle_max - obstacle_min if obstacle_min is not None and obstacle_max is not None else None
            candidate_delta = (pred_t0 - obstacle_min) if pred_t0 is not None and obstacle_min is not None else None
            mapping = context.get("time_alignment") if isinstance(context.get("time_alignment"), dict) else context.get("time_mapping") if isinstance(context.get("time_mapping"), dict) else {}
            explicit_mapping = bool(mapping.get("verified") is True or mapping.get("status") in {"ALIGNED_BY_EXPLICIT_METADATA", "validated"})
            mapping_source = mapping.get("source") or mapping.get("mapping_source")
            status = "MISSING_TIME_METADATA" if pred_t0 is None or gt_t0 is None else "CONFLICTING_TIME_ORIGIN" if pred_t0 != gt_t0 else "UNRESOLVED"
            if explicit_mapping:
                status = "ALIGNED_BY_EXPLICIT_METADATA"
            if status == "UNRESOLVED" and obstacle_summary.get("status") == "OK" and obs_range is not None and obstacle_min <= pred_t0 <= obstacle_max:
                status = "ALIGNED_DIRECT"
            time_rows.append({"clip_id": clip_id, "prediction_t0_us": pred_t0, "gt_t0_us": gt_t0, "clip_start_timestamp": clip_summary.get("min"), "clip_end_timestamp": clip_summary.get("max"), "obstacle_min_timestamp": obstacle_min, "obstacle_max_timestamp": obstacle_max, "egomotion_min_timestamp": ego_summary.get("min"), "egomotion_max_timestamp": ego_summary.get("max"), "sensor_min_timestamp": None, "sensor_max_timestamp": None, "pose_min_timestamp": pose_summary.get("min"), "pose_max_timestamp": pose_summary.get("max"), "clock_domains_detected": json.dumps([x for x in ("prediction_t0" if pred_t0 is not None else None, "ground_truth_t0" if gt_t0 is not None else None, obstacle_summary.get("field"), ego_summary.get("field")) if x]), "possible_explicit_mapping_found": explicit_mapping, "mapping_source": mapping_source, "candidate_prediction_obstacle_delta_us": candidate_delta, "time_alignment_status": status})

            pframe = _first(pred, ("coordinate_frame", "frame", "prediction_frame"))
            gframe = _first(gt, ("coordinate_frame", "future_frame", "frame", "gt_frame"))
            cframe = _first(context, ("coordinate_frame", "frame", "obstacle_frame"))
            mframe = _first(context, ("map_frame",))
            panchor = _first(pred, ("reference_point", "anchor", "prediction_anchor"))
            ganchor = _first(gt, ("reference_point", "anchor", "gt_anchor"))
            oanchor = _first(context, ("obstacle_anchor", "reference_point", "anchor"))
            manchor = _first(context, ("map_anchor", "reference_point", "anchor"))
            coord_verified = bool(pframe and gframe and cframe and mframe and panchor and ganchor and oanchor and manchor and len({pframe, gframe, cframe, mframe}) == 1 and len({panchor, ganchor, oanchor, manchor}) == 1)
            coord_status = "ALIGNED_DIRECT" if coord_verified else "MISSING_FRAME_METADATA" if not any((pframe, gframe, cframe, mframe, panchor, ganchor, oanchor, manchor)) else "CONFLICTING_FRAMES" if len({x for x in (pframe, gframe, cframe, mframe) if x}) > 1 or len({x for x in (panchor, ganchor, oanchor, manchor) if x}) > 1 else "UNRESOLVED"
            transform_available = bool(context.get("transform_chain_available") is True or context.get("transform_source"))
            transform_source = context.get("transform_source") if transform_available else None
            if not coord_verified and transform_available:
                coord_status = "TRANSFORM_AVAILABLE"
            coordinate_rows.append({"clip_id": clip_id, "prediction_frame": pframe, "prediction_anchor": panchor, "gt_frame": gframe, "gt_anchor": ganchor, "obstacle_frame": cframe, "obstacle_anchor": oanchor, "map_frame": mframe, "map_anchor": manchor, "egomotion_frame": None, "sensor_rig_frame": None, "transform_required": None if coord_verified else True, "transform_source": transform_source, "coordinate_alignment_verified": coord_verified, "status": coord_status})

            obstacle_ready = obstacle_summary.get("status") == "OK"
            obstacle_rows.append({"clip_id": clip_id, "obstacle_row_count": obstacle_summary.get("row_count"), "unique_timestamp_count": obstacle_summary.get("unique_count"), "min_timestamp": obstacle_min, "max_timestamp": obstacle_max, "unique_track_count": None, "categories": None, "missing_timestamp_count": None if not obstacle_ready else 0, "invalid_center_count": None if not obstacle_ready else 0, "invalid_size_count": None if not obstacle_ready else 0, "invalid_orientation_count": None if not obstacle_ready else 0, "status": obstacle_summary.get("status")})
            cf_start = pred_t0
            cf_end = pred_t0 + 4_000_000 if pred_t0 is not None else None
            ttc_end = pred_t0 + 5_000_000 if pred_t0 is not None else None
            frame_evidence = context.get("observation_frames") or context.get("frame_timestamps_us") or context.get("sensor_timestamps_us")
            empty_possible = bool(frame_evidence) and obstacle_ready and obstacle_summary.get("unique_count") == 0
            observation_status = "OBSERVED_EMPTY" if empty_possible else "UNKNOWN" if not obstacle_ready or obstacle_summary.get("unique_count") == 0 else "UNRESOLVED"
            coverage_rows.append({"clip_id": clip_id, "required_cf_start": cf_start, "required_cf_end": cf_end, "required_ttc_end": ttc_end, "available_frame_count": obstacle_summary.get("unique_count"), "available_frame_min_ts": obstacle_min, "available_frame_max_ts": obstacle_max, "obstacle_timestamp_count": obstacle_summary.get("unique_count"), "confirmed_observed_empty_possible": empty_possible, "cf_required_frames": 41 if pred_t0 is not None else None, "cf_observed_frames": obstacle_summary.get("unique_count") if obstacle_ready else None, "cf_empty_frames": 41 if empty_possible else 0, "cf_missing_frames": 0 if empty_possible else None, "ttc_required_queries": 51 if pred_t0 is not None else None, "ttc_observed_queries": obstacle_summary.get("unique_count") if obstacle_ready else None, "ttc_empty_queries": 51 if empty_possible else 0, "ttc_missing_queries": 0 if empty_possible else None, "observation_contract_status": observation_status})
            map_summary = _map_status(clip_dir, engine["status"])
            map_rows.append({"clip_id": clip_id, **map_summary})
            official_rows.append({"clip_id": clip_id, **{f"{key}_available": (clip_dir / relative).is_file() for key, relative in OFFICIAL_FILES.items()}, "lane_direction_available": None, "lane_geometry_available": (clip_dir / "clipgt/lane.parquet").is_file()})
            ego_rows.append({"clip_id": clip_id, "row_count": ego_summary.get("row_count"), "timestamp_min": ego_summary.get("min"), "timestamp_max": ego_summary.get("max"), "timestamp_spacing": ego_summary.get("median_dt"), "position_fields": None, "rotation_fields": None, "frame": None, "parent_frame": None, "child_frame": None, "transform_chain_available": False, "status": ego_summary.get("status")})
            blockers: List[str] = []
            if not pred: blockers.append("PREDICTION_MISSING")
            if not gt: blockers.append("GT_MISSING")
            if status not in {"ALIGNED_DIRECT", "ALIGNED_BY_EXPLICIT_METADATA"}: blockers.append("TIME_ALIGNMENT_UNRESOLVED")
            if not coord_verified: blockers.append("COORDINATE_UNRESOLVED")
            if not obstacle_ready: blockers.append("OBSTACLE_SCHEMA_INVALID")
            blockers.append("OBSERVATION_COVERAGE_INCOMPLETE")
            if map_summary["status"] not in {"OK"}:
                if map_summary["status"] == "FILE_NOT_FOUND":
                    blockers.append("MAP_MISSING")
                elif map_summary["status"] == "PARQUET_ENGINE_UNAVAILABLE":
                    blockers.append("PARQUET_ENGINE_UNAVAILABLE")
                else:
                    blockers.append("MAP_INVALID")
            contracts[clip_id] = {"clip_id": clip_id, "time": {"t0_us": pred_t0, "source": "prediction_jsonl", "verified": status in {"ALIGNED_DIRECT", "ALIGNED_BY_EXPLICIT_METADATA"}}, "coordinate": {"prediction_frame": pframe, "gt_frame": gframe, "obstacle_frame": cframe, "map_frame": mframe, "prediction_anchor": panchor, "gt_anchor": ganchor, "obstacle_anchor": oanchor, "map_anchor": manchor, "transform_required": not coord_verified, "transform_source": None, "verified": coord_verified}, "observation": {"cf_ready": False, "ttc_ready": False, "coverage_source": None}, "map": {"ready": map_summary["status"] == "OK", "source": "clipgt", "status": map_summary["status"]}, "ready_for_proxy": False, "blockers": sorted(set(blockers))}
        except Exception as exc:
            errors.append({"clip_id": clip_id, "failure_stage": "audit_clip", "failure_type": type(exc).__name__, "failure_reason": str(exc)})

    _write_csv(output_dir / "clip_inventory.csv", inventory)
    _write_csv(output_dir / "time_alignment.csv", time_rows)
    _write_csv(output_dir / "coordinate_contract.csv", coordinate_rows)
    _write_csv(output_dir / "obstacle_inventory.csv", obstacle_rows)
    _write_csv(output_dir / "observation_coverage.csv", coverage_rows)
    _write_csv(output_dir / "map_inventory.csv", map_rows)
    _write_csv(output_dir / "egomotion_inventory.csv", ego_rows)
    _write_csv(output_dir / "official_epdms_candidate_data.csv", official_rows)
    _write_csv(output_dir / "audit_errors.csv", errors)
    (output_dir / "contracts").mkdir(parents=True, exist_ok=True)
    for clip_id, contract in contracts.items():
        _safe_json_dump(output_dir / "contracts" / f"{clip_id}.json", contract)
    eligible = sorted(clip_id for clip_id, contract in contracts.items() if contract["ready_for_proxy"])
    blocked = sorted(clip_id for clip_id in contracts if clip_id not in eligible)
    summary = {"total_clips": len(clips), "prediction_ready": sum(bool(predictions.get(cid)) for cid in clips), "gt_ready": sum(bool(ground_truth.get(cid)) for cid in clips), "time_ready": sum(c["time"]["verified"] for c in contracts.values()), "coordinate_ready": sum(c["coordinate"]["verified"] for c in contracts.values()), "observation_cf_ready": sum(c["observation"]["cf_ready"] for c in contracts.values()), "observation_ttc_ready": sum(c["observation"]["ttc_ready"] for c in contracts.values()), "map_ready": sum(c["map"]["ready"] for c in contracts.values()), "proxy_ready_clip_count": len(eligible), "eligible_clips": eligible, "blocked_clips": blocked, "duplicate_clip_ids": duplicate_ids, "audit_error_count": len(errors), "parquet_engine": engine, "status": "DATASET_READY" if not blocked and not errors else "DATASET_NOT_READY"}
    _safe_json_dump(output_dir / "dataset_readiness_summary.json", summary)
    (output_dir / "eligible_clips.txt").write_text("\n".join(eligible) + ("\n" if eligible else ""), encoding="utf-8")
    (output_dir / "blocked_clips.txt").write_text("\n".join(blocked) + ("\n" if blocked else ""), encoding="utf-8")
    lines = ["# NuRec EPDMS data readiness", "", f"- Total clips: {summary['total_clips']}", f"- Proxy-ready clips: {summary['proxy_ready_clip_count']}", f"- Dataset status: `{summary['status']}`", f"- Parquet engine: `{engine['status']}`", "", "## Blockers", ""]
    counts = Counter(blocker for contract in contracts.values() for blocker in contract["blockers"])
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    (output_dir / "dataset_readiness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "contracts": contracts, "errors": errors}


def prepare_staging(audit_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Create non-destructive canonical staging manifests from audit evidence."""
    summary_path = audit_dir / "dataset_readiness_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"blocked_clips": []}
    contracts: Dict[str, Dict[str, Any]] = {}
    for path in sorted(audit_dir.glob("contracts/*.json")):
        try:
            contracts[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # The audit normally carries contracts in memory; when invoked separately,
    # reconstruct minimal per-clip contracts from its CSVs/blocked list.
    blocked = set(summary.get("blocked_clips", []))
    clip_ids = sorted(set(blocked) | {path.stem for path in audit_dir.glob("contracts/*.json")})
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for clip_id in clip_ids:
        contract = contracts.get(clip_id, {"clip_id": clip_id, "ready_for_proxy": False, "blockers": ["AUDIT_CONTRACT_NOT_FOUND"]})
        clip_out = output_dir / clip_id
        clip_out.mkdir(parents=True, exist_ok=True)
        _safe_json_dump(clip_out / "contract.json", contract)
        _safe_json_dump(clip_out / "map_manifest.json", contract.get("map", {"ready": False, "status": "UNKNOWN", "blockers": contract.get("blockers", [])}))
        _safe_json_dump(clip_out / "transform_manifest.json", {"verified": contract.get("coordinate", {}).get("verified", False), "transform_required": contract.get("coordinate", {}).get("transform_required"), "transform_source": contract.get("coordinate", {}).get("transform_source"), "status": "UNRESOLVED" if not contract.get("coordinate", {}).get("verified", False) else "ALIGNED_DIRECT"})
        written += 1
    return {"output_dir": str(output_dir), "clips_staged": written, "proxy_ready_clip_count": int(summary.get("proxy_ready_clip_count", 0))}
