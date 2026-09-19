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

from tools.epdms.condition_identity import parse_condition_identity
from tools.epdms.observation_contract import build_ttc_projection_timestamps


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
COORDINATE_RE = re.compile(r"(?:^|[_\.])(x|y|z|w|center|position|translation|geometry|quaternion|yaw|rotation|frame|sensor|track|category)(?:$|[_\.])", re.I)
PHYSICAL_TIMESTAMP_TERMS = {"timestamp", "timestamp_us", "timestamp_micros", "timestamp_microseconds", "timestamp_ms", "timestamp_ns", "time_us", "time_micros", "time_microseconds", "time_ms", "time_ns", "start_micros", "end_micros"}


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


def _leaf_paths(value: Any, prefix: str) -> List[str]:
    """Return scalar leaf paths for pandas/pyarrow struct-like values."""
    if isinstance(value, Mapping):
        paths: List[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        for child in value:
            if child is not None:
                return _leaf_paths(child, f"{prefix}[]")
        return []
    return [prefix]


def _nested_frame_paths(frame: Any) -> List[str]:
    paths: List[str] = []
    for column in frame.columns:
        name = str(column)
        paths.append(name)
        samples = [value for value in frame[name].tolist() if value is not None][:64]
        for sample in samples:
            paths.extend(_leaf_paths(sample, name))
    return list(dict.fromkeys(paths))


def _path_value(value: Any, path: str) -> Any:
    parts = path.split(".") if path else []

    def walk(current: Any, index: int) -> Any:
        if index >= len(parts):
            return current
        part = parts[index]
        list_marker = part.endswith("[]")
        key = part[:-2] if list_marker else part
        if isinstance(current, Mapping):
            child = current.get(key)
            return walk(child, index + 1)
        if isinstance(current, (list, tuple)):
            return [walk(child, index) for child in current]
        return None

    return walk(value, 0)


def _frame_path_values(frame: Any, path: str) -> List[Any]:
    top, *rest = path.split(".")
    if top not in frame.columns:
        return []
    values = frame[top].tolist()
    if not rest:
        return values
    suffix = ".".join(rest)
    return [_path_value(value, suffix) for value in values]


def _path_numeric_summary(frame: Any, path: str) -> Dict[str, Any]:
    return _numeric_summary(_frame_path_values(frame, path))


def _has_geometry_points(value: Any, minimum: int = 2) -> bool:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return len(value) >= minimum
    except TypeError:
        return False


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, bytes)) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _timestamp_value_valid(value: Any) -> bool:
    if not _is_finite_number(value):
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _timestamp_summary(values: Sequence[Any]) -> Dict[str, Any]:
    valid = [int(float(value)) for value in values if _timestamp_value_valid(value)]
    invalid_count = len(values) - len(valid)
    unique = sorted(set(valid))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return {
        "timestamp_total_count": len(values),
        "timestamp_valid_count": len(valid),
        "timestamp_invalid_count": invalid_count,
        "count": len(valid),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        "unique_count": len(unique),
        "spacing_sample": diffs[:10],
        "median_dt": sorted(diffs)[len(diffs) // 2] if diffs else None,
    }


def _timestamp_field_name(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower()


def _resolve_parquet_timestamp_field(frame: Any, candidates: Sequence[str]) -> Dict[str, Any]:
    summaries = {field: _timestamp_summary(_frame_path_values(frame, field)) for field in candidates}
    usable = [field for field in candidates if summaries[field]["timestamp_valid_count"] > 0]
    canonical = [field for field in usable if field == "key.timestamp_micros"]
    if canonical:
        selected = canonical[0]
        status = "SELECTED_CANONICAL"
    else:
        physical = [field for field in usable if _timestamp_field_name(field) in PHYSICAL_TIMESTAMP_TERMS]
        if len(physical) == 1:
            selected = physical[0]
            status = "SELECTED_EXPLICIT"
        elif len(physical) > 1:
            selected = None
            status = "AMBIGUOUS_TIMESTAMP_FIELDS"
        else:
            selected = None
            status = "MISSING_TIME_METADATA"
    return {"timestamp_candidates": summaries, "selected_timestamp_field": selected, "timestamp_selection_status": status}


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
        nested_paths = _nested_frame_paths(frame)
        result.update({
            "status": "OK",
            "row_count": int(len(frame)),
            "columns": columns,
            "nested_field_paths": nested_paths,
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "null_counts": {str(column): int(value) for column, value in frame.isna().sum().items()},
            "sample_rows": _dataframe_records(frame, sample_rows),
        })
        timestamp_columns = [path for path in nested_paths if _looks_like_timestamp(path)]
        result.update(_resolve_parquet_timestamp_field(frame, timestamp_columns))
        result["timestamp_columns"] = result["timestamp_candidates"]
        result["coordinate_columns"] = [path for path in nested_paths if _looks_like_coordinate(path)]
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
    lines += ["## Parquet files", "", "| File | Status | Rows | Selected timestamp | Timestamp selection | Timestamp candidates | Coordinate candidates |", "|---|---|---:|---|---|---|---|"]
    for name, item in report.get("parquet", {}).items():
        lines.append(f"| `{name}` | `{item.get('status')}` | {item.get('row_count', '')} | `{item.get('selected_timestamp_field') or ''}` | `{item.get('timestamp_selection_status') or ''}` | `{', '.join(item.get('timestamp_field_candidates', []))}` | `{', '.join(item.get('coordinate_field_candidates', []))}` |")
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


def _strict_clip_id(row: Mapping[str, Any]) -> Optional[str]:
    value = row.get("clip_id")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _jsonl_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    return _jsonl_index_with_errors(path, "jsonl")[0]


def _jsonl_index_with_errors(path: Optional[Path], source: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    result: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    duplicates: List[str] = []
    if path is None or not path.is_file():
        return result, errors, duplicates
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append({"source": source, "line_number": line_number, "failure_type": "MALFORMED_JSON", "failure_reason": str(exc)})
                continue
            if not isinstance(row, dict):
                errors.append({"source": source, "line_number": line_number, "failure_type": "INVALID_JSON_ROW", "failure_reason": "JSONL row is not an object"})
                continue
            clip_id = _strict_clip_id(row)
            if clip_id is None:
                errors.append({"source": source, "line_number": line_number, "failure_type": "MISSING_CLIP_ID", "failure_reason": "clip_id is missing or empty"})
                continue
            if clip_id in result:
                duplicates.append(clip_id)
                errors.append({"source": source, "line_number": line_number, "failure_type": "DUPLICATE_CLIP_ID", "failure_reason": clip_id})
                continue
            result[clip_id] = {**row, "clip_id": clip_id}
    return result, errors, sorted(set(duplicates))


def _load_prediction_conditions(path: Optional[Path]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[str], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    """Load all prediction conditions; uniqueness is clip|mode|alpha."""
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[Dict[str, Any]] = []
    duplicate_keys: List[str] = []
    identity_errors: Dict[str, List[Dict[str, Any]]] = {}
    seen_keys: set[str] = set()
    stats: Dict[str, Dict[str, int]] = {}
    if path is None or not path.is_file():
        return predictions, errors, duplicate_keys, identity_errors, stats
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append({"source": "prediction", "line_number": line_number, "failure_type": "MALFORMED_JSON", "failure_reason": str(exc)})
                continue
            if not isinstance(row, dict):
                errors.append({"source": "prediction", "line_number": line_number, "failure_type": "INVALID_JSON_ROW", "failure_reason": "JSONL row is not an object"})
                continue
            clip_id = _strict_clip_id(row)
            if clip_id is None:
                errors.append({"source": "prediction", "line_number": line_number, "failure_type": "PREDICTION_IDENTITY_INVALID", "failure_reason": "MISSING_CLIP_ID: clip_id is missing, null, or blank"})
                continue
            normalized = {**row, "clip_id": clip_id}
            mode_value = normalized.get("mode")
            if mode_value is None or not str(mode_value).strip():
                error = {"source": "prediction", "line_number": line_number, "clip_id": clip_id, "failure_type": "PREDICTION_IDENTITY_INVALID", "failure_reason": "MISSING_MODE: mode is missing or blank"}
                errors.append(error)
                identity_errors.setdefault(clip_id, []).append(error)
                continue
            normalized["mode"] = str(mode_value).strip()
            if "alpha" not in normalized:
                error = {"source": "prediction", "line_number": line_number, "clip_id": clip_id, "failure_type": "PREDICTION_IDENTITY_INVALID", "failure_reason": "MISSING_ALPHA: alpha is missing"}
                errors.append(error)
                identity_errors.setdefault(clip_id, []).append(error)
                continue
            identity = parse_condition_identity(normalized, row_index=line_number - 1)
            if not identity.valid:
                error = {"source": "prediction", "line_number": line_number, "clip_id": clip_id, "failure_type": "PREDICTION_IDENTITY_INVALID", "failure_reason": f"INVALID_ALPHA: {identity.failure_reason}"}
                errors.append(error)
                identity_errors.setdefault(clip_id, []).append(error)
                continue
            normalized["alpha"] = identity.alpha
            clip_stats = stats.setdefault(clip_id, {"raw_condition_count": 0, "unique_condition_count": 0, "duplicate_condition_count": 0})
            clip_stats["raw_condition_count"] += 1
            if identity.record_key in seen_keys:
                duplicate_keys.append(identity.record_key)
                clip_stats["duplicate_condition_count"] += 1
                errors.append({"source": "prediction", "line_number": line_number, "clip_id": clip_id, "failure_type": "PREDICTION_CONDITION_DUPLICATE", "failure_reason": identity.record_key})
            else:
                predictions.setdefault(clip_id, []).append(normalized)
                clip_stats["unique_condition_count"] += 1
            seen_keys.add(identity.record_key)
    return predictions, errors, sorted(set(duplicate_keys)), identity_errors, stats


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


def _condition_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[Any]:
    values: List[Any] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(value)
    return values


def _common_condition_value(rows: Sequence[Mapping[str, Any]], key: str) -> Tuple[Any, List[Any], bool]:
    raw_values = _condition_values(rows, key)
    values = list(dict.fromkeys(str(value) if isinstance(value, str) else value for value in raw_values))
    return (values[0] if len(values) == 1 else None, values, len(values) <= 1)


def _condition_first_values(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[str]:
    values: List[str] = []
    for row in rows:
        value = _first(row, keys)
        if value is not None:
            values.append(value.strip())
    return list(dict.fromkeys(values))


def _query_grid_settings(evaluation_config: Any = None) -> Dict[str, Any]:
    if evaluation_config is None:
        return {"horizon_s": 4.0, "frequency_hz": 10.0, "future_poses": 40, "ttc_horizon_s": 1.0, "source": "DEFAULT_EVALUATION_CONFIG", "verified": False}
    values = evaluation_config if isinstance(evaluation_config, Mapping) else {name: getattr(evaluation_config, name) for name in ("horizon_s", "frequency_hz", "future_poses", "ttc_horizon_s") if hasattr(evaluation_config, name)}
    horizon_s = float(values["horizon_s"])
    frequency_hz = float(values["frequency_hz"])
    future_poses = int(values.get("future_poses", round(horizon_s * frequency_hz)))
    ttc_horizon_s = float(values["ttc_horizon_s"])
    if not math.isfinite(horizon_s) or not math.isfinite(frequency_hz) or not math.isfinite(ttc_horizon_s) or horizon_s <= 0 or frequency_hz <= 0 or ttc_horizon_s < 0 or future_poses < 1:
        raise ValueError("QUERY_GRID_CONTRACT_UNRESOLVED: invalid effective evaluation settings")
    source = str(values.get("source") or getattr(evaluation_config, "config_path", None) or "EFFECTIVE_EVALUATION_CONFIG")
    return {"horizon_s": horizon_s, "frequency_hz": frequency_hz, "future_poses": future_poses, "ttc_horizon_s": ttc_horizon_s, "source": source, "verified": bool(values.get("verified", evaluation_config is not None))}


def _canonical_query_timestamps(t0_us: int, evaluation_config: Any = None) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Build CF and scorer TTC grids from one effective, auditable config."""
    import numpy as np

    settings = _query_grid_settings(evaluation_config)
    cf = [int(t0_us) + int(round(index * 1_000_000 / settings["frequency_hz"])) for index in range(settings["future_poses"] + 1)]
    ttc = build_ttc_projection_timestamps(np.asarray(cf, dtype=np.int64), settings["ttc_horizon_s"])
    settings["cf_required_frames"] = len(cf)
    settings["ttc_required_queries"] = len(ttc)
    settings["query_grid_fingerprint"] = json.dumps({key: settings[key] for key in ("horizon_s", "frequency_hz", "future_poses", "ttc_horizon_s")}, sort_keys=True, separators=(",", ":"))
    return cf, [int(value) for value in ttc.tolist()], settings


def _parquet_timestamp_values(path: Path) -> set[int]:
    info = inspect_parquet(path, sample_rows=0)
    selected = info.get("selected_timestamp_field")
    if info.get("status") != "OK" or not selected:
        return set()
    try:
        import pandas as pd
        frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
        return {int(float(value)) for value in _frame_path_values(frame, selected) if _timestamp_value_valid(value)}
    except Exception:
        return set()


def _matched_evidence(query_values: Sequence[int], evidence_values: set[int], tolerance_us: int) -> set[int]:
    matched: set[int] = set()
    if not evidence_values:
        return matched
    ordered = sorted(evidence_values)
    for query in query_values:
        nearest = min(ordered, key=lambda value: abs(value - query))
        if abs(nearest - query) <= tolerance_us:
            matched.add(int(query))
    return matched


def _normalize_frame_evidence(context: Mapping[str, Any]) -> set[int]:
    for key in ("observation_frames", "frame_timestamps_us", "sensor_timestamps_us"):
        if key not in context or context.get(key) is None:
            continue
        raw_values = context.get(key)
        if not isinstance(raw_values, list):
            return set()
        normalized: set[int] = set()
        for item in raw_values:
            value = item
            if isinstance(item, Mapping):
                value = next((item.get(name) for name in ("timestamp_micros", "timestamp_us", "t_us", "timestamp") if item.get(name) is not None), None)
            if _timestamp_value_valid(value):
                normalized.add(int(float(value)))
        return normalized
    return set()


def _coverage_counts(required: Sequence[int], evidence: set[int], obstacles: set[int], complete: bool, tolerance_us: int) -> Dict[str, Any]:
    """Mirror evaluate_query_coverage: object matching may be tolerant, empty is exact."""
    import numpy as np
    from tools.epdms.observation_contract import evaluate_query_coverage

    required_array = np.asarray(list(required), dtype=np.int64)
    obstacle_array = np.asarray(sorted(obstacles), dtype=np.int64)
    obstacle_index = {int(value): [{}] for value in obstacles}
    confirmed_empty = set(evidence) if complete else set()
    _, observed, empty, scorer_missing, _ = evaluate_query_coverage(
        required_array, obstacle_index, obstacle_array, confirmed_empty, half_step_us=tolerance_us
    )
    # Object evidence is authoritative and does not require frame evidence.
    # An exact frame attestation without an object is UNKNOWN when the table
    # completeness is unverified; it is never confirmed empty in that case.
    unknown = 0
    if not complete:
        states, _, _, _, _ = evaluate_query_coverage(
            required_array, obstacle_index, obstacle_array, set(), half_step_us=tolerance_us
        )
        unknown = sum(
            1
            for query, state in zip(required_array.tolist(), states)
            if int(query) in evidence and state.name != "OBSERVED_WITH_OBJECTS"
        )
    missing = max(0, scorer_missing - unknown)
    return {"required": len(required), "observed": observed, "empty": empty, "unknown": unknown, "missing": missing, "matched": observed + empty + unknown}


def _parquet_timestamp_summary(path: Path) -> Dict[str, Any]:
    info = inspect_parquet(path, sample_rows=0)
    result: Dict[str, Any] = {"status": info.get("status"), "min": None, "max": None, "unique_count": None, "row_count": info.get("row_count"), "timestamp_candidates": info.get("timestamp_candidates", {}), "selected_timestamp_field": info.get("selected_timestamp_field"), "timestamp_selection_status": info.get("timestamp_selection_status")}
    if info.get("status") != "OK":
        return result
    candidates = info.get("timestamp_field_candidates", [])
    selected = info.get("selected_timestamp_field")
    if not candidates or not selected:
        result["status"] = info.get("timestamp_selection_status") or "MISSING_TIME_METADATA"
        return result
    try:
        import pandas as pd
        frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
        summary = _timestamp_summary(_frame_path_values(frame, selected))
        result.update(summary)
        result["field"] = selected
        result["status"] = "OK" if summary["timestamp_valid_count"] > 0 else "MISSING_TIME_METADATA"
    except Exception as exc:
        result.update({"status": "PARQUET_READ_ERROR", "error": str(exc)})
    return result


def _parquet_clip_interval_summary(path: Path) -> Dict[str, Any]:
    """Read clip start/end as an interval, without treating both as one clock field."""
    info = inspect_parquet(path, sample_rows=0)
    if info.get("status") != "OK":
        return {"status": info.get("status"), "min": None, "max": None}
    start_field = "key.time_range.start_micros"
    end_field = "key.time_range.end_micros"
    if start_field not in info.get("nested_field_paths", []) or end_field not in info.get("nested_field_paths", []):
        return {"status": "MISSING_TIME_METADATA", "min": None, "max": None}
    try:
        import pandas as pd
        frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
        starts = _timestamp_summary(_frame_path_values(frame, start_field))
        ends = _timestamp_summary(_frame_path_values(frame, end_field))
        if starts["timestamp_invalid_count"] or ends["timestamp_invalid_count"] or not starts["count"] or not ends["count"]:
            return {"status": "INVALID_TIME_METADATA", "min": starts["min"], "max": ends["max"], "start_field": start_field, "end_field": end_field}
        return {"status": "OK", "min": starts["min"], "max": ends["max"], "start_field": start_field, "end_field": end_field}
    except Exception as exc:
        return {"status": "PARQUET_READ_ERROR", "min": None, "max": None, "error": str(exc)}


def _obstacle_inventory(path: Path) -> Dict[str, Any]:
    """Audit NuRec's nested obstacle struct without dropping invalid rows."""
    info = inspect_parquet(path, sample_rows=0)
    if info.get("status") != "OK":
        return {"status": info.get("status"), "row_count": info.get("row_count")}
    required = {
        "key.timestamp_micros",
        "obstacle.trackline_id",
        "obstacle.category",
        "obstacle.center.x",
        "obstacle.center.y",
        "obstacle.center.z",
        "obstacle.size.x",
        "obstacle.size.y",
        "obstacle.size.z",
        "obstacle.orientation.x",
        "obstacle.orientation.y",
        "obstacle.orientation.z",
        "obstacle.orientation.w",
    }
    fields = set(info.get("nested_field_paths", []))
    missing = sorted(required - fields)
    if missing:
        return {"status": "MISSING_REQUIRED_FIELDS", "row_count": info.get("row_count"), "missing_fields": missing}
    import pandas as pd

    frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
    timestamps = _frame_path_values(frame, "key.timestamp_micros")
    tracks = _frame_path_values(frame, "obstacle.trackline_id")
    categories = _frame_path_values(frame, "obstacle.category")
    centers = [_frame_path_values(frame, f"obstacle.center.{axis}") for axis in "xyz"]
    sizes = [_frame_path_values(frame, f"obstacle.size.{axis}") for axis in "xyz"]
    orientations = [_frame_path_values(frame, f"obstacle.orientation.{axis}") for axis in "xyzw"]
    def finite(value: Any) -> bool:
        return _is_finite_number(value)

    def valid_text(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            if math.isnan(float(value)):
                return False
        except (TypeError, ValueError):
            pass
        return bool(str(value).strip())

    missing_timestamp = sum(not _timestamp_value_valid(value) for value in timestamps)
    invalid_track_id = sum(not valid_text(value) for value in tracks)
    invalid_category = sum(not valid_text(value) for value in categories)
    invalid_center = sum(not all(finite(axis[index]) for axis in centers) for index in range(len(timestamps)))
    invalid_size = sum(not all(finite(axis[index]) and float(axis[index]) > 0 for axis in sizes) for index in range(len(timestamps)))
    invalid_orientation = 0
    epsilon = 1e-12
    for index in range(len(timestamps)):
        values = [axis[index] for axis in orientations]
        invalid_orientation += int(not all(finite(value) for value in values) or math.sqrt(sum(float(value) ** 2 for value in values)) <= epsilon)
    invalid_row_count = 0
    for index in range(len(timestamps)):
        row_invalid = (
            not _timestamp_value_valid(timestamps[index])
            or not valid_text(tracks[index])
            or not valid_text(categories[index])
            or not all(finite(axis[index]) for axis in centers)
            or not all(finite(axis[index]) and float(axis[index]) > 0 for axis in sizes)
            or not all(finite(axis[index]) for axis in orientations)
            or math.sqrt(sum(float(axis[index]) ** 2 for axis in orientations)) <= epsilon
        )
        invalid_row_count += int(row_invalid)
    status = "OK" if invalid_row_count == 0 else "INVALID_ROWS"
    time_summary = _timestamp_summary(timestamps)
    return {
        "status": status,
        "row_count": len(frame),
        "valid_row_count": len(frame) - invalid_row_count,
        "invalid_row_count": invalid_row_count,
        "unique_timestamp_count": time_summary["unique_count"],
        "min_timestamp": time_summary["min"],
        "max_timestamp": time_summary["max"],
        "unique_track_count": len({str(value).strip() for value in tracks if valid_text(value)}),
        "categories": sorted({str(value).strip() for value in categories if valid_text(value)}),
        "missing_timestamp_count": missing_timestamp,
        "invalid_track_id_count": invalid_track_id,
        "invalid_category_count": invalid_category,
        "invalid_center_count": invalid_center,
        "invalid_size_count": invalid_size,
        "invalid_orientation_count": invalid_orientation,
        "field_map": {
            "timestamp": "key.timestamp_micros",
            "track_id": "obstacle.trackline_id",
            "category": "obstacle.category",
            "center": "obstacle.center.{x,y,z}",
            "size": "obstacle.size.{x,y,z}",
            "orientation": "obstacle.orientation.{x,y,z,w}",
        },
    }


def _finite_point(value: Any) -> bool:
    return isinstance(value, Mapping) and all(_is_finite_number(value.get(axis)) for axis in ("x", "y", "z"))


def _valid_point_sequence(value: Any, minimum: int = 2) -> bool:
    return _has_geometry_points(value, minimum) and all(_finite_point(point) for point in value)


def _map_status(clip_dir: Path, engine_status: str) -> Dict[str, Any]:
    sources: Dict[str, Any] = {}
    for name in ("clipgt/drivable_space.parquet", "clipgt/lane.parquet", "clipgt/intersection_area.parquet", "clipgt/road_boundary.parquet"):
        path = clip_dir / name
        item: Dict[str, Any] = {"exists": path.is_file(), "status": "FILE_NOT_FOUND" if not path.is_file() else engine_status}
        if path.is_file() and engine_status == "READY":
            inspected = inspect_parquet(path, sample_rows=0)
            geometry_fields = [
                field for field in inspected.get("nested_field_paths", [])
                if any(token in field.lower() for token in ("location", "geometry", "polygon", "left_rail", "right_rail", "shape"))
            ]
            item.update({"status": inspected.get("status"), "row_count": inspected.get("row_count"), "geometry_fields": geometry_fields, "structurally_valid_geometry_count": None, "structurally_invalid_geometry_count": None})
            if inspected.get("status") == "OK" and geometry_fields:
                import pandas as pd

                frame = pd.read_parquet(path, engine=parquet_engine()["engines"][0])
                source_name = path.stem
                if source_name == "lane":
                    left = _frame_path_values(frame, "lane.left_rail")
                    right = _frame_path_values(frame, "lane.right_rail")
                    valid = [_valid_point_sequence(a) and _valid_point_sequence(b) for a, b in zip(left, right)]
                    item["geometry_field"] = "lane.left_rail + lane.right_rail"
                else:
                    top = source_name
                    geometry_path = next((field for field in geometry_fields if field == f"{top}.location"), None)
                    if geometry_path is None:
                        geometry_path = next((field.split("[]", 1)[0] for field in geometry_fields if field.startswith(f"{top}.location[]")), None)
                    locations = _frame_path_values(frame, geometry_path) if geometry_path else []
                    valid = [_valid_point_sequence(value) for value in locations]
                    item["geometry_field"] = geometry_path
                item["structurally_valid_geometry_count"] = sum(valid)
                item["structurally_invalid_geometry_count"] = len(valid) - sum(valid)
                if item["structurally_valid_geometry_count"] == 0:
                    item["status"] = "NO_VALID_GEOMETRY"
        sources[name] = item
    exists = [value["exists"] for value in sources.values()]
    if not any(exists):
        status = "FILE_NOT_FOUND"
    elif engine_status != "READY":
        status = "PARQUET_ENGINE_UNAVAILABLE"
    else:
        existing_sources = [value for value in sources.values() if value["exists"]]
        if any(value.get("status") == "NO_VALID_GEOMETRY" or (value.get("structurally_valid_geometry_count") == 0 and value.get("structurally_invalid_geometry_count", 0) > 0) for value in existing_sources):
            status = "NO_VALID_GEOMETRY"
        else:
            status = "STRUCTURAL_GEOMETRY_AVAILABLE" if all(value.get("status") == "OK" and value.get("structurally_valid_geometry_count", 0) > 0 for value in existing_sources) else "PARTIAL"
    drivable = sources.get("clipgt/drivable_space.parquet", {})
    lane = sources.get("clipgt/lane.parquet", {})
    intersection = sources.get("clipgt/intersection_area.parquet", {})
    drivable_structural = drivable.get("structurally_valid_geometry_count", 0) > 0
    candidate_lane = lane.get("structurally_valid_geometry_count", 0) > 0
    candidate_intersection = intersection.get("structurally_valid_geometry_count", 0) > 0
    dac_candidate = drivable_structural or (candidate_lane and candidate_intersection)
    recommended = "drivable_space" if drivable_structural else "lane_plus_intersection" if candidate_lane and candidate_intersection else None
    dac_verified = False
    if dac_candidate and status not in {"FILE_NOT_FOUND", "PARQUET_ENGINE_UNAVAILABLE", "NO_VALID_GEOMETRY"}:
        status = "UNVERIFIED_DAC_GEOMETRY" if drivable_structural else "DAC_CANDIDATE_AVAILABLE"
    return {"status": status, "ready": dac_verified, "sources": sources, "geometry_available": any(exists), "dac_candidate_available": dac_candidate, "dac_geometry_verified": dac_verified, "drivable_space_available": bool((clip_dir / "clipgt/drivable_space.parquet").is_file()), "lane_available": bool((clip_dir / "clipgt/lane.parquet").is_file()), "intersection_available": bool((clip_dir / "clipgt/intersection_area.parquet").is_file()), "recommended_dac_source": recommended}


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


def audit_dataset(dataset_root: Path, prediction_jsonl: Path, ground_truth_jsonl: Path, output_dir: Path, context_jsonl: Optional[Path] = None, evaluation_config: Any = None) -> Dict[str, Any]:
    clips, duplicate_ids = discover_clip_dirs(dataset_root)
    predictions_by_clip, prediction_errors, prediction_condition_duplicates, prediction_identity_errors, prediction_condition_stats = _load_prediction_conditions(prediction_jsonl)
    ground_truth, ground_truth_errors, ground_truth_duplicates = _jsonl_index_with_errors(ground_truth_jsonl, "ground_truth")
    contexts, context_errors, context_duplicates = _jsonl_index_with_errors(context_jsonl, "context")
    engine = parquet_engine()
    inventory: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []
    obstacle_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    map_rows: List[Dict[str, Any]] = []
    ego_rows: List[Dict[str, Any]] = []
    official_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = prediction_errors + ground_truth_errors + context_errors
    contracts: Dict[str, Dict[str, Any]] = {}

    for clip_id, clip_dir in clips.items():
        try:
            pred_rows = predictions_by_clip.get(clip_id, [])
            pred = pred_rows[0] if pred_rows else {}
            gt = ground_truth.get(clip_id, {})
            context = contexts.get(clip_id, {})
            raw_prediction_t0_values = [_row_t0(row) for row in pred_rows]
            prediction_t0_values = sorted({value for value in raw_prediction_t0_values if value is not None})
            prediction_t0_consistent = bool(pred_rows) and len(raw_prediction_t0_values) == len(pred_rows) and len(prediction_t0_values) == 1
            common_pred_t0 = prediction_t0_values[0] if prediction_t0_consistent else None
            prediction_frame_values = _condition_first_values(pred_rows, ("coordinate_frame", "frame", "prediction_frame"))
            prediction_anchor_values = _condition_first_values(pred_rows, ("reference_point", "anchor", "prediction_anchor"))
            prediction_frame_present = sum(_first(row, ("coordinate_frame", "frame", "prediction_frame")) is not None for row in pred_rows)
            prediction_anchor_present = sum(_first(row, ("reference_point", "anchor", "prediction_anchor")) is not None for row in pred_rows)
            prediction_frame_consistent = len(prediction_frame_values) <= 1 and prediction_frame_present in {0, len(pred_rows)}
            prediction_anchor_consistent = len(prediction_anchor_values) <= 1 and prediction_anchor_present in {0, len(pred_rows)}
            present = {_inventory_field(name): (clip_dir / name).is_file() for name in REQUIRED_CLIP_FILES}
            prediction_modes = sorted({str(row.get("mode")) for row in pred_rows})
            prediction_alphas = sorted({float(row.get("alpha")) for row in pred_rows})
            condition_stats = prediction_condition_stats.get(clip_id, {"raw_condition_count": 0, "unique_condition_count": 0, "duplicate_condition_count": 0})
            inventory.append({"clip_id": clip_id, "prediction_exists": bool(pred_rows), "ground_truth_exists": bool(gt), "prediction_raw_condition_count": condition_stats["raw_condition_count"], "prediction_unique_condition_count": condition_stats["unique_condition_count"], "prediction_duplicate_condition_count": condition_stats["duplicate_condition_count"], "prediction_condition_count": len(pred_rows), "prediction_modes": prediction_modes, "prediction_alphas": prediction_alphas, "prediction_t0_values": prediction_t0_values, "prediction_frame_values": prediction_frame_values, "prediction_anchor_values": prediction_anchor_values, **present, "duplicate_clip_id": clip_id in duplicate_ids, "prediction_condition_duplicate": clip_id in {key.split("|", 1)[0] for key in prediction_condition_duplicates}, "ground_truth_duplicate": clip_id in ground_truth_duplicates, "context_duplicate": clip_id in context_duplicates})
            pred_t0, gt_t0 = common_pred_t0, _row_t0(gt)
            obstacle_summary = _parquet_timestamp_summary(clip_dir / "clipgt/obstacle.parquet")
            obstacle_details = _obstacle_inventory(clip_dir / "clipgt/obstacle.parquet") if (clip_dir / "clipgt/obstacle.parquet").is_file() and obstacle_summary.get("status") == "OK" else {}
            if obstacle_details:
                obstacle_summary.update({key: value for key, value in obstacle_details.items() if key not in {"status", "row_count"}})
                obstacle_summary["status"] = obstacle_details.get("status", obstacle_summary.get("status"))
            ego_summary = _parquet_timestamp_summary(clip_dir / "clipgt/egomotion_estimate.parquet")
            clip_summary = _parquet_clip_interval_summary(clip_dir / "clipgt/clip.parquet")
            pose_summary = _parquet_timestamp_summary(clip_dir / "clipgt/pose_record.parquet") if (clip_dir / "clipgt/pose_record.parquet").is_file() else {"status": "FILE_NOT_FOUND"}
            obstacle_min, obstacle_max = obstacle_summary.get("min"), obstacle_summary.get("max")
            obs_range = obstacle_max - obstacle_min if obstacle_min is not None and obstacle_max is not None else None
            candidate_delta = (pred_t0 - obstacle_min) if pred_t0 is not None and obstacle_min is not None else None
            mapping = context.get("time_alignment") if isinstance(context.get("time_alignment"), dict) else context.get("time_mapping") if isinstance(context.get("time_mapping"), dict) else {}
            mapping_source = mapping.get("source") or mapping.get("mapping_source")
            explicit_mapping = bool(mapping.get("verified") is True and bool(mapping_source))
            status = "MISSING_TIME_METADATA" if pred_t0 is None or gt_t0 is None else "CONFLICTING_TIME_ORIGIN" if pred_t0 != gt_t0 else "UNRESOLVED"
            if explicit_mapping:
                status = mapping.get("status") if mapping.get("status") in {"ALIGNED_DIRECT", "ALIGNED_BY_EXPLICIT_METADATA"} else "ALIGNED_BY_EXPLICIT_METADATA"
            numeric_overlap = bool(pred_t0 is not None and obstacle_min is not None and obstacle_max is not None and obstacle_min <= pred_t0 <= obstacle_max)
            time_rows.append({"clip_id": clip_id, "prediction_t0_us": pred_t0, "prediction_t0_values": prediction_t0_values, "prediction_condition_count": len(pred_rows), "gt_t0_us": gt_t0, "clip_start_timestamp": clip_summary.get("min"), "clip_end_timestamp": clip_summary.get("max"), "obstacle_min_timestamp": obstacle_min, "obstacle_max_timestamp": obstacle_max, "egomotion_min_timestamp": ego_summary.get("min"), "egomotion_max_timestamp": ego_summary.get("max"), "sensor_min_timestamp": None, "sensor_max_timestamp": None, "pose_min_timestamp": pose_summary.get("min"), "pose_max_timestamp": pose_summary.get("max"), "clock_domains_detected": json.dumps([x for x in ("prediction_t0" if pred_t0 is not None else None, "ground_truth_t0" if gt_t0 is not None else None, obstacle_summary.get("selected_timestamp_field"), ego_summary.get("selected_timestamp_field")) if x]), "possible_explicit_mapping_found": explicit_mapping, "mapping_source": mapping_source, "candidate_prediction_obstacle_delta_us": candidate_delta, "numeric_range_overlap": numeric_overlap, "time_alignment_status": "PREDICTION_T0_CONFLICT" if len(prediction_t0_values) > 1 else status})

            pframe = prediction_frame_values[0] if prediction_frame_consistent and prediction_frame_values else None
            gframe = _first(gt, ("coordinate_frame", "future_frame", "frame", "gt_frame"))
            cframe = _first(context, ("coordinate_frame", "frame", "obstacle_frame"))
            mframe = _first(context, ("map_frame",))
            panchor = prediction_anchor_values[0] if prediction_anchor_consistent and prediction_anchor_values else None
            ganchor = _first(gt, ("reference_point", "anchor", "gt_anchor"))
            oanchor = _first(context, ("obstacle_anchor", "reference_point", "anchor"))
            manchor = _first(context, ("map_anchor", "reference_point", "anchor"))
            coord_verified = bool(pframe and gframe and cframe and mframe and panchor and ganchor and oanchor and manchor and len({pframe, gframe, cframe, mframe}) == 1 and len({panchor, ganchor, oanchor, manchor}) == 1)
            coord_status = "ALIGNED_DIRECT" if coord_verified else "MISSING_FRAME_METADATA" if not any((pframe, gframe, cframe, mframe, panchor, ganchor, oanchor, manchor)) else "CONFLICTING_FRAMES" if not prediction_frame_consistent or not prediction_anchor_consistent or len({x for x in (pframe, gframe, cframe, mframe) if x}) > 1 or len({x for x in (panchor, ganchor, oanchor, manchor) if x}) > 1 else "UNRESOLVED"
            metadata_transform_available = all((clip_dir / relative).is_file() for relative in ("rig_trajectories.json", "clipgt/calibration_estimate.parquet"))
            transform_available = bool(context.get("transform_chain_available") is True or context.get("transform_source") or metadata_transform_available)
            transform_source = context.get("transform_source") if context.get("transform_source") else "rig_trajectories.json + calibration_estimate.parquet" if metadata_transform_available else None
            if not coord_verified and transform_available:
                coord_status = "TRANSFORM_METADATA_AVAILABLE"
            coordinate_rows.append({"clip_id": clip_id, "prediction_frame": pframe, "prediction_anchor": panchor, "gt_frame": gframe, "gt_anchor": ganchor, "obstacle_frame": cframe, "obstacle_anchor": oanchor, "map_frame": mframe, "map_anchor": manchor, "egomotion_frame": None, "sensor_rig_frame": None, "transform_required": None if coord_verified else True, "transform_source": transform_source, "coordinate_alignment_verified": coord_verified, "status": coord_status})

            obstacle_ready = obstacle_summary.get("status") == "OK" and obstacle_summary.get("field") is not None
            obstacle_rows.append({"clip_id": clip_id, "obstacle_row_count": obstacle_summary.get("row_count"), "valid_row_count": obstacle_summary.get("valid_row_count"), "invalid_row_count": obstacle_summary.get("invalid_row_count"), "unique_timestamp_count": obstacle_summary.get("unique_timestamp_count", obstacle_summary.get("unique_count")), "min_timestamp": obstacle_summary.get("min_timestamp", obstacle_min), "max_timestamp": obstacle_summary.get("max_timestamp", obstacle_max), "selected_timestamp_field": obstacle_summary.get("selected_timestamp_field"), "unique_track_count": obstacle_summary.get("unique_track_count"), "categories": obstacle_summary.get("categories"), "missing_timestamp_count": obstacle_summary.get("missing_timestamp_count"), "invalid_track_id_count": obstacle_summary.get("invalid_track_id_count"), "invalid_category_count": obstacle_summary.get("invalid_category_count"), "invalid_center_count": obstacle_summary.get("invalid_center_count"), "invalid_size_count": obstacle_summary.get("invalid_size_count"), "invalid_orientation_count": obstacle_summary.get("invalid_orientation_count"), "status": obstacle_summary.get("status")})
            time_verified = status in {"ALIGNED_DIRECT", "ALIGNED_BY_EXPLICIT_METADATA"} and prediction_t0_consistent
            query_grid_error = None
            if pred_t0 is not None:
                try:
                    grid_cf, grid_ttc, query_grid = _canonical_query_timestamps(pred_t0, evaluation_config)
                    cf_required, ttc_required = (grid_cf, grid_ttc) if time_verified else ([], [])
                except Exception as exc:
                    cf_required, ttc_required, query_grid = [], [], None
                    query_grid_error = f"{type(exc).__name__}: {exc}"
            else:
                cf_required, ttc_required, query_grid = [], [], None
            frame_evidence = _normalize_frame_evidence(context)
            completeness_verified = bool(context.get("obstacle_table_complete") is True or context.get("observation_completeness_verified") is True or isinstance(context.get("observation_contract"), dict) and context["observation_contract"].get("complete") is True)
            obstacle_timestamp_values = _parquet_timestamp_values(clip_dir / "clipgt/obstacle.parquet") if obstacle_ready else set()
            cf_counts = _coverage_counts(cf_required, frame_evidence, obstacle_timestamp_values, completeness_verified and obstacle_ready, 50_000) if time_verified else {"required": None, "observed": None, "empty": None, "unknown": None, "missing": None, "matched": None}
            ttc_counts = _coverage_counts(ttc_required, frame_evidence, obstacle_timestamp_values, completeness_verified and obstacle_ready, 100_000) if time_verified else {"required": None, "observed": None, "empty": None, "unknown": None, "missing": None, "matched": None}
            cf_ready = bool(time_verified and completeness_verified and obstacle_ready and cf_counts["missing"] == 0 and cf_counts["unknown"] == 0)
            ttc_ready = bool(time_verified and completeness_verified and obstacle_ready and ttc_counts["missing"] == 0 and ttc_counts["unknown"] == 0)
            if query_grid_error:
                observation_status = "QUERY_GRID_BUILD_ERROR"
            elif not time_verified:
                observation_status = "TIME_ALIGNMENT_UNRESOLVED"
            elif not frame_evidence:
                observation_status = "UNKNOWN"
            elif not completeness_verified:
                observation_status = "OBSERVATION_COMPLETENESS_UNVERIFIED"
            elif cf_ready and ttc_ready:
                observation_status = "COMPLETE"
            else:
                observation_status = "INCOMPLETE"
            empty_possible = bool((cf_counts["empty"] or 0) or (ttc_counts["empty"] or 0))
            coverage_rows.append({"clip_id": clip_id, "required_cf_start": cf_required[0] if cf_required else None, "required_cf_end": cf_required[-1] if cf_required else None, "required_ttc_end": ttc_required[-1] if ttc_required else None, "available_frame_count": len(frame_evidence) if frame_evidence else 0, "available_frame_min_ts": min(frame_evidence) if frame_evidence else None, "available_frame_max_ts": max(frame_evidence) if frame_evidence else None, "obstacle_timestamp_count": obstacle_summary.get("unique_timestamp_count", obstacle_summary.get("unique_count")), "confirmed_observed_empty_possible": empty_possible, "cf_required_frames": cf_counts["required"], "cf_observed_frames": cf_counts["observed"], "cf_empty_frames": cf_counts["empty"], "cf_unknown_frames": cf_counts["unknown"], "cf_missing_frames": cf_counts["missing"], "cf_ready": cf_ready, "ttc_required_queries": ttc_counts["required"], "ttc_observed_queries": ttc_counts["observed"], "ttc_empty_queries": ttc_counts["empty"], "ttc_unknown_queries": ttc_counts["unknown"], "ttc_missing_queries": ttc_counts["missing"], "ttc_ready": ttc_ready, "observation_contract_status": observation_status})
            map_summary = _map_status(clip_dir, engine["status"])
            map_rows.append({"clip_id": clip_id, **map_summary})
            official_rows.append({"clip_id": clip_id, **{f"{key}_available": (clip_dir / relative).is_file() for key, relative in OFFICIAL_FILES.items()}, "lane_direction_available": "lane.lane_direction" in inspect_parquet(clip_dir / "clipgt/lane.parquet", sample_rows=0).get("nested_field_paths", []), "lane_geometry_available": (clip_dir / "clipgt/lane.parquet").is_file()})
            ego_info = inspect_parquet(clip_dir / "clipgt/egomotion_estimate.parquet", sample_rows=0)
            transform_candidate = all((clip_dir / relative).is_file() for relative in ("rig_trajectories.json", "clipgt/calibration_estimate.parquet"))
            ego_rows.append({"clip_id": clip_id, "row_count": ego_summary.get("row_count"), "timestamp_min": ego_summary.get("min"), "timestamp_max": ego_summary.get("max"), "timestamp_spacing": ego_summary.get("median_dt"), "timestamp_field": ego_summary.get("selected_timestamp_field"), "position_fields": [field for field in ego_info.get("nested_field_paths", []) if "location." in field], "rotation_fields": [field for field in ego_info.get("nested_field_paths", []) if "orientation." in field], "frame": None, "parent_frame": None, "child_frame": None, "transform_metadata_available": transform_candidate, "transform_chain_verified": False, "transform_chain_available": False, "status": ego_summary.get("status")})
            blockers: List[str] = []
            if not pred: blockers.append("PREDICTION_MISSING")
            if not gt: blockers.append("GT_MISSING")
            if clip_id in prediction_identity_errors: blockers.append("PREDICTION_IDENTITY_INVALID")
            if clip_id in {key.split("|", 1)[0] for key in prediction_condition_duplicates}: blockers.append("PREDICTION_CONDITION_DUPLICATE")
            if len(prediction_t0_values) > 1: blockers.append("PREDICTION_T0_CONFLICT")
            if not prediction_frame_consistent or not prediction_anchor_consistent: blockers.append("PREDICTION_COORDINATE_METADATA_CONFLICT")
            if clip_id in ground_truth_duplicates: blockers.append("GT_DUPLICATE")
            if clip_id in context_duplicates: blockers.append("CONTEXT_DUPLICATE")
            if status not in {"ALIGNED_DIRECT", "ALIGNED_BY_EXPLICIT_METADATA"}: blockers.append("TIME_ALIGNMENT_UNRESOLVED")
            if not coord_verified: blockers.append("COORDINATE_UNRESOLVED")
            if not obstacle_ready: blockers.append("OBSTACLE_SCHEMA_INVALID")
            if query_grid_error: blockers.append("QUERY_GRID_CONTRACT_UNRESOLVED")
            query_grid_verified = bool(query_grid and query_grid.get("verified") is True and not query_grid_error)
            if not query_grid_error and not query_grid_verified: blockers.append("QUERY_GRID_CONFIG_UNVERIFIED")
            if not cf_ready or not ttc_ready: blockers.append("OBSERVATION_COVERAGE_INCOMPLETE")
            if not map_summary.get("dac_geometry_verified", False):
                if map_summary.get("status") == "FILE_NOT_FOUND":
                    blockers.append("MAP_MISSING")
                elif map_summary.get("status") == "PARQUET_ENGINE_UNAVAILABLE":
                    blockers.append("PARQUET_ENGINE_UNAVAILABLE")
                elif map_summary.get("dac_candidate_available"):
                    blockers.append("DAC_GEOMETRY_UNVERIFIED")
                else:
                    blockers.append("MAP_INVALID")
            elif map_summary["status"] not in {"OK"}:
                if map_summary["status"] == "FILE_NOT_FOUND":
                    blockers.append("MAP_MISSING")
                elif map_summary["status"] == "PARQUET_ENGINE_UNAVAILABLE":
                    blockers.append("PARQUET_ENGINE_UNAVAILABLE")
                else:
                    blockers.append("MAP_INVALID")
            contracts[clip_id] = {"clip_id": clip_id, "prediction": {"raw_condition_count": condition_stats["raw_condition_count"], "unique_condition_count": condition_stats["unique_condition_count"], "duplicate_condition_count": condition_stats["duplicate_condition_count"], "modes": prediction_modes, "alphas": prediction_alphas, "t0_values": prediction_t0_values}, "time": {"t0_us": pred_t0, "source": "prediction_jsonl", "verified": time_verified}, "query_grid": query_grid or {"source": "UNRESOLVED", "verified": False, "error": query_grid_error}, "coordinate": {"prediction_frame": pframe, "prediction_frame_values": prediction_frame_values, "prediction_anchor": panchor, "prediction_anchor_values": prediction_anchor_values, "gt_frame": gframe, "gt_anchor": ganchor, "obstacle_frame": cframe, "obstacle_anchor": oanchor, "map_frame": mframe, "map_anchor": manchor, "transform_required": not coord_verified, "transform_source": transform_source, "transform_metadata_available": transform_available, "transform_chain_verified": False, "verified": coord_verified}, "observation": {"cf_ready": cf_ready, "ttc_ready": ttc_ready, "coverage_source": "independent_frame_evidence" if frame_evidence else None, "status": observation_status, "cf_counts": cf_counts, "ttc_counts": ttc_counts}, "map": {"ready": bool(map_summary.get("dac_geometry_verified", False)), "source": "clipgt", "status": map_summary["status"], "geometry_available": map_summary.get("geometry_available"), "dac_candidate_available": map_summary.get("dac_candidate_available"), "dac_geometry_verified": map_summary.get("dac_geometry_verified"), "recommended_dac_source": map_summary.get("recommended_dac_source")}, "ready_for_proxy": bool(pred_rows and gt and time_verified and query_grid_verified and coord_verified and obstacle_ready and cf_ready and ttc_ready and map_summary.get("dac_geometry_verified", False) and not blockers), "blockers": sorted(set(blockers))}
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
    summary = {"total_clips": len(clips), "prediction_ready": sum(bool(predictions_by_clip.get(cid)) and cid not in prediction_identity_errors and cid not in {key.split("|", 1)[0] for key in prediction_condition_duplicates} for cid in clips), "gt_ready": sum(bool(ground_truth.get(cid)) and cid not in ground_truth_duplicates for cid in clips), "time_ready": sum(c["time"]["verified"] for c in contracts.values()), "coordinate_ready": sum(c["coordinate"]["verified"] for c in contracts.values()), "observation_cf_ready": sum(c["observation"]["cf_ready"] for c in contracts.values()), "observation_ttc_ready": sum(c["observation"]["ttc_ready"] for c in contracts.values()), "map_ready": sum(c["map"]["ready"] for c in contracts.values()), "proxy_ready_clip_count": len(eligible), "eligible_clips": eligible, "blocked_clips": blocked, "duplicate_clip_ids": duplicate_ids, "prediction_condition_duplicate_keys": prediction_condition_duplicates, "ground_truth_duplicate_clip_ids": ground_truth_duplicates, "context_duplicate_clip_ids": context_duplicates, "audit_error_count": len(errors), "parquet_engine": engine, "status": "DATASET_READY" if not blocked and not errors else "DATASET_NOT_READY"}
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
