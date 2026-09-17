"""Map geometry loader and inspection for NuRec filtered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def inspect_clip_map_status(filtered_dir: Path, clip_id: str) -> Dict[str, Any]:
    """Inspects drivable map files and returns structured status for a clip.
    
    Status categories:
      - 'OK': All present map files read cleanly with zero errors, and >= 1 valid drivable polygon.
      - 'PARTIAL': At least one valid polygon loaded, but other rows/files had read, schema, or geometry errors.
      - 'FILE_NOT_FOUND': Neither lane.parquet nor intersection_area.parquet exists.
      - 'NO_DRIVABLE_POLYGON': Files exist and read cleanly, but contain 0 drivable polygons.
      - 'PARQUET_READ_ERROR': Exception raised when reading parquet file, 0 polygons loaded.
      - 'UNSUPPORTED_SCHEMA': Parquet file lacks expected columns or structure, 0 polygons loaded.
      - 'INVALID_GEOMETRY': Polygons contain NaN, Inf, or invalid coordinate values, 0 polygons loaded.
    """
    clip_dir = filtered_dir / clip_id / "clipgt"
    if not clip_dir.is_dir():
        return {
            "clip_id": clip_id,
            "status": "FILE_NOT_FOUND",
            "lane_polygon_count": 0,
            "intersection_polygon_count": 0,
            "total_polygons": 0,
            "valid_polygon_count": 0,
            "invalid_polygon_count": 0,
            "skipped_row_count": 0,
            "read_errors": [],
            "schema_errors": [],
            "geometry_errors": [],
            "usable_for_strict_scoring": False,
            "detail": f"clipgt directory does not exist: {clip_dir}",
            "lane_polygons": [],
            "intersection_polygons": [],
        }

    try:
        import pandas as pd
    except ImportError as e:
        return {
            "clip_id": clip_id,
            "status": "PARQUET_READ_ERROR",
            "lane_polygon_count": 0,
            "intersection_polygon_count": 0,
            "total_polygons": 0,
            "valid_polygon_count": 0,
            "invalid_polygon_count": 0,
            "skipped_row_count": 0,
            "read_errors": [f"pandas import failed: {e}"],
            "schema_errors": [],
            "geometry_errors": [],
            "usable_for_strict_scoring": False,
            "detail": f"pandas import failed: {e}",
            "lane_polygons": [],
            "intersection_polygons": [],
        }

    lane_pq = clip_dir / "lane.parquet"
    ia_pq = clip_dir / "intersection_area.parquet"

    if not lane_pq.is_file() and not ia_pq.is_file():
        return {
            "clip_id": clip_id,
            "status": "FILE_NOT_FOUND",
            "lane_polygon_count": 0,
            "intersection_polygon_count": 0,
            "total_polygons": 0,
            "valid_polygon_count": 0,
            "invalid_polygon_count": 0,
            "skipped_row_count": 0,
            "read_errors": [],
            "schema_errors": [],
            "geometry_errors": [],
            "usable_for_strict_scoring": False,
            "detail": "Neither lane.parquet nor intersection_area.parquet exists in clipgt",
            "lane_polygons": [],
            "intersection_polygons": [],
        }

    lane_polys: List[np.ndarray] = []
    ia_polys: List[np.ndarray] = []
    read_errors: List[str] = []
    schema_errors: List[str] = []
    geometry_errors: List[str] = []
    invalid_polygon_count = 0
    skipped_row_count = 0

    # 1. Inspect lane.parquet
    if lane_pq.is_file():
        try:
            df = pd.read_parquet(lane_pq)
            if "lane" not in df.columns:
                schema_errors.append(f"lane.parquet missing 'lane' column (columns={list(df.columns)})")
            else:
                for row_idx, row in df.iterrows():
                    lane_data = row.get("lane")
                    if not isinstance(lane_data, dict):
                        skipped_row_count += 1
                        schema_errors.append(f"lane row {row_idx} is not dict: {type(lane_data)}")
                        continue

                    left = lane_data.get("left_rail")
                    right = lane_data.get("right_rail")
                    if left is None or right is None or len(left) < 2 or len(right) < 2:
                        skipped_row_count += 1
                        continue

                    pts: List[List[float]] = []
                    has_invalid_pt = False

                    for rail_name, rail_pts in [("left_rail", left), ("right_rail", reversed(right))]:
                        for pt_idx, p in enumerate(rail_pts):
                            if not isinstance(p, dict) or "x" not in p or "y" not in p:
                                has_invalid_pt = True
                                geometry_errors.append(f"lane row {row_idx} {rail_name}[{pt_idx}] missing x/y")
                                break
                            try:
                                px = float(p["x"])
                                py = float(p["y"])
                                if not (np.isfinite(px) and np.isfinite(py)):
                                    has_invalid_pt = True
                                    geometry_errors.append(f"lane row {row_idx} {rail_name}[{pt_idx}] non-finite ({px}, {py})")
                                    break
                                pts.append([px, py])
                            except (TypeError, ValueError) as ex:
                                has_invalid_pt = True
                                geometry_errors.append(f"lane row {row_idx} {rail_name}[{pt_idx}] non-numeric: {ex}")
                                break
                        if has_invalid_pt:
                            break

                    if has_invalid_pt:
                        invalid_polygon_count += 1
                    elif len(pts) >= 3:
                        poly = np.array(pts, dtype=float)
                        lane_polys.append(poly)
                    else:
                        invalid_polygon_count += 1
                        geometry_errors.append(f"lane row {row_idx} produced fewer than 3 vertices")
        except Exception as ex:
            read_errors.append(f"Failed to read lane.parquet: {ex}")

    # 2. Inspect intersection_area.parquet
    if ia_pq.is_file():
        try:
            df = pd.read_parquet(ia_pq)
            if "intersection_area" not in df.columns:
                schema_errors.append(f"intersection_area.parquet missing 'intersection_area' column (columns={list(df.columns)})")
            else:
                for row_idx, row in df.iterrows():
                    ia_data = row.get("intersection_area")
                    if not isinstance(ia_data, dict):
                        skipped_row_count += 1
                        schema_errors.append(f"intersection_area row {row_idx} is not dict: {type(ia_data)}")
                        continue

                    loc = ia_data.get("location")
                    if loc is None or len(loc) == 0:
                        skipped_row_count += 1
                        continue

                    if len(loc) < 3:
                        invalid_polygon_count += 1
                        geometry_errors.append(f"intersection_area row {row_idx} has fewer than 3 vertices ({len(loc)})")
                        continue

                    pts = []
                    has_invalid_pt = False
                    for pt_idx, p in enumerate(loc):
                        if not isinstance(p, dict) or "x" not in p or "y" not in p:
                            has_invalid_pt = True
                            geometry_errors.append(f"intersection_area row {row_idx}[{pt_idx}] missing x/y")
                            break
                        try:
                            px = float(p["x"])
                            py = float(p["y"])
                            if not (np.isfinite(px) and np.isfinite(py)):
                                has_invalid_pt = True
                                geometry_errors.append(f"intersection_area row {row_idx}[{pt_idx}] non-finite ({px}, {py})")
                                break
                            pts.append([px, py])
                        except (TypeError, ValueError) as ex:
                            has_invalid_pt = True
                            geometry_errors.append(f"intersection_area row {row_idx}[{pt_idx}] non-numeric: {ex}")
                            break

                    if has_invalid_pt:
                        invalid_polygon_count += 1
                    elif len(pts) >= 3:
                        poly = np.array(pts, dtype=float)
                        ia_polys.append(poly)
                    else:
                        invalid_polygon_count += 1
        except Exception as ex:
            read_errors.append(f"Failed to read intersection_area.parquet: {ex}")

    total_polys = len(lane_polys) + len(ia_polys)
    has_issues = bool(read_errors or schema_errors or geometry_errors or invalid_polygon_count > 0)

    if total_polys > 0:
        if has_issues:
            status = "PARTIAL"
            usable_strict = False
            error_details = []
            if read_errors: error_details.append(f"read_errors={len(read_errors)}")
            if schema_errors: error_details.append(f"schema_errors={len(schema_errors)}")
            if geometry_errors: error_details.append(f"geometry_errors={len(geometry_errors)}")
            if invalid_polygon_count > 0: error_details.append(f"invalid_polys={invalid_polygon_count}")
            detail = f"PARTIAL: {total_polys} valid polygons loaded, but issues detected ({', '.join(error_details)})"
        else:
            status = "OK"
            usable_strict = True
            detail = f"Loaded {len(lane_polys)} lane and {len(ia_polys)} intersection polygons"
    elif read_errors:
        status = "PARQUET_READ_ERROR"
        usable_strict = False
        detail = "; ".join(read_errors)
    elif schema_errors:
        status = "UNSUPPORTED_SCHEMA"
        usable_strict = False
        detail = "; ".join(schema_errors)
    elif geometry_errors:
        status = "INVALID_GEOMETRY"
        usable_strict = False
        detail = "; ".join(geometry_errors)
    else:
        status = "NO_DRIVABLE_POLYGON"
        usable_strict = False
        detail = "Files exist but contain 0 drivable polygons (empty location/rails)"

    return {
        "clip_id": clip_id,
        "status": status,
        "lane_polygon_count": len(lane_polys),
        "intersection_polygon_count": len(ia_polys),
        "total_polygons": total_polys,
        "valid_polygon_count": total_polys,
        "invalid_polygon_count": invalid_polygon_count,
        "skipped_row_count": skipped_row_count,
        "read_errors": read_errors,
        "schema_errors": schema_errors,
        "geometry_errors": geometry_errors,
        "usable_for_strict_scoring": usable_strict,
        "detail": detail,
        "lane_polygons": lane_polys,
        "intersection_polygons": ia_polys,
    }


def load_lane_polygons_for_clip(
    filtered_dir: Path,
    clip_id: str,
    strict_mode: bool = True,
) -> List[np.ndarray]:
    """Loads drivable lane and intersection polygons for a clip.
    
    In strict_mode: only returns polygons if status == 'OK' (usable_for_strict_scoring is True).
    If status is PARTIAL or error in strict_mode, returns empty list so strict evaluation rejects partial/corrupted maps.
    In diagnostic mode (strict_mode=False): returns whatever valid polygons were recovered.
    """
    res = inspect_clip_map_status(filtered_dir, clip_id)
    all_polys = res.get("lane_polygons", []) + res.get("intersection_polygons", [])
    if strict_mode:
        if res.get("usable_for_strict_scoring", False):
            return all_polys
        return []
    return all_polys
