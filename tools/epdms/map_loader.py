"""Map geometry loader and inspection for NuRec filtered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def inspect_clip_map_status(filtered_dir: Path, clip_id: str) -> Dict[str, Any]:
    """Inspects drivable map files and returns structured status for a clip.
    
    Status categories:
      - 'OK': At least one valid drivable polygon loaded.
      - 'FILE_NOT_FOUND': Neither lane.parquet nor intersection_area.parquet exists.
      - 'NO_DRIVABLE_POLYGON': Files exist but contain 0 drivable polygons (e.g. empty location array).
      - 'PARQUET_READ_ERROR': Exception raised when reading parquet file.
      - 'UNSUPPORTED_SCHEMA': Parquet file lacks expected columns or structure.
      - 'INVALID_GEOMETRY': Polygons contain NaN, Inf, or invalid coordinate values.
    """
    clip_dir = filtered_dir / clip_id / "clipgt"
    if not clip_dir.is_dir():
        return {
            "clip_id": clip_id,
            "status": "FILE_NOT_FOUND",
            "lane_polygon_count": 0,
            "intersection_polygon_count": 0,
            "total_polygons": 0,
            "detail": f"clipgt directory does not exist: {clip_dir}",
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
            "detail": f"pandas import failed: {e}",
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
            "detail": "Neither lane.parquet nor intersection_area.parquet exists in clipgt",
        }

    lane_polys: List[np.ndarray] = []
    ia_polys: List[np.ndarray] = []
    read_errors: List[str] = []
    schema_errors: List[str] = []
    geometry_errors: List[str] = []

    # 1. Inspect lane.parquet
    if lane_pq.is_file():
        try:
            df = pd.read_parquet(lane_pq)
            if "lane" not in df.columns:
                schema_errors.append(f"lane.parquet missing 'lane' column (columns={list(df.columns)})")
            else:
                for _, row in df.iterrows():
                    lane_data = row.get("lane")
                    if isinstance(lane_data, dict):
                        left = lane_data.get("left_rail")
                        right = lane_data.get("right_rail")
                        if left is not None and right is not None and len(left) > 1 and len(right) > 1:
                            try:
                                pts_left = [[float(p["x"]), float(p["y"])] for p in left if "x" in p and "y" in p]
                                pts_right = [[float(p["x"]), float(p["y"])] for p in reversed(right) if "x" in p and "y" in p]
                                poly = np.array(pts_left + pts_right, dtype=float)
                                if not np.all(np.isfinite(poly)):
                                    geometry_errors.append("Non-finite coordinates found in lane polygon")
                                elif len(poly) >= 3:
                                    lane_polys.append(poly)
                            except Exception as ex:
                                geometry_errors.append(f"Failed to parse lane vertices: {ex}")
        except Exception as ex:
            read_errors.append(f"Failed to read lane.parquet: {ex}")

    # 2. Inspect intersection_area.parquet
    if ia_pq.is_file():
        try:
            df = pd.read_parquet(ia_pq)
            if "intersection_area" not in df.columns:
                schema_errors.append(f"intersection_area.parquet missing 'intersection_area' column (columns={list(df.columns)})")
            else:
                for _, row in df.iterrows():
                    ia_data = row.get("intersection_area")
                    if isinstance(ia_data, dict):
                        loc = ia_data.get("location")
                        if loc is not None:
                            try:
                                if len(loc) >= 3:
                                    poly = np.array([[float(p["x"]), float(p["y"])] for p in loc if "x" in p and "y" in p], dtype=float)
                                    if not np.all(np.isfinite(poly)):
                                        geometry_errors.append("Non-finite coordinates found in intersection area")
                                    elif len(poly) >= 3:
                                        ia_polys.append(poly)
                            except Exception as ex:
                                geometry_errors.append(f"Failed to parse intersection area vertices: {ex}")
        except Exception as ex:
            read_errors.append(f"Failed to read intersection_area.parquet: {ex}")

    total_polys = len(lane_polys) + len(ia_polys)

    if total_polys > 0:
        status = "OK"
        detail = f"Loaded {len(lane_polys)} lane and {len(ia_polys)} intersection polygons"
    elif read_errors:
        status = "PARQUET_READ_ERROR"
        detail = "; ".join(read_errors)
    elif schema_errors:
        status = "UNSUPPORTED_SCHEMA"
        detail = "; ".join(schema_errors)
    elif geometry_errors:
        status = "INVALID_GEOMETRY"
        detail = "; ".join(geometry_errors)
    else:
        status = "NO_DRIVABLE_POLYGON"
        detail = "Files exist but contain 0 drivable polygons (empty location/rails)"

    return {
        "clip_id": clip_id,
        "status": status,
        "lane_polygon_count": len(lane_polys),
        "intersection_polygon_count": len(ia_polys),
        "total_polygons": total_polys,
        "detail": detail,
        "lane_polygons": lane_polys,
        "intersection_polygons": ia_polys,
    }


def load_lane_polygons_for_clip(filtered_dir: Path, clip_id: str) -> List[np.ndarray]:
    """Loads drivable lane and intersection polygons for a clip if available."""
    res = inspect_clip_map_status(filtered_dir, clip_id)
    if res["status"] == "OK":
        return res.get("lane_polygons", []) + res.get("intersection_polygons", [])
    return []
