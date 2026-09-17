"""Map geometry loader for NuRec filtered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import numpy as np


def load_lane_polygons_for_clip(filtered_dir: Path, clip_id: str) -> List[np.ndarray]:
    """Loads drivable lane and intersection polygons for a clip if available."""
    clip_dir = filtered_dir / clip_id / "clipgt"
    polygons: List[np.ndarray] = []
    if not clip_dir.is_dir():
        return polygons

    try:
        import pandas as pd
    except ImportError:
        return polygons

    # 1. Check lane.parquet
    lane_pq = clip_dir / "lane.parquet"
    if lane_pq.is_file():
        try:
            df = pd.read_parquet(lane_pq)
            for _, row in df.iterrows():
                lane_data = row.get("lane", {})
                if isinstance(lane_data, dict):
                    left = lane_data.get("left_rail")
                    right = lane_data.get("right_rail")
                    if left is not None and right is not None and len(left) > 1 and len(right) > 1:
                        pts_left = [[float(p["x"]), float(p["y"])] for p in left if "x" in p and "y" in p]
                        pts_right = [[float(p["x"]), float(p["y"])] for p in reversed(right) if "x" in p and "y" in p]
                        poly = np.array(pts_left + pts_right, dtype=float)
                        if len(poly) >= 3 and np.all(np.isfinite(poly)):
                            polygons.append(poly)
        except Exception:
            pass

    # 2. Check intersection_area.parquet
    ia_pq = clip_dir / "intersection_area.parquet"
    if ia_pq.is_file():
        try:
            df = pd.read_parquet(ia_pq)
            for _, row in df.iterrows():
                ia_data = row.get("intersection_area", {})
                if isinstance(ia_data, dict):
                    loc = ia_data.get("location")
                    if loc is not None and len(loc) >= 3:
                        poly = np.array([[float(p["x"]), float(p["y"])] for p in loc if "x" in p and "y" in p], dtype=float)
                        if len(poly) >= 3 and np.all(np.isfinite(poly)):
                            polygons.append(poly)
        except Exception:
            pass

    return polygons
