"""Lane-geometry diagnostics for the partial NuRec EPDMS vector.

This module is intentionally separate from the frozen safety proxy.  It reads
NuRec lane rails, derives deterministic centerlines, and returns proxy/diagnostic
results.  It never populates official NAVSIM fields and never fits geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .geometry_numpy import point_in_polygon_ray_casting, point_to_segment_dist
from .map_transform import interpolate_rig_world_pose, invert_se3, load_rig_world_samples


LK_READY = "READY"
LK_MISSING = "MISSING_LANE_GEOMETRY"
LK_AMBIGUOUS = "LANE_ASSOCIATION_AMBIGUOUS"
LK_MAP_UNREADY = "MAP_TRANSFORM_NOT_READY"
LK_INVALID = "INVALID_GEOMETRY"
LK_INPUT = "INPUT_ERROR"


@dataclass(frozen=True)
class LaneCenterline:
    lane_id: str
    centerline: np.ndarray
    corridor: np.ndarray
    lane_direction: Optional[str]
    source_row: int


def _as_xy(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    values: list[list[float]] = []
    try:
        for point in list(points):
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                return None
            x, y = float(point["x"]), float(point["y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            values.append([x, y])
    except (TypeError, ValueError):
        return None
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    keep = np.ones(len(array), dtype=bool)
    if len(array) > 1:
        keep[1:] = np.linalg.norm(np.diff(array, axis=0), axis=1) > 1e-7
    array = array[keep]
    return array if len(array) >= 2 else None


def _resample_polyline(points: np.ndarray, sample_spacing_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total = float(distances[-1])
    if total <= 1e-7:
        return points[:1].copy()
    count = max(2, int(math.ceil(total / max(sample_spacing_m, 0.1))) + 1)
    targets = np.linspace(0.0, total, count)
    return np.column_stack([
        np.interp(targets, distances, points[:, 0]),
        np.interp(targets, distances, points[:, 1]),
    ])


def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if len(points) == 1 or distances[-1] <= 1e-7:
        return np.repeat(points[:1], max(1, count), axis=0)
    targets = np.linspace(0.0, float(distances[-1]), max(2, count))
    return np.column_stack([
        np.interp(targets, distances, points[:, 0]),
        np.interp(targets, distances, points[:, 1]),
    ])


def derive_lane_centerline(
    left_rail: Any,
    right_rail: Any,
    *,
    sample_spacing_m: float = 1.0,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(centerline, corridor_polygon)`` after arclength resampling."""

    left = _as_xy(left_rail)
    right = _as_xy(right_rail)
    if left is None or right is None:
        return None
    direct = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))
    reversed_pair = float(np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0]))
    if reversed_pair < direct:
        right = right[::-1].copy()
    left_r = _resample_polyline(left, sample_spacing_m)
    right_r = _resample_polyline(right, sample_spacing_m)
    count = max(len(left_r), len(right_r))
    left_r = _resample_to_count(left_r, count)
    right_r = _resample_to_count(right_r, count)
    center = 0.5 * (left_r + right_r)
    corridor = np.vstack([left_r, right_r[::-1]])
    if len(center) < 2 or len(corridor) < 3 or not np.isfinite(center).all() or not np.isfinite(corridor).all():
        return None
    return center, corridor


def _transform_xy(points: np.ndarray, world_to_ego: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points)), np.ones(len(points))])
    return (world_to_ego @ homogeneous.T).T[:, :2]


def _load_intersections(clip_dir: Path, world_to_ego: np.ndarray) -> list[np.ndarray]:
    path = clip_dir / "clipgt" / "intersection_area.parquet"
    if not path.is_file():
        return []
    try:
        frame = pd.read_parquet(path, columns=["intersection_area"])
    except Exception:
        return []
    result: list[np.ndarray] = []
    for value in frame["intersection_area"].tolist():
        if not isinstance(value, dict):
            continue
        points = _as_xy(value.get("location"))
        if points is not None and len(points) >= 3:
            result.append(_transform_xy(points, world_to_ego))
    return result


def load_lane_centerlines_for_clip(
    map_root: Path,
    clip_id: str,
    nurec_t0_us: int,
    *,
    sample_spacing_m: float = 1.0,
) -> dict[str, Any]:
    """Read and transform lane rails to ``EGO_AT_T0`` without modifying raw data."""

    clip_dir = map_root / clip_id
    lane_path = clip_dir / "clipgt" / "lane.parquet"
    if not lane_path.is_file():
        return {"status": LK_MISSING, "lanes": [], "intersections": [], "world_to_ego": None}
    try:
        samples = load_rig_world_samples(clip_dir)
        pose = interpolate_rig_world_pose(samples, int(nurec_t0_us))
        world_to_ego = invert_se3(pose)
        frame = pd.read_parquet(lane_path, columns=["key", "lane"])
    except Exception as exc:
        return {"status": LK_MAP_UNREADY, "lanes": [], "intersections": [], "error": str(exc)}

    lanes: list[LaneCenterline] = []
    for row_index, row in frame.iterrows():
        lane = row.get("lane")
        if not isinstance(lane, dict):
            continue
        derived = derive_lane_centerline(lane.get("left_rail"), lane.get("right_rail"), sample_spacing_m=sample_spacing_m)
        if derived is None:
            continue
        center, corridor = derived
        key = row.get("key") if isinstance(row.get("key"), dict) else {}
        map_id = str(key.get("map_id") or "unknown_map")
        lane_id = f"{map_id}:row_{int(row_index)}"
        lanes.append(LaneCenterline(
            lane_id=lane_id,
            centerline=_transform_xy(center, world_to_ego),
            corridor=_transform_xy(corridor, world_to_ego),
            lane_direction=str(lane.get("lane_direction")) if lane.get("lane_direction") is not None else None,
            source_row=int(row_index),
        ))
    intersections = _load_intersections(clip_dir, world_to_ego)
    status = LK_READY if lanes else LK_MISSING
    return {
        "status": status,
        "lanes": lanes,
        "intersections": intersections,
        "world_to_ego": world_to_ego.tolist(),
        "lane_count": len(lanes),
        "intersection_count": len(intersections),
    }


def _polyline_distance(point: np.ndarray, line: np.ndarray) -> tuple[float, int]:
    starts = line[:-1]
    ends = line[1:]
    deltas = ends - starts
    denom = np.einsum("ij,ij->i", deltas, deltas)
    factors = np.divide(
        np.einsum("ij,ij->i", np.broadcast_to(point, starts.shape) - starts, deltas),
        denom,
        out=np.zeros(len(starts), dtype=float),
        where=denom > 1e-12,
    )
    factors = np.clip(factors, 0.0, 1.0)
    projections = starts + factors[:, None] * deltas
    distances = np.linalg.norm(projections - point, axis=1)
    if len(distances) == 0:
        return float("inf"), 0
    index = int(np.argmin(distances))
    return float(distances[index]), index


def _lane_contains(point: np.ndarray, lane: LaneCenterline) -> bool:
    return point_in_polygon_ray_casting(float(point[0]), float(point[1]), lane.corridor)


def _select_lane(point: np.ndarray, lanes: Sequence[LaneCenterline], current: Optional[LaneCenterline], max_distance_m: float) -> tuple[Optional[LaneCenterline], str]:
    if current is not None and _lane_contains(point, current):
        return current, "CURRENT_CONTAINMENT"
    containing = [lane for lane in lanes if _lane_contains(point, lane)]
    if len(containing) == 1:
        return containing[0], "UNIQUE_CONTAINMENT"
    if len(containing) > 1:
        return None, "AMBIGUOUS_CONTAINMENT"
    distances = sorted([(_polyline_distance(point, lane.centerline)[0], lane) for lane in lanes], key=lambda item: item[0])
    if not distances or distances[0][0] > max_distance_m:
        return None, "NO_PLAUSIBLE_LANE"
    if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) < 0.25:
        return None, "AMBIGUOUS_NEAREST_LANE"
    return distances[0][1], "NEAREST_PLAUSIBLE_LANE"


def _inside_any(point: np.ndarray, polygons: Sequence[np.ndarray]) -> bool:
    return any(point_in_polygon_ray_casting(float(point[0]), float(point[1]), poly) for poly in polygons)


def _line_heading(line: np.ndarray, segment_index: int) -> float:
    index = min(max(int(segment_index), 0), len(line) - 2)
    delta = line[index + 1] - line[index]
    return float(math.atan2(delta[1], delta[0]))


def compute_lk_proxy(
    trajectory_xy: np.ndarray,
    lane_bundle: dict[str, Any],
    *,
    dt_s: float = 0.1,
    lateral_deviation_limit_m: float = 0.5,
    continuous_violation_window_s: float = 2.0,
    lane_association_max_distance_m: float = 4.0,
) -> dict[str, Any]:
    """Compute a lane-centerline proxy; ambiguous evidence fails closed."""

    points = np.asarray(trajectory_xy, dtype=float)
    lanes = lane_bundle.get("lanes", [])
    intersections = lane_bundle.get("intersections", [])
    if len(points) == 0 or not lanes or not np.isfinite(points).all():
        return {"lk_proxy": None, "status": LK_MISSING, "associated_lane_ids": []}
    current: Optional[LaneCenterline] = None
    distances: list[float] = []
    excluded = 0
    violation_frames = 0
    max_run = 0
    run = 0
    lane_ids: list[str] = []
    lane_headings: list[Optional[float]] = []
    for point in points:
        if _inside_any(point, intersections):
            excluded += 1
            run = 0
            lane_headings.append(None)
            continue
        current, selection_status = _select_lane(point, lanes, current, lane_association_max_distance_m)
        if current is None:
            return {
                "lk_proxy": None,
                "status": LK_AMBIGUOUS,
                "associated_lane_ids": list(dict.fromkeys(lane_ids)),
                "intersection_excluded_frames": excluded,
            }
        distance, segment = _polyline_distance(point, current.centerline)
        distances.append(distance)
        lane_ids.append(current.lane_id)
        lane_headings.append(_line_heading(current.centerline, segment))
        violating = distance > lateral_deviation_limit_m
        if violating:
            violation_frames += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    max_continuous_s = float(max_run * dt_s)
    score = 0.0 if max_continuous_s >= continuous_violation_window_s else 1.0
    return {
        "lk_proxy": score,
        "status": LK_READY,
        "max_lateral_deviation_m": max(distances) if distances else None,
        "violation_frame_count": violation_frames,
        "max_continuous_violation_s": max_continuous_s,
        "associated_lane_ids": list(dict.fromkeys(lane_ids)),
        "intersection_excluded_frames": excluded,
        "lane_headings_rad": lane_headings,
    }


def compute_lane_direction_alignment_diagnostic(
    trajectory_xy: np.ndarray,
    lk_result: dict[str, Any],
    *,
    min_motion_distance_m: float = 0.05,
) -> dict[str, Any]:
    """Compare motion direction with the selected lane tangent only diagnostically."""

    points = np.asarray(trajectory_xy, dtype=float)
    lane_headings = lk_result.get("lane_headings_rad", [])
    aligned = 0
    compared = 0
    opposite_distance = 0.0
    misalignments: list[float] = []
    for index in range(1, min(len(points), len(lane_headings))):
        lane_heading = lane_headings[index]
        delta = points[index] - points[index - 1]
        motion = float(np.linalg.norm(delta))
        if lane_heading is None or motion < min_motion_distance_m:
            continue
        motion_heading = math.atan2(float(delta[1]), float(delta[0]))
        diff = abs((motion_heading - lane_heading + math.pi) % (2.0 * math.pi) - math.pi)
        misalignments.append(math.degrees(diff))
        compared += 1
        if diff <= math.pi / 2.0:
            aligned += 1
        else:
            opposite_distance += motion
    return {
        "lane_direction_alignment_fraction": (aligned / compared) if compared else None,
        "max_heading_misalignment_deg": max(misalignments) if misalignments else None,
        "opposite_heading_frame_count": (compared - aligned) if compared else 0,
        "opposite_motion_distance_m": opposite_distance,
        "status": "DIAGNOSTIC_ONLY_NO_LEGAL_DIRECTION_PROOF",
    }


def audit_lane_direction_contract(map_root: Path, clip_ids: Iterable[str]) -> dict[str, Any]:
    """Audit field availability without treating ``lane_direction`` as traffic law."""

    field_counts: dict[str, int] = {}
    value_counts: dict[str, int] = {}
    inspected = 0
    for clip_id in clip_ids:
        path = map_root / str(clip_id) / "clipgt" / "lane.parquet"
        if not path.is_file():
            continue
        inspected += 1
        try:
            frame = pd.read_parquet(path, columns=["lane"])
        except Exception:
            continue
        for value in frame["lane"].tolist():
            if not isinstance(value, dict):
                continue
            for key in value:
                field_counts[key] = field_counts.get(key, 0) + 1
            direction = value.get("lane_direction")
            if direction is not None:
                name = str(direction)
                value_counts[name] = value_counts.get(name, 0) + 1
    explicit_travel_direction = any(key in field_counts for key in ("travel_direction", "allowed_direction", "direction_of_travel"))
    status = "LANE_DIRECTION_VERIFIED" if explicit_travel_direction else "LANE_DIRECTION_UNRESOLVED"
    return {
        "status": status,
        "inspected_clip_count": inspected,
        "field_counts": field_counts,
        "lane_direction_values": value_counts,
        "evidence": "lane_direction is present but is not treated as legal travel direction without an explicit source contract" if "lane_direction" in field_counts else "no explicit direction field found",
    }
