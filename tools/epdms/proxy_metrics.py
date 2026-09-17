"""Implementation of NuRec Safety Proxy v1 metrics with strict required data contracts."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .coordinates import derive_heading_from_xy
from .geometry_numpy import (
    get_ego_box_corners,
    get_oriented_box_corners,
    point_in_polygon_ray_casting,
    points_in_any_polygon,
    project_point_onto_polyline,
    sat_box_intersection,
)
from .kinematics_numpy import ComfortThresholds, KinematicProfile, compute_kinematics
from .schemas import VehicleParameters


class CorruptedObservationDataError(ValueError):
    """Raised when an obstacle observation contains corrupted, missing, or non-finite geometry."""
    pass


def normalize_obstacle_record(obs: Dict[str, Any], raise_on_corrupt: bool = False) -> Optional[Dict[str, Any]]:
    """Normalizes an obstacle record from either flat or nested schema into a standard format.
    
    Supports:
      - Flat schema: { "center": ..., "size": ..., "orientation": ..., "timestamp_micros": ..., ... }
      - Nested schema: { "obstacle": { "center": ..., "size": ..., "orientation": ... }, "timestamp_micros": ..., ... }
      - Or wrapper where timestamp_micros is inside "key": { "key": { "timestamp_micros": ... }, "obstacle": ... }
      
    Returns None if required geometric attributes are missing or non-finite (or raises if raise_on_corrupt=True).
    """
    if not isinstance(obs, dict):
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle is not a dict: {obs}")
        return None

    data = obs.get("obstacle") if isinstance(obs.get("obstacle"), dict) else obs

    # 1. Extract timestamp_micros
    ts = obs.get("timestamp_micros")
    if ts is None and "key" in obs and isinstance(obs["key"], dict):
        ts = obs["key"].get("timestamp_micros")
    if ts is None and "timestamp_micros" in data:
        ts = data.get("timestamp_micros")

    # 2. Extract Center
    center = data.get("center")
    if not isinstance(center, dict) or "x" not in center or "y" not in center:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle missing center coordinates: {center}")
        return None
    try:
        ox = float(center["x"])
        oy = float(center["y"])
        oz = float(center.get("z", 0.0))
        if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(oz)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(f"Obstacle contains non-finite center coordinates: ({ox}, {oy}, {oz})")
            return None
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle non-numeric center coordinates: {ex}")
        return None

    # 3. Extract Size
    size = data.get("size")
    if not isinstance(size, dict) or "x" not in size or "y" not in size:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle missing size dimensions: {size}")
        return None
    try:
        olength = float(size["x"])
        owidth = float(size["y"])
        oheight = float(size.get("z", 1.5))
        if not (np.isfinite(olength) and np.isfinite(owidth) and np.isfinite(oheight)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(f"Obstacle contains non-finite size dimensions: ({olength}, {owidth}, {oheight})")
            return None
        if olength <= 0.0 or owidth <= 0.0:
            if raise_on_corrupt:
                raise CorruptedObservationDataError(f"Obstacle non-positive dimensions: length={olength}, width={owidth}")
            return None
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle non-numeric size: {ex}")
        return None

    # 4. Extract Orientation
    orient = data.get("orientation")
    if not isinstance(orient, dict) or "w" not in orient:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle missing orientation: {orient}")
        return None
    try:
        qw = float(orient["w"])
        qz = float(orient.get("z", 0.0))
        if not (np.isfinite(qw) and np.isfinite(qz)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(f"Obstacle non-finite orientation: ({qw}, {qz})")
            return None
        obs_yaw = 2.0 * np.arctan2(qz, qw)
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(f"Obstacle non-numeric orientation: {ex}")
        return None

    trackline_id = str(data.get("trackline_id", obs.get("trackline_id", "unknown")))
    category = str(data.get("category", obs.get("category", "unknown")))

    return {
        "timestamp_micros": int(ts) if ts is not None else None,
        "center_x": ox,
        "center_y": oy,
        "center_z": oz,
        "length_m": olength,
        "width_m": owidth,
        "height_m": oheight,
        "yaw_rad": obs_yaw,
        "trackline_id": trackline_id,
        "category": category,
    }


class CollisionFreeResult(tuple):
    """5-element tuple (cf_score, first_col_time, min_clearance, collided_tracks, collided_types)
    extended with observation coverage metadata for reviewer contract compliance.
    """
    def __new__(
        cls,
        cf_score: Optional[float],
        first_collision_time_s: Optional[float],
        min_clearance_m: Optional[float],
        collided_tracks: List[str],
        collided_types: List[str],
        matched_observation_frames: int = 0,
        required_observation_frames: int = 0,
        observation_coverage_ratio: float = 0.0,
    ):
        inst = super().__new__(
            cls,
            (cf_score, first_collision_time_s, min_clearance_m, collided_tracks, collided_types),
        )
        inst.matched_observation_frames = matched_observation_frames
        inst.required_observation_frames = required_observation_frames
        inst.observation_coverage_ratio = observation_coverage_ratio
        return inst


def compute_collision_free_proxy(
    x: np.ndarray,
    y: np.ndarray,
    headings: np.ndarray,
    timestamps_us: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    vehicle: VehicleParameters,
    t0_us: int,
    touch_is_collision: bool = True,
    context_present: bool = True,
) -> CollisionFreeResult:
    """Evaluates Collision Free (CF) proxy metric across the trajectory.
    
    Returns:
      CollisionFreeResult tuple:
      (cf_score, first_collision_time_s, min_clearance_m, collided_tracks, collided_types)
      with attributes:
      .matched_observation_frames
      .required_observation_frames
      .observation_coverage_ratio
      
    State rules:
      1. Missing context: returns (None, None, None, [], []) -> invalid
      2. Clear road (0 obstacles): returns (1.0, None, None, [], []) -> pass
      3. Obstacles exist but lack timestamps or all timestamps are out of window:
         returns (None, None, None, [], []) -> invalid (insufficient observation)
      4. Normal matching: returns (cf_score, col_time, min_clearance, tracks, types)
    """
    if not context_present or obstacles is None:
        return CollisionFreeResult(None, None, None, [], [], 0, len(x) if x is not None else 0, 0.0)

    n_poses = len(x)
    required_frames = n_poses
    if len(obstacles) == 0:
        return CollisionFreeResult(1.0, None, None, [], [], 0, 0, 1.0)

    # Normalize all obstacles; reject non-finite / corrupted geometry
    norm_obstacles: List[Dict[str, Any]] = []
    for obs in obstacles:
        normalized = normalize_obstacle_record(obs, raise_on_corrupt=True)
        if normalized is not None:
            norm_obstacles.append(normalized)

    # If obstacles were provided but none could be parsed, fail validation
    if len(norm_obstacles) == 0:
        return CollisionFreeResult(None, None, None, [], [], 0, required_frames, 0.0)

    # Filter obstacles with timestamps
    obs_with_ts = [o for o in norm_obstacles if o["timestamp_micros"] is not None]
    if len(obs_with_ts) == 0:
        # Obstacles exist, but none have timestamps -> insufficient observation data
        return CollisionFreeResult(None, None, None, [], [], 0, required_frames, 0.0)

    t_start_us = int(timestamps_us[0])
    t_end_us = int(timestamps_us[-1])
    # Check if observation timestamps overlap evaluation horizon (with +/- 0.5s tolerance)
    window_start_us = t_start_us - 500_000
    window_end_us = t_end_us + 500_000
    in_window_obs = [o for o in obs_with_ts if window_start_us <= o["timestamp_micros"] <= window_end_us]
    if len(in_window_obs) == 0:
        # Obstacles exist in context, but all timestamps are completely outside evaluation window
        return CollisionFreeResult(None, None, None, [], [], 0, required_frames, 0.0)

    # Index in-window obstacles by timestamp
    obs_by_time: Dict[int, List[Dict[str, Any]]] = {}
    for obs in in_window_obs:
        obs_by_time.setdefault(obs["timestamp_micros"], []).append(obs)

    all_obs_timestamps = np.array(sorted(obs_by_time.keys())) if obs_by_time else np.array([], dtype=np.int64)

    min_clearance = float("inf")
    first_col_time = None
    collided_tracks = []
    collided_types = []
    matched_observation_frames = 0

    for i in range(n_poses):
        t_us = timestamps_us[i]
        ego_corners = get_ego_box_corners(
            x[i], y[i], headings[i],
            vehicle.length_m, vehicle.width_m, vehicle.rear_axle_to_center_m
        )

        matching_obs = []
        if t_us in obs_by_time:
            matching_obs = obs_by_time[t_us]
        elif len(all_obs_timestamps) > 0:
            idx = np.searchsorted(all_obs_timestamps, t_us)
            candidates = []
            if idx < len(all_obs_timestamps):
                candidates.append(all_obs_timestamps[idx])
            if idx > 0:
                candidates.append(all_obs_timestamps[idx - 1])
            best_ts = min(candidates, key=lambda c: abs(c - t_us))
            half_step = int(round(abs(timestamps_us[1] - timestamps_us[0]) / 2.0)) if len(timestamps_us) > 1 else 50_000
            if abs(best_ts - t_us) <= half_step:
                matching_obs = obs_by_time[best_ts]

        if len(matching_obs) > 0:
            matched_observation_frames += 1

        for obs in matching_obs:
            obs_corners = get_oriented_box_corners(
                obs["center_x"], obs["center_y"], obs["yaw_rad"], obs["length_m"], obs["width_m"]
            )
            is_col, clearance = sat_box_intersection(ego_corners, obs_corners, touch_is_collision)

            if is_col:
                t_s = float((t_us - t0_us) / 1_000_000.0)
                if first_col_time is None:
                    first_col_time = t_s
                track_id = obs["trackline_id"]
                cat = obs["category"]
                if track_id not in collided_tracks:
                    collided_tracks.append(track_id)
                if cat not in collided_types:
                    collided_types.append(cat)
                min_clearance = min(min_clearance, clearance)
            else:
                if clearance < min_clearance:
                    min_clearance = clearance

    coverage_ratio = float(matched_observation_frames / required_frames) if required_frames > 0 else 0.0

    # Reject if obstacles existed in context but none matched any trajectory frame
    if len(obstacles) > 0 and matched_observation_frames == 0:
        return CollisionFreeResult(
            None, None, None, [], [],
            matched_observation_frames=0,
            required_observation_frames=required_frames,
            observation_coverage_ratio=0.0,
        )

    cf_score = 0.0 if first_col_time is not None else 1.0
    # Clearance must be finite number or None (never Infinity in JSON output)
    final_clearance = float(min_clearance) if (min_clearance < float("inf") and np.isfinite(min_clearance)) else None
    return CollisionFreeResult(
        cf_score,
        first_col_time,
        final_clearance,
        collided_tracks,
        collided_types,
        matched_observation_frames=matched_observation_frames,
        required_observation_frames=required_frames,
        observation_coverage_ratio=coverage_ratio,
    )


def compute_dac_proxy(
    x: np.ndarray,
    y: np.ndarray,
    headings: np.ndarray,
    timestamps_us: np.ndarray,
    road_polygons: Optional[List[np.ndarray]],
    vehicle: VehicleParameters,
    t0_us: int,
    require_drivable_geometry: bool = True,
) -> Tuple[Optional[float], Optional[float], int]:
    """Evaluates Drivable Area Compliance (DAC) proxy metric.
    
    If road_polygons is missing, empty, or contains non-finite vertices, returns (None, None, 0).
    """
    if road_polygons is None or len(road_polygons) == 0:
        if require_drivable_geometry:
            return None, None, 0
        return 1.0, None, 0

    # Strict geometry validation: reject NaN / Inf polygons
    for poly in road_polygons:
        if not isinstance(poly, np.ndarray) or len(poly) < 3 or not np.all(np.isfinite(poly)):
            return None, None, 0

    n_poses = len(x)
    offroad_count = 0
    first_offroad_time = None

    for i in range(n_poses):
        corners = get_ego_box_corners(
            x[i], y[i], headings[i],
            vehicle.length_m, vehicle.width_m, vehicle.rear_axle_to_center_m
        )
        try:
            inside_mask = points_in_any_polygon(corners, road_polygons)
        except ValueError:
            return None, None, 0

        if not np.all(inside_mask):
            offroad_count += 1
            if first_offroad_time is None:
                first_offroad_time = float((timestamps_us[i] - t0_us) / 1_000_000.0)

    dac_score = 1.0 if offroad_count == 0 else 0.0
    return dac_score, first_offroad_time, offroad_count


def compute_ttc_proxy(
    x: np.ndarray,
    y: np.ndarray,
    headings: np.ndarray,
    speeds: np.ndarray,
    timestamps_us: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]],
    vehicle: VehicleParameters,
    t0_us: int,
    ttc_horizon_s: float = 1.0,
    context_present: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Evaluates Time-to-Collision (TTC) proxy metric using forward projections."""
    if not context_present or obstacles is None:
        return None, None, None, None

    if len(obstacles) == 0 or len(x) < 2:
        return 1.0, None, None, None

    # Normalize all obstacles; reject non-finite / corrupted geometry
    norm_obstacles: List[Dict[str, Any]] = []
    for obs in obstacles:
        normalized = normalize_obstacle_record(obs, raise_on_corrupt=True)
        if normalized is not None:
            norm_obstacles.append(normalized)

    if len(norm_obstacles) == 0:
        return None, None, None, None

    obs_with_ts = [o for o in norm_obstacles if o["timestamp_micros"] is not None]
    if len(obs_with_ts) == 0:
        return None, None, None, None

    t_start_us = int(timestamps_us[0])
    t_end_us = int(timestamps_us[-1])
    window_start_us = t_start_us - 500_000
    window_end_us = t_end_us + int(ttc_horizon_s * 1_000_000) + 500_000
    in_window_obs = [o for o in obs_with_ts if window_start_us <= o["timestamp_micros"] <= window_end_us]
    if len(in_window_obs) == 0:
        return None, None, None, None

    step_s = 0.2
    dt_proj_list = [round(float(dt), 2) for dt in np.arange(0.0, float(ttc_horizon_s) + 1e-6, step_s)]
    min_ttc = float("inf")
    failure_time = None
    failure_track = None
    matched_projections = 0

    obs_by_time: Dict[int, List[Dict[str, Any]]] = {}
    for obs in in_window_obs:
        obs_by_time.setdefault(obs["timestamp_micros"], []).append(obs)
    all_obs_timestamps = np.array(sorted(obs_by_time.keys())) if obs_by_time else np.array([], dtype=np.int64)

    n_poses = len(x)
    for i in range(n_poses):
        curr_t_us = timestamps_us[i]
        curr_x, curr_y = x[i], y[i]
        curr_h = headings[i]
        curr_v = speeds[i]

        for dt_proj in dt_proj_list:
            proj_t_us = int(curr_t_us + dt_proj * 1_000_000)
            proj_x = curr_x + curr_v * dt_proj * np.cos(curr_h)
            proj_y = curr_y + curr_v * dt_proj * np.sin(curr_h)
            proj_corners = get_ego_box_corners(
                proj_x, proj_y, curr_h,
                vehicle.length_m, vehicle.width_m, vehicle.rear_axle_to_center_m
            )

            matching_obs = []
            if proj_t_us in obs_by_time:
                matching_obs = obs_by_time[proj_t_us]
            elif len(all_obs_timestamps) > 0:
                idx = np.searchsorted(all_obs_timestamps, proj_t_us)
                candidates = []
                if idx < len(all_obs_timestamps):
                    candidates.append(all_obs_timestamps[idx])
                if idx > 0:
                    candidates.append(all_obs_timestamps[idx - 1])
                best_ts = min(candidates, key=lambda c: abs(c - proj_t_us))
                if abs(best_ts - proj_t_us) <= 100_000:
                    matching_obs = obs_by_time[best_ts]

            if len(matching_obs) > 0:
                matched_projections += 1

            for obs in matching_obs:
                obs_corners = get_oriented_box_corners(
                    obs["center_x"], obs["center_y"], obs["yaw_rad"], obs["length_m"], obs["width_m"]
                )
                is_col, _ = sat_box_intersection(proj_corners, obs_corners, touch_is_collision=True)

                if is_col:
                    if dt_proj < min_ttc:
                        min_ttc = dt_proj
                        failure_time = float((curr_t_us - t0_us) / 1_000_000.0)
                        failure_track = obs["trackline_id"]

    if len(obstacles) > 0 and matched_projections == 0:
        return None, None, None, None

    ttc_score = 0.0 if (min_ttc <= ttc_horizon_s) else 1.0
    final_ttc = float(min_ttc) if (min_ttc < float("inf") and np.isfinite(min_ttc)) else None
    return ttc_score, final_ttc, failure_time, failure_track


def compute_progress_gt_proxy(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_xyz: Optional[np.ndarray],
    stationary_threshold_m: float = 5.0,
    prepend_t0: bool = True,
    min_future_poses: int = 40,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Computes Ego Progress proxy (EP_GT) along Ground Truth polyline.
    
    If GT is missing or has fewer than min_future_poses points, returns None to flag missing data.
    """
    if gt_xyz is None or len(gt_xyz) < min_future_poses:
        return None, None, None, None

    if not np.all(np.isfinite(gt_xyz)):
        return None, None, None, None

    gt_polyline = gt_xyz[:, :2]
    # Prepend (0, 0) state at t0 if not already present, ensuring full 0->4s baseline
    if prepend_t0 and (abs(gt_polyline[0, 0]) > 1e-4 or abs(gt_polyline[0, 1]) > 1e-4):
        gt_polyline = np.vstack([[0.0, 0.0], gt_polyline])

    diffs = np.diff(gt_polyline, axis=0)
    s_gt = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    endpoint = np.array([pred_x[-1], pred_y[-1]])
    s_pred, lat_err = project_point_onto_polyline(endpoint, gt_polyline)

    if s_gt <= stationary_threshold_m:
        ep_score = 1.0
    else:
        ep_score = float(np.clip(s_pred / s_gt, 0.0, 1.0))

    gt_endpoint = gt_polyline[-1]
    endpoint_err = float(np.hypot(endpoint[0] - gt_endpoint[0], endpoint[1] - gt_endpoint[1]))

    return ep_score, s_gt, s_pred, endpoint_err


def compute_nurec_safety_proxy_v1_composite(
    cf: Optional[float],
    dac: Optional[float],
    ttc: Optional[float],
    ep_gt: Optional[float],
    fc: Optional[float],
) -> Optional[float]:
    """Computes composite NuRec Safety Proxy v1 score:
    
    S_proxy = CF * DAC_p * (5*TTC_p + 5*EP_GT + 2*FC) / 12
    Returns None if any component is None.
    """
    if any(v is None for v in (cf, dac, ttc, ep_gt, fc)):
        return None
    weighted_quality = (5.0 * ttc + 5.0 * ep_gt + 2.0 * fc) / 12.0
    return float(cf * dac * weighted_quality)
