"""Implementation of NuRec Safety Proxy v1 metrics with strict required data contracts."""

from __future__ import annotations

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
) -> Tuple[Optional[float], Optional[float], Optional[float], List[str], List[str]]:
    """Evaluates Collision Free (CF) proxy metric across the trajectory.
    
    If context is missing, returns None for score to fail validation.
    If context is present and obstacle list is legitimately empty (clear road), returns 1.0.
    """
    if not context_present or obstacles is None:
        return None, None, None, [], []

    n_poses = len(x)
    if len(obstacles) == 0:
        return 1.0, None, float("inf"), [], []

    # Index obstacles by timestamp for fast lookup
    obs_by_time: Dict[int, List[Dict[str, Any]]] = {}
    for obs in obstacles:
        ts = obs.get("timestamp_micros")
        if ts is not None:
            obs_by_time.setdefault(int(ts), []).append(obs)

    all_obs_timestamps = np.array(sorted(obs_by_time.keys())) if obs_by_time else np.array([])

    min_clearance = float("inf")
    first_col_time = None
    collided_tracks = []
    collided_types = []

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
            if abs(best_ts - t_us) <= 50_000:
                matching_obs = obs_by_time[best_ts]

        for obs in matching_obs:
            center = obs.get("center", {})
            size = obs.get("size", {})
            orient = obs.get("orientation", {})
            ox = float(center.get("x", 0.0))
            oy = float(center.get("y", 0.0))
            olength = float(size.get("x", 4.0))
            owidth = float(size.get("y", 2.0))

            qw = float(orient.get("w", 1.0))
            qz = float(orient.get("z", 0.0))
            obs_yaw = 2.0 * np.arctan2(qz, qw)

            obs_corners = get_oriented_box_corners(ox, oy, obs_yaw, olength, owidth)
            is_col, clearance = sat_box_intersection(ego_corners, obs_corners, touch_is_collision)

            if is_col:
                t_s = float((t_us - t0_us) / 1_000_000.0)
                if first_col_time is None:
                    first_col_time = t_s
                track_id = str(obs.get("trackline_id", "unknown"))
                cat = str(obs.get("category", "unknown"))
                if track_id not in collided_tracks:
                    collided_tracks.append(track_id)
                if cat not in collided_types:
                    collided_types.append(cat)
                min_clearance = min(min_clearance, clearance)
            else:
                if clearance < min_clearance:
                    min_clearance = clearance

    cf_score = 0.0 if first_col_time is not None else 1.0
    return cf_score, first_col_time, min_clearance, collided_tracks, collided_types


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
    
    If road_polygons is missing and required, returns (None, None, 0) to flag missing data.
    """
    if road_polygons is None or len(road_polygons) == 0:
        if require_drivable_geometry:
            return None, None, 0
        return 1.0, None, 0

    n_poses = len(x)
    offroad_count = 0
    first_offroad_time = None

    for i in range(n_poses):
        corners = get_ego_box_corners(
            x[i], y[i], headings[i],
            vehicle.length_m, vehicle.width_m, vehicle.rear_axle_to_center_m
        )
        inside_mask = points_in_any_polygon(corners, road_polygons)
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
    """Evaluates Time-to-Collision (TTC) proxy metric using forward projections.
    
    If context is missing, returns (None, None, None, None).
    """
    if not context_present or obstacles is None:
        return None, None, None, None

    if len(obstacles) == 0 or len(x) < 2:
        return 1.0, None, None, None

    dt_proj_list = [0.0, 0.3, 0.6, 0.9]
    min_ttc = float("inf")
    failure_time = None
    failure_track = None

    obs_by_time: Dict[int, List[Dict[str, Any]]] = {}
    for obs in obstacles:
        ts = obs.get("timestamp_micros")
        if ts is not None:
            obs_by_time.setdefault(int(ts), []).append(obs)
    all_obs_timestamps = np.array(sorted(obs_by_time.keys())) if obs_by_time else np.array([])

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

            for obs in matching_obs:
                center = obs.get("center", {})
                size = obs.get("size", {})
                orient = obs.get("orientation", {})
                ox = float(center.get("x", 0.0))
                oy = float(center.get("y", 0.0))
                olength = float(size.get("x", 4.0))
                owidth = float(size.get("y", 2.0))
                qw = float(orient.get("w", 1.0))
                qz = float(orient.get("z", 0.0))
                obs_yaw = 2.0 * np.arctan2(qz, qw)

                obs_corners = get_oriented_box_corners(ox, oy, obs_yaw, olength, owidth)
                is_col, _ = sat_box_intersection(proj_corners, obs_corners, touch_is_collision=True)

                if is_col:
                    if dt_proj < min_ttc:
                        min_ttc = dt_proj
                        failure_time = float((curr_t_us - t0_us) / 1_000_000.0)
                        failure_track = str(obs.get("trackline_id", "unknown"))

    ttc_score = 0.0 if (min_ttc <= ttc_horizon_s) else 1.0
    return ttc_score, (min_ttc if min_ttc < float("inf") else None), failure_time, failure_track


def compute_progress_gt_proxy(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_xyz: Optional[np.ndarray],
    stationary_threshold_m: float = 5.0,
    prepend_t0: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Computes Ego Progress proxy (EP_GT) along Ground Truth polyline.
    
    If GT is missing or has fewer than 2 points, returns None to flag missing data.
    """
    if gt_xyz is None or len(gt_xyz) < 2:
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
