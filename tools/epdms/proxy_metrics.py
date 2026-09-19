"""Implementation of NuRec Safety Proxy v1 metrics with strict required data contracts."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

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
from .observation_contract import (
    CorruptedObservationDataError,
    ObservationCoverageDiagnostics,
    ObservationState,
    evaluate_query_coverage,
    index_and_filter_obstacles,
    normalize_and_validate_obstacle,
)
from .schemas import VehicleParameters


def normalize_obstacle_record(obs: Dict[str, Any], raise_on_corrupt: bool = True) -> Optional[Dict[str, Any]]:
    """Normalizes an obstacle record from either flat or nested schema into a standard format.
    
    Delegates to normalize_and_validate_obstacle in observation_contract.
    """
    return normalize_and_validate_obstacle(obs, raise_on_corrupt=raise_on_corrupt)


class CollisionFreeResult(tuple):
    """5-element tuple (cf_score, first_col_time, min_clearance, collided_tracks, collided_types)
    extended with comprehensive observation coverage metadata for reviewer contract compliance.
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
        confirmed_empty_frames: int = 0,
        missing_frames: int = 0,
        invalid_obstacle_count: int = 0,
        missing_timestamp_obstacle_count: int = 0,
    ):
        inst = super().__new__(
            cls,
            (cf_score, first_collision_time_s, min_clearance_m, collided_tracks, collided_types),
        )
        inst.matched_observation_frames = matched_observation_frames
        inst.required_observation_frames = required_observation_frames
        inst.observation_coverage_ratio = observation_coverage_ratio
        inst.cf_observed_frames = matched_observation_frames
        inst.cf_required_frames = required_observation_frames
        inst.cf_confirmed_empty_frames = confirmed_empty_frames
        inst.cf_missing_frames = missing_frames
        inst.cf_coverage_ratio = observation_coverage_ratio
        inst.invalid_obstacle_count = invalid_obstacle_count
        inst.missing_timestamp_obstacle_count = missing_timestamp_obstacle_count
        return inst


class TtcResult(tuple):
    """4-element tuple (ttc_score, min_ttc_s, ttc_failure_time_s, ttc_track_id)
    extended with TTC projection observation coverage metadata.
    """
    def __new__(
        cls,
        ttc_score: Optional[float],
        min_ttc_s: Optional[float],
        ttc_failure_time_s: Optional[float],
        ttc_track_id: Optional[str],
        ttc_required_observations: int = 0,
        ttc_observed_observations: int = 0,
        ttc_confirmed_empty_observations: int = 0,
        ttc_missing_observations: int = 0,
        ttc_coverage_ratio: float = 0.0,
    ):
        inst = super().__new__(
            cls,
            (ttc_score, min_ttc_s, ttc_failure_time_s, ttc_track_id),
        )
        inst.ttc_required_observations = ttc_required_observations
        inst.ttc_observed_observations = ttc_observed_observations
        inst.ttc_confirmed_empty_observations = ttc_confirmed_empty_observations
        inst.ttc_missing_observations = ttc_missing_observations
        inst.ttc_coverage_ratio = ttc_coverage_ratio
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
    strict_mode: bool = False,
    confirmed_empty_timestamps: Optional[Set[int]] = None,
    confirmed_empty_scene: bool = False,
) -> CollisionFreeResult:
    """Evaluates Collision Free (CF) proxy metric across the trajectory with observation coverage gating.
    
    Returns:
      CollisionFreeResult tuple:
      (cf_score, first_collision_time_s, min_clearance_m, collided_tracks, collided_types)
    """
    if not context_present or obstacles is None:
        return CollisionFreeResult(
            None, None, None, [], [],
            matched_observation_frames=0,
            required_observation_frames=len(x) if x is not None else 0,
            observation_coverage_ratio=0.0,
            missing_frames=len(x) if x is not None else 0,
        )

    n_poses = len(x)
    required_frames = n_poses

    # Confirmed clear road (empty scene)
    if len(obstacles) == 0:
        # An empty container is absence of records, not evidence that every
        # required frame was observed.  Strict production scoring therefore
        # requires an explicit dataset-level observation attestation.
        if confirmed_empty_scene or not strict_mode:
            return CollisionFreeResult(
                1.0, None, None, [], [],
                matched_observation_frames=0,
                required_observation_frames=required_frames,
                observation_coverage_ratio=1.0,
                confirmed_empty_frames=required_frames,
            )
        return CollisionFreeResult(
            None, None, None, [], [],
            matched_observation_frames=0,
            required_observation_frames=required_frames,
            observation_coverage_ratio=0.0,
            missing_frames=required_frames,
        )

    # Index obstacles and validate every record (strict mode raises CorruptedObservationDataError)
    obs_by_time, all_obs_timestamps, invalid_count, missing_ts_count = index_and_filter_obstacles(
        obstacles, raise_on_corrupt=strict_mode
    )

    # Evaluate observation coverage on the required ego timestamps
    states, observed_count, confirmed_empty_count, missing_count, coverage_ratio = evaluate_query_coverage(
        timestamps_us, obs_by_time, all_obs_timestamps, confirmed_empty_timestamps=confirmed_empty_timestamps
    )

    # Gating: In strict mode, if obstacles were present but coverage is incomplete, reject
    if strict_mode and (missing_count > 0 or observed_count == 0):
        return CollisionFreeResult(
            None, None, None, [], [],
            matched_observation_frames=observed_count,
            required_observation_frames=required_frames,
            observation_coverage_ratio=coverage_ratio,
            confirmed_empty_frames=confirmed_empty_count,
            missing_frames=missing_count,
            invalid_obstacle_count=invalid_count,
            missing_timestamp_obstacle_count=missing_ts_count,
        )

    # Non-strict mode fallback check
    if len(obstacles) > 0 and observed_count == 0 and confirmed_empty_count == 0:
        return CollisionFreeResult(
            None, None, None, [], [],
            matched_observation_frames=0,
            required_observation_frames=required_frames,
            observation_coverage_ratio=0.0,
            missing_frames=required_frames,
            invalid_obstacle_count=invalid_count,
            missing_timestamp_obstacle_count=missing_ts_count,
        )

    min_clearance = float("inf")
    first_col_time = None
    collided_tracks: List[str] = []
    collided_types: List[str] = []

    for i in range(n_poses):
        t_us = int(timestamps_us[i])
        ego_corners = get_ego_box_corners(
            x[i], y[i], headings[i],
            vehicle.length_m, vehicle.width_m, vehicle.rear_axle_to_center_m
        )

        matching_obs: List[Dict[str, Any]] = []
        if t_us in obs_by_time:
            matching_obs = obs_by_time[t_us]
        elif len(all_obs_timestamps) > 0:
            idx = np.searchsorted(all_obs_timestamps, t_us)
            candidates = []
            if idx < len(all_obs_timestamps):
                candidates.append(all_obs_timestamps[idx])
            if idx > 0:
                candidates.append(all_obs_timestamps[idx - 1])
            if candidates:
                best_ts = min(candidates, key=lambda c: abs(c - t_us))
                half_step = int(round(abs(timestamps_us[1] - timestamps_us[0]) / 2.0)) if len(timestamps_us) > 1 else 50_000
                if abs(best_ts - t_us) <= half_step:
                    matching_obs = obs_by_time[best_ts]

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

    cf_score = 0.0 if first_col_time is not None else 1.0
    final_clearance = float(min_clearance) if (min_clearance < float("inf") and np.isfinite(min_clearance)) else None
    return CollisionFreeResult(
        cf_score,
        first_col_time,
        final_clearance,
        collided_tracks,
        collided_types,
        matched_observation_frames=observed_count,
        required_observation_frames=required_frames,
        observation_coverage_ratio=coverage_ratio,
        confirmed_empty_frames=confirmed_empty_count,
        missing_frames=missing_count,
        invalid_obstacle_count=invalid_count,
        missing_timestamp_obstacle_count=missing_ts_count,
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
    """Evaluates Drivable Area Compliance (DAC) proxy metric."""
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
    strict_mode: bool = False,
    confirmed_empty_timestamps: Optional[Set[int]] = None,
    confirmed_empty_scene: bool = False,
) -> TtcResult:
    """Evaluates Time-to-Collision (TTC) proxy metric with separate TTC observation coverage gating."""
    if not context_present or obstacles is None:
        return TtcResult(None, None, None, None)

    if len(x) < 2:
        return TtcResult(None, None, None, None)

    if len(obstacles) == 0:
        # Match CF: strict mode accepts an empty scene only when the caller
        # supplies independent evidence that the queried frames were observed.
        if strict_mode and not confirmed_empty_scene:
            return TtcResult(
                None, None, None, None,
                ttc_required_observations=len(x),
                ttc_observed_observations=0,
                ttc_confirmed_empty_observations=0,
                ttc_missing_observations=len(x),
                ttc_coverage_ratio=0.0,
            )
        return TtcResult(1.0, None, None, None, len(x), 0, len(x), 0, 1.0)

    # Index obstacles and validate
    obs_by_time, all_obs_timestamps, invalid_count, missing_ts_count = index_and_filter_obstacles(
        obstacles, raise_on_corrupt=strict_mode
    )

    step_s = 0.2
    dt_proj_list = [round(float(dt), 2) for dt in np.arange(0.0, float(ttc_horizon_s) + 1e-6, step_s)]

    # Collect all unique projection timestamps required by TTC
    n_poses = len(x)
    all_proj_ts_set: Set[int] = set()
    for i in range(n_poses):
        curr_t_us = int(timestamps_us[i])
        for dt_proj in dt_proj_list:
            all_proj_ts_set.add(curr_t_us + int(round(dt_proj * 1_000_000)))

    unique_proj_ts = np.array(sorted(all_proj_ts_set), dtype=np.int64)
    total_proj_queries = len(unique_proj_ts)

    # Evaluate TTC coverage across all projection queries
    ttc_states, ttc_obs, ttc_empty, ttc_missing, ttc_cov_ratio = evaluate_query_coverage(
        unique_proj_ts, obs_by_time, all_obs_timestamps, confirmed_empty_timestamps=confirmed_empty_timestamps,
        half_step_us=100_000
    )

    # Gating in strict mode: if obstacles present but TTC coverage incomplete, reject
    if strict_mode and (ttc_missing > 0 or ttc_obs == 0):
        return TtcResult(
            None, None, None, None,
            ttc_required_observations=total_proj_queries,
            ttc_observed_observations=ttc_obs,
            ttc_confirmed_empty_observations=ttc_empty,
            ttc_missing_observations=ttc_missing,
            ttc_coverage_ratio=ttc_cov_ratio,
        )

    if len(obstacles) > 0 and ttc_obs == 0 and ttc_empty == 0:
        return TtcResult(
            None, None, None, None,
            ttc_required_observations=total_proj_queries,
            ttc_observed_observations=0,
            ttc_confirmed_empty_observations=0,
            ttc_missing_observations=total_proj_queries,
            ttc_coverage_ratio=0.0,
        )

    min_ttc = float("inf")
    failure_time = None
    failure_track = None

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
                if candidates:
                    best_ts = min(candidates, key=lambda c: abs(c - proj_t_us))
                    if abs(best_ts - proj_t_us) <= 100_000:
                        matching_obs = obs_by_time[best_ts]

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

    ttc_score = 0.0 if failure_time is not None else 1.0
    final_min_ttc = float(min_ttc) if (min_ttc < float("inf") and np.isfinite(min_ttc)) else None

    return TtcResult(
        ttc_score,
        final_min_ttc,
        failure_time,
        failure_track,
        ttc_required_observations=total_proj_queries,
        ttc_observed_observations=ttc_obs,
        ttc_confirmed_empty_observations=ttc_empty,
        ttc_missing_observations=ttc_missing,
        ttc_coverage_ratio=ttc_cov_ratio,
    )


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
