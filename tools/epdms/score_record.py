"""Single-condition evaluation engine with strict contract validation and error boundary."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .coordinates import derive_heading_from_xy
from .geometry_numpy import (
    get_ego_box_corners,
    point_in_polygon_ray_casting,
    points_in_any_polygon,
)
from .kinematics_numpy import compute_kinematics
from .proxy_metrics import (
    compute_collision_free_proxy,
    compute_dac_proxy,
    compute_nurec_safety_proxy_v1_composite,
    compute_progress_gt_proxy,
    compute_ttc_proxy,
)
from .schemas import EvaluationScoreRecord, VehicleParameters


def evaluate_single_condition(
    pred_row: Dict[str, Any],
    context_row: Optional[Dict[str, Any]],
    gt_row: Optional[Dict[str, Any]],
    vehicle: VehicleParameters,
    horizon_s: float = 4.0,
    frequency_hz: float = 10.0,
    rule_group: Optional[str] = None,
    lane_polygons: Optional[List[np.ndarray]] = None,
    touch_is_collision: bool = True,
    ttc_horizon_s: float = 1.0,
    progress_stationary_threshold_m: float = 5.0,
    strict_mode: bool = True,
    metric_profile: str = "nurec_safety_proxy_v1",
) -> EvaluationScoreRecord:
    """Evaluates a single condition (clip_id, mode, alpha) safely within an error boundary."""
    if metric_profile in ("navsim_v2_full", "navsim_v2_stage1"):
        raise NotImplementedError("OFFICIAL_PROFILE_NOT_IMPLEMENTED")

    t_start = time.perf_counter()

    rec = EvaluationScoreRecord(
        record_key="unknown",
        clip_id="unknown",
        mode="unknown",
        alpha=0.0,
        rule_group=rule_group,
        metric_profile=metric_profile,
        horizon_s=horizon_s,
        frequency_hz=frequency_hz,
    )

    try:
        # 1. Parse and validate identity
        if not isinstance(pred_row, dict):
            raise TypeError(f"pred_row must be dict, got {type(pred_row)}")

        clip_id = str(pred_row.get("clip_id", ""))
        if not clip_id:
            raise ValueError("Missing 'clip_id' in prediction record")

        mode = str(pred_row.get("mode", ""))
        if not mode:
            raise ValueError("Missing 'mode' in prediction record")

        raw_alpha = pred_row.get("alpha")
        if raw_alpha is None:
            raise ValueError("Missing 'alpha' in prediction record")
        alpha = float(raw_alpha)
        if not np.isfinite(alpha):
            raise ValueError(f"Invalid non-finite alpha value: {raw_alpha}")

        record_key = f"{clip_id}|{mode}|{alpha:.3f}".rstrip("0").rstrip(".") if alpha != 0 else f"{clip_id}|{mode}|0"
        rec.record_key = record_key
        rec.clip_id = clip_id
        rec.mode = mode
        rec.alpha = alpha

        # 2. Extract and validate waypoints
        target_future_poses = int(round(horizon_s * frequency_hz))  # 40 for 4.0s @ 10Hz

        if alpha == 0.0:
            raw_wps = pred_row.get("clean_waypoints")
            if raw_wps is None:
                # Fallback to guided only if clean not provided at alpha 0
                raw_wps = pred_row.get("guided_waypoints")
            if raw_wps is None:
                raise ValueError("Missing 'clean_waypoints' for baseline alpha=0")
        else:
            raw_wps = pred_row.get("guided_waypoints")
            if raw_wps is None or len(raw_wps) == 0:
                # Per contract: do NOT silently fallback to clean_waypoints
                raise ValueError("missing_guided_trajectory: alpha > 0 requires 'guided_waypoints'")

        if len(raw_wps) < target_future_poses:
            raise ValueError(
                f"InsufficientWaypointsError: Expected at least {target_future_poses} future waypoints for {horizon_s}s horizon, found {len(raw_wps)}"
            )

        future_wps = raw_wps[:target_future_poses]

        # Extract coordinates and check finite & required schema
        xs: List[float] = []
        ys: List[float] = []
        for i, wp in enumerate(future_wps):
            if not isinstance(wp, dict):
                raise TypeError(f"Waypoint {i} is not a dict: {wp}")
            if "x_m" not in wp or "y_m" not in wp:
                raise ValueError(f"Waypoint {i} missing 'x_m' or 'y_m': {wp}")
            x_val = float(wp["x_m"])
            y_val = float(wp["y_m"])
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                raise ValueError(f"Non-finite coordinate at waypoint {i}: x={x_val}, y={y_val}")
            xs.append(x_val)
            ys.append(y_val)

        # Prepend state at t=0.0 (origin of ar1_ego frame)
        x_arr = np.array([0.0] + xs, dtype=float)
        y_arr = np.array([0.0] + ys, dtype=float)
        n_poses = len(x_arr)

        raw_t0 = pred_row.get("t0_us")
        if raw_t0 is None:
            if strict_mode:
                raise ValueError("Missing 't0_us' in prediction record")
            t0_us = 5_100_000
        else:
            t0_us = int(raw_t0)

        dt_us = int(round(1_000_000 / frequency_hz))
        timestamps_us = t0_us + np.arange(n_poses, dtype=np.int64) * dt_us

        # 3. Derive headings with stationary protection
        headings = derive_heading_from_xy(x_arr, y_arr)

        # 4. Kinematics and Future Comfort
        kin = compute_kinematics(x_arr, y_arr, headings, dt=1.0 / frequency_hz)
        rec.max_abs_longitudinal_accel = kin.max_abs_longitudinal_accel
        rec.max_abs_lateral_accel = kin.max_abs_lateral_accel
        rec.max_jerk_magnitude = kin.max_jerk_magnitude
        rec.max_abs_longitudinal_jerk = kin.max_abs_longitudinal_jerk
        rec.max_abs_yaw_rate = kin.max_abs_yaw_rate
        rec.max_abs_yaw_acceleration = kin.max_abs_yaw_acceleration
        rec.comfort_failure_reasons = kin.failure_reasons
        fc_score = 1.0 if kin.comfort_pass else 0.0
        rec.future_comfort_proxy = fc_score

        # 5. Context validation (Obstacles)
        if context_row is None:
            raise ValueError("missing_context: Context record missing for clip")

        sc = context_row.get("semantic_context")
        if not isinstance(sc, dict):
            raise ValueError("missing_semantic_context: 'semantic_context' missing or not dict")

        obstacles = sc.get("obstacle", {}).get("all_obstacles")
        if obstacles is None:
            raise ValueError("missing_obstacles: 'all_obstacles' not found in context")

        # Compute Collision Free (CF)
        cf_score, col_t, min_clear, col_tracks, col_types = compute_collision_free_proxy(
            x_arr, y_arr, headings, timestamps_us, obstacles, vehicle,
            t0_us=t0_us, touch_is_collision=touch_is_collision, context_present=True
        )
        rec.collision_free_proxy = cf_score
        rec.first_collision_time_s = col_t
        rec.minimum_clearance_m = min_clear
        rec.collided_track_ids = col_tracks
        rec.collided_object_types = col_types

        # Compute Time-to-Collision (TTC)
        speeds = kin.velocities
        ttc_score, min_ttc, ttc_fail_t, ttc_tr = compute_ttc_proxy(
            x_arr, y_arr, headings, speeds, timestamps_us, obstacles, vehicle,
            t0_us=t0_us, ttc_horizon_s=ttc_horizon_s, context_present=True
        )
        rec.ttc_proxy = ttc_score
        rec.min_ttc_s = min_ttc
        rec.ttc_failure_time_s = ttc_fail_t
        rec.ttc_track_id = ttc_tr

        # 6. Drivable Area Compliance (DAC)
        if lane_polygons is None or len(lane_polygons) == 0:
            if strict_mode:
                raise ValueError("missing_drivable_geometry: No road/lane polygons loaded for clip")
            dac_score = None
        else:
            dac_score, off_t, off_count = compute_dac_proxy(
                x_arr, y_arr, headings, timestamps_us, lane_polygons, vehicle,
                t0_us=t0_us, require_drivable_geometry=strict_mode
            )
            rec.dac_proxy = dac_score
            rec.first_offroad_time_s = off_t
            rec.offroad_frame_count = off_count

        # 7. Ground Truth validation and Progress
        if gt_row is None:
            raise ValueError("missing_ground_truth: Ground truth record missing for clip")

        raw_gt = gt_row.get("ego_future_xyz")
        if raw_gt is None or len(raw_gt) < 2:
            raise ValueError("missing_ground_truth_waypoints: 'ego_future_xyz' missing or invalid")

        gt_xyz = np.array(raw_gt[:target_future_poses], dtype=float)
        if not np.all(np.isfinite(gt_xyz)):
            raise ValueError("non_finite_ground_truth: NaN or Inf found in Ground Truth")

        ep_score, gt_prog, pred_prog, ep_err = compute_progress_gt_proxy(
            x_arr, y_arr, gt_xyz, stationary_threshold_m=progress_stationary_threshold_m
        )
        rec.progress_gt_proxy = ep_score
        rec.gt_progress_m = gt_prog
        rec.pred_projected_progress_m = pred_prog
        rec.endpoint_displacement_error_m = ep_err

        # 8. Composite NuRec Safety Proxy v1
        composite = compute_nurec_safety_proxy_v1_composite(
            cf=cf_score,
            dac=dac_score,
            ttc=ttc_score,
            ep_gt=ep_score,
            fc=fc_score,
        )
        rec.nurec_safety_proxy_v1 = composite

        if composite is None:
            rec.valid = False
            rec.failure_stage = "compute_composite"
            rec.failure_reason = "One or more required metrics returned None"
        else:
            rec.valid = True

    except Exception as e:
        rec.valid = False
        rec.failure_stage = "evaluate_single_condition"
        rec.failure_type = type(e).__name__
        rec.failure_reason = str(e)
        # Clear out scores to avoid false reporting of failed conditions
        rec.nurec_safety_proxy_v1 = None
        rec.collision_free_proxy = None
        rec.dac_proxy = None
        rec.ttc_proxy = None
        rec.progress_gt_proxy = None
        rec.future_comfort_proxy = None

    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return rec
