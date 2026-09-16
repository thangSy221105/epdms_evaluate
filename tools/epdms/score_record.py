"""Single-condition evaluation engine with robust error boundary."""

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
) -> EvaluationScoreRecord:
    """Evaluates a single condition (clip_id, mode, alpha) safely within an error boundary."""
    t_start = time.perf_counter()

    clip_id = str(pred_row.get("clip_id", ""))
    mode = str(pred_row.get("mode", ""))
    alpha = float(pred_row.get("alpha", 0.0))
    record_key = f"{clip_id}|{mode}|{alpha:.3f}".rstrip("0").rstrip(".") if alpha != 0 else f"{clip_id}|{mode}|0"

    rec = EvaluationScoreRecord(
        record_key=record_key,
        clip_id=clip_id,
        mode=mode,
        alpha=alpha,
        rule_group=rule_group,
        metric_profile="nurec_safety_proxy_v1",
        horizon_s=horizon_s,
        frequency_hz=frequency_hz,
    )

    try:
        # 1. Extract waypoints
        # For alpha == 0, clean_waypoints represents the unguided baseline
        raw_wps = pred_row.get("clean_waypoints", []) if alpha == 0.0 else pred_row.get("guided_waypoints", [])
        if not raw_wps:
            raw_wps = pred_row.get("guided_waypoints", []) or pred_row.get("clean_waypoints", [])

        if not raw_wps or len(raw_wps) < 10:
            rec.valid = False
            rec.failure_stage = "extract_waypoints"
            rec.failure_type = "InsufficientWaypointsError"
            rec.failure_reason = f"Expected at least 10 waypoints, found {len(raw_wps)}"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        # Filter up to horizon_s (40 future waypoints)
        target_future_poses = int(round(horizon_s * frequency_hz))
        future_wps = raw_wps[:target_future_poses]

        xs = [float(wp.get("x_m", 0.0)) for wp in future_wps]
        ys = [float(wp.get("y_m", 0.0)) for wp in future_wps]

        # Prepend state at t=0.0 (origin of ar1_ego frame)
        x_arr = np.array([0.0] + xs, dtype=float)
        y_arr = np.array([0.0] + ys, dtype=float)
        n_poses = len(x_arr)

        t0_us = int(pred_row.get("t0_us", 5_100_000))
        dt_us = int(round(1_000_000 / frequency_hz))
        timestamps_us = t0_us + np.arange(n_poses, dtype=np.int64) * dt_us

        # 2. Derive headings
        headings = derive_heading_from_xy(x_arr, y_arr)

        # 3. Kinematics and Future Comfort
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

        # 4. Obstacles for Collision and TTC
        obstacles: List[Dict[str, Any]] = []
        if context_row:
            sc = context_row.get("semantic_context", {})
            obstacles = sc.get("obstacle", {}).get("all_obstacles", [])

        cf_score, col_t, min_clear, col_tracks, col_types = compute_collision_free_proxy(
            x_arr, y_arr, headings, timestamps_us, obstacles, vehicle, touch_is_collision=True
        )
        rec.collision_free_proxy = cf_score
        rec.first_collision_time_s = col_t
        rec.minimum_clearance_m = min_clear if min_clear < float("inf") else 10.0
        rec.collided_track_ids = col_tracks
        rec.collided_object_types = col_types

        # Speeds for TTC
        speeds = kin.velocities
        ttc_score, min_ttc, ttc_fail_t, ttc_tr = compute_ttc_proxy(
            x_arr, y_arr, headings, speeds, timestamps_us, obstacles, vehicle
        )
        rec.ttc_proxy = ttc_score
        rec.min_ttc_s = min_ttc
        rec.ttc_failure_time_s = ttc_fail_t
        rec.ttc_track_id = ttc_tr

        # 5. Drivable Area Compliance (DAC)
        polygons = lane_polygons or []
        dac_score, off_t, off_count = compute_dac_proxy(
            x_arr, y_arr, headings, polygons, vehicle
        )
        rec.dac_proxy = dac_score
        rec.first_offroad_time_s = off_t
        rec.offroad_frame_count = off_count

        # 6. Progress along Ground Truth
        gt_xyz = np.empty((0, 3), dtype=float)
        if gt_row:
            raw_gt = gt_row.get("ego_future_xyz", [])
            if raw_gt:
                gt_xyz = np.array(raw_gt[:target_future_poses], dtype=float)

        ep_score, gt_prog, pred_prog, ep_err = compute_progress_gt_proxy(
            x_arr, y_arr, gt_xyz
        )
        rec.progress_gt_proxy = ep_score
        rec.gt_progress_m = gt_prog
        rec.pred_projected_progress_m = pred_prog
        rec.endpoint_displacement_error_m = ep_err

        # 7. Composite NuRec Safety Proxy v1
        composite = compute_nurec_safety_proxy_v1_composite(
            cf=cf_score,
            dac=dac_score,
            ttc=ttc_score,
            ep_gt=ep_score,
            fc=fc_score,
        )
        rec.nurec_safety_proxy_v1 = composite
        rec.valid = True

    except Exception as e:
        rec.valid = False
        rec.failure_stage = "compute_metrics"
        rec.failure_type = type(e).__name__
        rec.failure_reason = str(e)

    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return rec
