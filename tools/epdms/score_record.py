"""Score record calculation with strict time and observation contracts."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .coordinates import derive_heading_from_xy
from .kinematics_numpy import compute_kinematics
from .observation_contract import CorruptedObservationDataError
from .proxy_metrics import (
    compute_collision_free_proxy,
    compute_dac_proxy,
    compute_progress_gt_proxy,
    compute_ttc_proxy,
)
from .schemas import EvaluationScoreRecord, VehicleParameters
from .time_contract import (
    ConflictingTimeOriginError,
    InconsistentWaypointTimelineError,
    MissingGroundTruthCoordinatesError,
    MissingTimeOriginError,
    NonMonotonicWaypointTimelineError,
    TimeContractError,
    TimelineHorizonMismatchError,
    resolve_time_origin,
    validate_and_normalize_timeline,
)


def compute_nurec_safety_proxy_v1_composite(
    cf: Optional[float],
    dac: Optional[float],
    ttc: Optional[float],
    ep_gt: Optional[float],
    fc: Optional[float],
) -> Optional[float]:
    """Computes composite score: S = CF * DAC * (5*TTC + 5*EP_GT + 2*FC) / 12."""
    if any(m is None for m in (cf, dac, ttc, ep_gt, fc)):
        return None
    return float(cf * dac * (5.0 * ttc + 5.0 * ep_gt + 2.0 * fc) / 12.0)


def extract_and_validate_trajectory(
    pred_row: Dict[str, Any],
    alpha: float,
    target_future_poses: int = 40,
    expected_frequency_hz: float = 10.0,
    expected_horizon_s: float = 4.0,
    strict_grid: bool = True,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """Extracts waypoints from prediction record and validates geometry and timeline."""
    if alpha == 0.0:
        raw_wps = pred_row.get("clean_waypoints")
        if raw_wps is None and "trajectories" in pred_row and isinstance(pred_row["trajectories"], dict):
            raw_wps = pred_row["trajectories"].get("clean") or pred_row["trajectories"].get("guided")
        if raw_wps is None:
            raw_wps = pred_row.get("guided_waypoints")
        if raw_wps is None:
            raise ValueError("Missing 'clean_waypoints' for baseline alpha=0")
    else:
        raw_wps = pred_row.get("guided_waypoints")
        if raw_wps is None and "trajectories" in pred_row and isinstance(pred_row["trajectories"], dict):
            raw_wps = pred_row["trajectories"].get("guided")
        if raw_wps is None or len(raw_wps) == 0:
            raise ValueError("missing_guided_trajectory: alpha > 0 requires 'guided_waypoints'")

    xs_arr, ys_arr, future_wps, _ = validate_and_normalize_timeline(
        raw_waypoints=raw_wps,
        target_poses=target_future_poses,
        expected_frequency_hz=expected_frequency_hz,
        expected_horizon_s=expected_horizon_s,
        allow_fixed_rate_grid=True,
        strict_grid=strict_grid,
    )
    return list(xs_arr), list(ys_arr), future_wps


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
    map_status: Optional[str] = None,
) -> EvaluationScoreRecord:
    """Evaluates a single condition safely within an error boundary under strict contracts."""
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
        observation_policy="strict_full_coverage" if strict_mode else "relaxed",
        timeline_policy="strict_grid" if strict_mode else "relaxed",
        map_status=map_status,
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

        # 2. Extract and validate trajectory waypoints and timeline
        target_future_poses = int(round(horizon_s * frequency_hz))
        try:
            xs, ys, future_wps = extract_and_validate_trajectory(
                pred_row, alpha, target_future_poses=target_future_poses,
                expected_frequency_hz=frequency_hz, expected_horizon_s=horizon_s,
                strict_grid=strict_mode,
            )
        except (NonMonotonicWaypointTimelineError, TimelineHorizonMismatchError, InconsistentWaypointTimelineError) as t_err:
            rec.valid = False
            rec.failure_stage = "timeline_contract"
            rec.failure_type = type(t_err).__name__
            rec.failure_reason = str(t_err)
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        # Prepend state at t=0.0 (origin of ar1_ego frame)
        x_arr = np.array([0.0] + xs, dtype=float)
        y_arr = np.array([0.0] + ys, dtype=float)
        n_poses = len(x_arr)

        # 3. Resolve time origin t0
        try:
            t0_us, t0_src = resolve_time_origin(
                pred_row, context_row, gt_row, strict_mode=strict_mode
            )
            rec.t0_source = t0_src
        except (MissingTimeOriginError, ConflictingTimeOriginError) as t0_err:
            rec.valid = False
            rec.failure_stage = "time_origin_contract"
            rec.failure_type = type(t0_err).__name__
            rec.failure_reason = str(t0_err)
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        dt_us = int(round(1_000_000 / frequency_hz))
        timestamps_us = t0_us + np.arange(n_poses, dtype=np.int64) * dt_us

        # 4. Derive headings and Kinematics
        headings = derive_heading_from_xy(x_arr, y_arr)
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
        try:
            cf_res = compute_collision_free_proxy(
                x_arr, y_arr, headings, timestamps_us, obstacles, vehicle,
                t0_us=t0_us, touch_is_collision=touch_is_collision, context_present=True,
                strict_mode=strict_mode,
            )
            cf_score, col_t, min_clear, col_tracks, col_types = cf_res
            rec.cf_required_frames = getattr(cf_res, "cf_required_frames", len(x_arr))
            rec.cf_observed_frames = getattr(cf_res, "cf_observed_frames", 0)
            rec.cf_confirmed_empty_frames = getattr(cf_res, "cf_confirmed_empty_frames", 0)
            rec.cf_missing_frames = getattr(cf_res, "cf_missing_frames", 0)
            rec.cf_coverage_ratio = getattr(cf_res, "cf_coverage_ratio", 0.0)
            rec.invalid_obstacle_count = getattr(cf_res, "invalid_obstacle_count", 0)
            rec.missing_timestamp_obstacle_count = getattr(cf_res, "missing_timestamp_obstacle_count", 0)
            # Backward compatibility aliases
            rec.matched_observation_frames = rec.cf_observed_frames
            rec.required_observation_frames = rec.cf_required_frames
            rec.observation_coverage_ratio = rec.cf_coverage_ratio or 0.0
        except CorruptedObservationDataError as e:
            rec.valid = False
            rec.failure_stage = "obstacle_observation_contract"
            rec.failure_type = "CORRUPTED_OBSERVATION_DATA"
            rec.failure_reason = str(e)
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        if obstacles is not None and len(obstacles) > 0 and cf_score is None:
            rec.valid = False
            rec.failure_stage = "obstacle_observation_contract"
            rec.failure_type = "INSUFFICIENT_OBSERVATION_DATA"
            rec.failure_reason = f"Obstacles exist but observation coverage is incomplete (coverage={rec.cf_coverage_ratio})"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        rec.collision_free_proxy = cf_score
        rec.first_collision_time_s = col_t
        rec.minimum_clearance_m = min_clear if (min_clear is not None and np.isfinite(min_clear) and min_clear < float("inf")) else None
        rec.collided_track_ids = col_tracks
        rec.collided_object_types = col_types

        # Compute Time-to-Collision (TTC)
        speeds = kin.velocities
        try:
            ttc_res = compute_ttc_proxy(
                x_arr, y_arr, headings, speeds, timestamps_us, obstacles, vehicle,
                t0_us=t0_us, ttc_horizon_s=ttc_horizon_s, context_present=True,
                strict_mode=strict_mode,
            )
            ttc_score, min_ttc, ttc_fail_t, ttc_tr = ttc_res
            rec.ttc_required_observations = getattr(ttc_res, "ttc_required_observations", 0)
            rec.ttc_observed_observations = getattr(ttc_res, "ttc_observed_observations", 0)
            rec.ttc_missing_observations = getattr(ttc_res, "ttc_missing_observations", 0)
            rec.ttc_coverage_ratio = getattr(ttc_res, "ttc_coverage_ratio", 0.0)
        except CorruptedObservationDataError as e:
            rec.valid = False
            rec.failure_stage = "obstacle_observation_contract"
            rec.failure_type = "CORRUPTED_OBSERVATION_DATA"
            rec.failure_reason = str(e)
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        if obstacles is not None and len(obstacles) > 0 and ttc_score is None:
            rec.valid = False
            rec.failure_stage = "obstacle_observation_contract"
            rec.failure_type = "INSUFFICIENT_OBSERVATION_DATA"
            rec.failure_reason = f"TTC observation coverage is incomplete (coverage={rec.ttc_coverage_ratio})"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        rec.ttc_proxy = ttc_score
        rec.min_ttc_s = min_ttc if (min_ttc is not None and np.isfinite(min_ttc) and min_ttc < float("inf")) else None
        rec.ttc_failure_time_s = ttc_fail_t
        rec.ttc_track_id = ttc_tr

        # 6. Drivable Area Compliance (DAC)
        if map_status == "PARTIAL" and strict_mode:
            rec.valid = False
            rec.failure_stage = "map_geometry_contract"
            rec.failure_type = "PartialCorruptedMapError"
            rec.failure_reason = "Map contains corrupted/partial polygons and cannot be used for strict scoring"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        if lane_polygons is not None and len(lane_polygons) > 0:
            for p in lane_polygons:
                if not isinstance(p, np.ndarray) or len(p) < 3 or not np.all(np.isfinite(p)):
                    rec.valid = False
                    rec.failure_stage = "map_geometry_contract"
                    rec.failure_type = "NonFiniteGeometryError"
                    rec.failure_reason = "Map polygon contains non-finite coordinates or fewer than 3 vertices"
                    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                    return rec
            dac_score, off_t, off_count = compute_dac_proxy(
                x_arr, y_arr, headings, timestamps_us, lane_polygons, vehicle,
                t0_us=t0_us, require_drivable_geometry=strict_mode,
            )
            rec.dac_proxy = dac_score
            rec.first_offroad_time_s = off_t
            rec.offroad_frame_count = off_count
        else:
            if strict_mode:
                raise ValueError("missing_drivable_geometry: No road/lane polygons loaded for clip")
            dac_score = None

        # 7. Ground Truth validation and Progress
        if gt_row is None:
            raise ValueError("missing_ground_truth: Ground truth record missing for clip")

        raw_gt = gt_row.get("ego_future_xyz") or gt_row.get("expert_future") or gt_row.get("future_waypoints")
        if raw_gt is None or len(raw_gt) < target_future_poses:
            rec.valid = False
            rec.failure_stage = "ground_truth_contract"
            rec.failure_type = "InsufficientWaypointsError"
            rec.failure_reason = f"Ground truth missing or has fewer than {target_future_poses} waypoints"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        if isinstance(raw_gt[0], dict):
            gt_pts = []
            prev_gt_time = None
            for idx, p in enumerate(raw_gt[:target_future_poses]):
                if not isinstance(p, dict):
                    rec.valid = False
                    rec.failure_stage = "ground_truth_contract"
                    rec.failure_type = "InvalidGroundTruthWaypointError"
                    rec.failure_reason = f"Ground truth waypoint {idx} is not a dict: {p}"
                    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                    return rec

                gt_t = p.get("timestamp_micros") or p.get("t_us") or p.get("t_s") or p.get("timestamp_s") or p.get("time_s")
                if gt_t is not None:
                    try:
                        gt_tf = float(gt_t)
                        if prev_gt_time is not None and gt_tf <= prev_gt_time:
                            rec.valid = False
                            rec.failure_stage = "ground_truth_contract"
                            rec.failure_type = "NonMonotonicWaypointTimelineError"
                            rec.failure_reason = f"Ground truth waypoint {idx} timestamp {gt_tf} <= previous {prev_gt_time}"
                            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                            return rec
                        prev_gt_time = gt_tf
                    except (TypeError, ValueError):
                        pass

                x_val = p.get("x_m", p.get("x"))
                y_val = p.get("y_m", p.get("y"))
                if x_val is None or y_val is None:
                    rec.valid = False
                    rec.failure_stage = "ground_truth_contract"
                    rec.failure_type = "MissingGroundTruthCoordinatesError"
                    rec.failure_reason = f"Ground truth waypoint {idx} missing 'x' and 'y' coordinates: {p}"
                    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                    return rec

                try:
                    xf = float(x_val)
                    yf = float(y_val)
                    zf = float(p.get("z_m", p.get("z", 0.0)))
                except (TypeError, ValueError) as ex:
                    rec.valid = False
                    rec.failure_stage = "ground_truth_contract"
                    rec.failure_type = "NonNumericGroundTruthError"
                    rec.failure_reason = f"Non-numeric Ground Truth coordinate at {idx}: {ex}"
                    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                    return rec

                if not (np.isfinite(xf) and np.isfinite(yf) and np.isfinite(zf)):
                    rec.valid = False
                    rec.failure_stage = "ground_truth_contract"
                    rec.failure_type = "NonFiniteGroundTruthError"
                    rec.failure_reason = f"NaN or Inf found in Ground Truth at {idx}"
                    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
                    return rec

                gt_pts.append([xf, yf, zf])
            gt_xyz = np.array(gt_pts, dtype=float)
        else:
            gt_xyz = np.array(raw_gt[:target_future_poses], dtype=float)

        if not np.all(np.isfinite(gt_xyz)):
            rec.valid = False
            rec.failure_stage = "ground_truth_contract"
            rec.failure_type = "NonFiniteGroundTruthError"
            rec.failure_reason = "NaN or Inf found in Ground Truth"
            rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return rec

        ep_score, gt_prog, pred_prog, ep_err = compute_progress_gt_proxy(
            x_arr, y_arr, gt_xyz, stationary_threshold_m=progress_stationary_threshold_m, min_future_poses=target_future_poses
        )
        rec.progress_gt_proxy = ep_score
        rec.gt_progress_m = gt_prog
        rec.pred_projected_progress_m = pred_prog
        rec.endpoint_displacement_error_m = ep_err

        # Compute ADE and FDE against Ground Truth
        pred_coords = np.column_stack([xs, ys])
        gt_coords = gt_xyz[:, :2]
        disp_errors = np.hypot(pred_coords[:, 0] - gt_coords[:, 0], pred_coords[:, 1] - gt_coords[:, 1])
        rec.ade_m = float(np.mean(disp_errors))
        rec.fde_m = float(disp_errors[-1])

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
        rec.nurec_safety_proxy_v1 = None
        rec.collision_free_proxy = None
        rec.dac_proxy = None
        rec.ttc_proxy = None
        rec.progress_gt_proxy = None
        rec.future_comfort_proxy = None

    rec.runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return rec
