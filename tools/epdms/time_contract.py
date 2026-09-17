"""Timeline normalization, validation, and contract enforcement for EPDMS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TimeContractError(ValueError):
    """Base error for time contract violations."""
    pass


class NonMonotonicWaypointTimelineError(TimeContractError):
    """Raised when waypoint timestamps are non-monotonic (decreasing or duplicate)."""
    pass


class InconsistentWaypointTimelineError(TimeContractError):
    """Raised when waypoints have inconsistent time metadata (e.g. partial timestamps)."""
    pass


class TimelineHorizonMismatchError(TimeContractError):
    """Raised when trajectory timestamps do not match expected horizon or sampling frequency."""
    pass


class MissingTimeOriginError(TimeContractError):
    """Raised when t0 cannot be determined across any input source in strict mode."""
    pass


class ConflictingTimeOriginError(TimeContractError):
    """Raised when multiple input sources provide conflicting t0 values."""
    pass


class MissingGroundTruthCoordinatesError(TimeContractError):
    """Raised when ground truth waypoints lack required coordinates."""
    pass


@dataclass
class TimelineProvenance:
    t0_us: int
    t0_source: str
    frequency_hz: float
    horizon_s: float
    first_waypoint_time_s: float
    last_waypoint_time_s: float
    dt_s: float
    is_strict_grid_compliant: bool
    timestamps_us: np.ndarray
    details: Dict[str, Any] = field(default_factory=dict)


def resolve_time_origin(
    pred_row: Optional[Dict[str, Any]],
    context_row: Optional[Dict[str, Any]],
    gt_row: Optional[Dict[str, Any]],
    strict_mode: bool = True,
    default_t0_us: Optional[int] = None,
) -> Tuple[int, str]:
    """Resolves t0_us across prediction, context, and ground truth with cross-source validation.
    
    Returns:
        (t0_us, t0_source)
    Raises:
        MissingTimeOriginError: if t0 is missing across all sources in strict mode.
        ConflictingTimeOriginError: if sources provide conflicting t0 values.
    """
    found_sources: Dict[str, int] = {}

    for src_name, row in [("prediction", pred_row), ("context", context_row), ("ground_truth", gt_row)]:
        if row and isinstance(row, dict) and "t0_us" in row and row["t0_us"] is not None:
            raw_val = row["t0_us"]
            if isinstance(raw_val, bool):
                continue
            try:
                t0_int = int(raw_val)
                found_sources[src_name] = t0_int
            except (TypeError, ValueError):
                continue

    if not found_sources:
        if strict_mode:
            raise MissingTimeOriginError("Missing 't0_us' across all input sources (prediction, context, ground truth)")
        if default_t0_us is not None:
            return default_t0_us, "fallback_default"
        raise MissingTimeOriginError("No t0_us available and no default provided")

    # Cross-source consistency check
    unique_vals = set(found_sources.values())
    if len(unique_vals) > 1:
        # Check if difference is greater than 100ms
        min_v = min(unique_vals)
        max_v = max(unique_vals)
        if abs(max_v - min_v) > 100_000:
            raise ConflictingTimeOriginError(
                f"Conflicting t0_us values across input sources: {found_sources}"
            )

    # Precedence: prediction > ground_truth > context
    for preferred in ["prediction", "ground_truth", "context"]:
        if preferred in found_sources:
            return found_sources[preferred], preferred

    first_src = next(iter(found_sources.keys()))
    return found_sources[first_src], first_src


def extract_waypoint_time(wp: Dict[str, Any]) -> Optional[float]:
    """Extracts timestamp in seconds from a waypoint dictionary without 'or' coercion.
    
    Validates cross-key consistency if multiple time representations are present.
    """
    if not isinstance(wp, dict):
        raise TypeError(f"Waypoint must be dict, got {type(wp)}")

    time_sec: Optional[float] = None
    time_us: Optional[float] = None

    # Check second keys explicitly
    for k in ["t_s", "timestamp_s", "time_s"]:
        if k in wp and wp[k] is not None:
            val = wp[k]
            if isinstance(val, bool):
                raise ValueError(f"Invalid boolean value for timestamp '{k}': {val}")
            try:
                f_val = float(val)
                if not np.isfinite(f_val):
                    raise ValueError(f"Non-finite timestamp '{k}': {val}")
                time_sec = f_val
                break
            except (TypeError, ValueError) as ex:
                raise ValueError(f"Invalid non-numeric timestamp '{k}': {ex}")

    # Check microsecond keys explicitly
    for k in ["timestamp_micros", "t_us", "timestamp_us"]:
        if k in wp and wp[k] is not None:
            val = wp[k]
            if isinstance(val, bool):
                raise ValueError(f"Invalid boolean value for timestamp '{k}': {val}")
            try:
                f_val = float(val)
                if not np.isfinite(f_val):
                    raise ValueError(f"Non-finite timestamp '{k}': {val}")
                time_us = f_val
                break
            except (TypeError, ValueError) as ex:
                raise ValueError(f"Invalid non-numeric timestamp '{k}': {ex}")

    # Cross-check consistency if both sec and us exist
    if time_sec is not None and time_us is not None:
        if abs(time_sec * 1_000_000.0 - time_us) > 1_000.0:  # > 1ms diff
            raise ValueError(f"Inconsistent waypoint timestamps: {time_sec}s vs {time_us}us")

    if time_sec is not None:
        return time_sec
    if time_us is not None:
        return time_us / 1_000_000.0
    return None


def validate_and_normalize_timeline(
    raw_waypoints: List[Dict[str, Any]],
    target_poses: int,
    expected_frequency_hz: float = 10.0,
    expected_horizon_s: float = 4.0,
    allow_fixed_rate_grid: bool = True,
    strict_grid: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], TimelineProvenance]:
    """Validates waypoints, verifies coordinates and timeline, and constructs strict grid.
    
    Returns:
        (xs, ys, validated_future_wps, provenance)
    """
    if not isinstance(raw_waypoints, list):
        raise TypeError(f"Waypoints must be list, got {type(raw_waypoints)}")

    if len(raw_waypoints) < target_poses:
        raise ValueError(
            f"InsufficientWaypointsError: Expected at least {target_poses} waypoints, got {len(raw_waypoints)}"
        )

    future_wps = raw_waypoints[:target_poses]
    expected_dt = 1.0 / expected_frequency_hz

    # Check timestamps across all waypoints
    times_s: List[float] = []
    has_any_time = False
    missing_time_count = 0

    for i, wp in enumerate(future_wps):
        t_val = extract_waypoint_time(wp)
        if t_val is not None:
            has_any_time = True
            times_s.append(t_val)
        else:
            missing_time_count += 1

    # Check partial timestamp inconsistency
    if has_any_time and missing_time_count > 0:
        raise InconsistentWaypointTimelineError(
            f"Waypoints have partial timestamps: {len(times_s)} present, {missing_time_count} missing"
        )

    # If timestamps are explicitly provided, validate monotonicity and grid compliance
    if has_any_time:
        for i in range(1, len(times_s)):
            dt = times_s[i] - times_s[i - 1]
            if dt <= 0.0:
                raise NonMonotonicWaypointTimelineError(
                    f"NonMonotonicWaypointTimelineError: Waypoint {i} timestamp {times_s[i]} <= previous {times_s[i-1]}"
                )

        first_t = times_s[0]
        last_t = times_s[-1]
        observed_horizon = last_t - first_t + expected_dt

        if strict_grid:
            # Check sampling interval
            for i in range(1, len(times_s)):
                dt = times_s[i] - times_s[i - 1]
                if abs(dt - expected_dt) > 0.05 * expected_dt:  # 5% tolerance
                    raise TimelineHorizonMismatchError(
                        f"Timeline step mismatch: observed dt={dt:.4f}s != expected {expected_dt:.4f}s"
                    )

            # Check expected horizon duration
            if abs(observed_horizon - expected_horizon_s) > 0.1:  # > 100ms tolerance
                raise TimelineHorizonMismatchError(
                    f"Timeline horizon mismatch: observed {observed_horizon:.2f}s != expected {expected_horizon_s:.2f}s"
                )
    else:
        if not allow_fixed_rate_grid:
            raise TimeContractError("Explicit waypoint timestamps required by observation policy")
        times_s = [round((i + 1) * expected_dt, 4) for i in range(target_poses)]

    # Validate coordinates
    xs: List[float] = []
    ys: List[float] = []

    for i, wp in enumerate(future_wps):
        x_val = wp.get("x_m", wp.get("x"))
        y_val = wp.get("y_m", wp.get("y"))

        if x_val is None or y_val is None:
            raise TimeContractError(f"Waypoint {i} missing 'x_m'/'y_m' or 'x'/'y': {wp}")
        if isinstance(x_val, bool) or isinstance(y_val, bool):
            raise TimeContractError(f"Boolean coordinate at waypoint {i}: x={x_val}, y={y_val}")

        try:
            xf = float(x_val)
            yf = float(y_val)
        except (TypeError, ValueError):
            raise TimeContractError(f"Non-numeric coordinate at waypoint {i}: x={x_val}, y={y_val}")

        if not (np.isfinite(xf) and np.isfinite(yf)):
            raise TimeContractError(f"Non-finite coordinate at waypoint {i}: x={xf}, y={yf}")

        xs.append(xf)
        ys.append(yf)

    provenance = TimelineProvenance(
        t0_us=0,  # will be populated with resolved t0_us
        t0_source="unresolved",
        frequency_hz=expected_frequency_hz,
        horizon_s=expected_horizon_s,
        first_waypoint_time_s=times_s[0] if times_s else 0.1,
        last_waypoint_time_s=times_s[-1] if times_s else expected_horizon_s,
        dt_s=expected_dt,
        is_strict_grid_compliant=True,
        timestamps_us=np.array([], dtype=np.int64),
    )

    return np.array(xs, dtype=float), np.array(ys, dtype=float), future_wps, provenance
