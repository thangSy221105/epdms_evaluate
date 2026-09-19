"""Single timeline contract shared by predictions and ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TimeContractError(ValueError):
    """Base error for time or trajectory contract violations."""


class NonMonotonicWaypointTimelineError(TimeContractError):
    """Raised when waypoint timestamps decrease or duplicate."""


class InconsistentWaypointTimelineError(TimeContractError):
    """Raised when timestamp metadata is partial or uses mixed clock fields."""


class TimelineHorizonMismatchError(TimeContractError):
    """Raised when sampling or horizon does not match the configured grid."""


class TimelineOriginMismatchError(TimeContractError):
    """Raised when timestamps are not anchored to the resolved clip origin."""


class MissingTimeOriginError(TimeContractError):
    """Raised when t0 cannot be determined in strict mode."""


class ConflictingTimeOriginError(TimeContractError):
    """Raised when input sources provide conflicting t0 values."""


class MissingGroundTruthCoordinatesError(TimeContractError):
    """Raised when ground-truth waypoints lack required coordinates."""


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
    role: str = "unknown"
    timestamp_kind: str = "implicit_relative"
    includes_t0: bool = False
    source_timestamp_fields: List[str] = field(default_factory=list)


@dataclass
class NormalizedTrajectoryTimeline:
    """Normalized trajectory consumed by time-sensitive metrics."""

    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray]
    timestamps_us: np.ndarray
    waypoints: List[Dict[str, Any]]
    provenance: TimelineProvenance

    @property
    def includes_t0(self) -> bool:
        return self.provenance.includes_t0


def _coerce_t0(value: Any, source: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Invalid t0_us from {source}: {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid t0_us from {source}: {value!r}") from exc
    if not np.isfinite(f) or abs(f - round(f)) > 1e-6:
        raise ValueError(f"Invalid t0_us from {source}: {value!r}")
    return int(round(f))


def resolve_time_origin(pred_row: Optional[Dict[str, Any]], context_row: Optional[Dict[str, Any]], gt_row: Optional[Dict[str, Any]], strict_mode: bool = True, default_t0_us: Optional[int] = None) -> Tuple[int, str]:
    """Resolve one clip origin and reject conflicting source origins."""
    found: Dict[str, int] = {}
    for name, row in (("prediction", pred_row), ("context", context_row), ("ground_truth", gt_row)):
        if isinstance(row, dict) and row.get("t0_us") is not None:
            try:
                found[name] = _coerce_t0(row.get("t0_us"), name)
            except ValueError:
                if strict_mode:
                    raise
    if not found:
        if strict_mode:
            raise MissingTimeOriginError("Missing 't0_us' across all input sources (prediction, context, ground_truth)")
        if default_t0_us is None:
            raise MissingTimeOriginError("No t0_us available and no default provided")
        return _coerce_t0(default_t0_us, "fallback_default"), "fallback_default"
    if len(set(found.values())) > 1 and strict_mode:
        raise ConflictingTimeOriginError(f"Conflicting t0_us values across input sources: {found}")
    if max(found.values()) - min(found.values()) > 100_000:
        raise ConflictingTimeOriginError(f"Conflicting t0_us values across input sources: {found}")
    for preferred in ("prediction", "ground_truth", "context"):
        if preferred in found:
            return found[preferred], preferred
    source = next(iter(found))
    return found[source], source


_RELATIVE_TIME_KEYS = ("t_s", "timestamp_s", "time_s")
_ABSOLUTE_TIME_KEYS = ("timestamp_micros", "t_us", "timestamp_us")


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TimeContractError(f"Invalid boolean value for timestamp '{field_name}': {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TimeContractError(f"Invalid non-numeric timestamp '{field_name}': {value!r}") from exc
    if not np.isfinite(number):
        raise TimeContractError(f"Non-finite timestamp '{field_name}': {value!r}")
    return number


def _extract_timestamp_metadata(wp: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if not isinstance(wp, dict):
        raise TypeError(f"Waypoint must be dict, got {type(wp)}")
    relative = [(k, _number(wp[k], k)) for k in _RELATIVE_TIME_KEYS if k in wp and wp[k] is not None]
    absolute = [(k, _number(wp[k], k)) for k in _ABSOLUTE_TIME_KEYS if k in wp and wp[k] is not None]
    if len(relative) > 1 and any(abs(v - relative[0][1]) > 1e-9 for _, v in relative[1:]):
        raise InconsistentWaypointTimelineError(f"Conflicting relative timestamps in waypoint: {wp}")
    if len(absolute) > 1 and any(abs(v - absolute[0][1]) > 1e-3 for _, v in absolute[1:]):
        raise InconsistentWaypointTimelineError(f"Conflicting absolute timestamps in waypoint: {wp}")
    if relative and absolute:
        raise InconsistentWaypointTimelineError(f"Waypoint mixes relative and absolute timestamp fields: {wp}")
    if relative:
        return relative[0][1], "relative", relative[0][0]
    if absolute:
        return absolute[0][1], "absolute", absolute[0][0]
    return None, None, None


def extract_waypoint_time(wp: Dict[str, Any]) -> Optional[float]:
    """Backward-compatible timestamp extraction in seconds."""
    value, kind, _ = _extract_timestamp_metadata(wp)
    if value is None:
        return None
    return value if kind == "relative" else value / 1_000_000.0


def _coordinate(wp: Dict[str, Any], key_a: str, key_b: str, idx: int, role: str) -> float:
    value = wp.get(key_a, wp.get(key_b))
    if value is None:
        exc = MissingGroundTruthCoordinatesError if role == "ground_truth" else TimeContractError
        raise exc(f"Waypoint {idx} missing '{key_a}'/'{key_b}' coordinates")
    if isinstance(value, bool):
        raise TimeContractError(f"Boolean coordinate at waypoint {idx}: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TimeContractError(f"Non-numeric coordinate at waypoint {idx}: {value!r}") from exc
    if not np.isfinite(result):
        raise TimeContractError(f"Non-finite coordinate at waypoint {idx}: {result!r}")
    return result


def _as_waypoint_dicts(waypoints: Sequence[Any], role: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(waypoints):
        if isinstance(item, dict):
            result.append(dict(item))
            continue
        arr = np.asarray(item)
        if arr.ndim != 1 or arr.size < 2:
            raise TimeContractError(f"{role} waypoint {idx} must be a dict or [x,y,(z)] array")
        result.append({"x_m": arr[0], "y_m": arr[1], "z_m": arr[2] if arr.size > 2 else 0.0})
    return result


def normalize_trajectory_timeline(waypoints: Sequence[Any], t0_us: int, expected_frequency_hz: float = 10.0, expected_horizon_s: float = 4.0, role: str = "prediction", strict_grid: bool = True) -> NormalizedTrajectoryTimeline:
    """Normalize prediction or GT onto one explicit clip clock."""
    if not isinstance(waypoints, (list, tuple)):
        raise TypeError(f"Waypoints must be a list/tuple, got {type(waypoints)}")
    if expected_frequency_hz <= 0 or expected_horizon_s <= 0:
        raise ValueError("frequency and horizon must be positive")
    t0 = _coerce_t0(t0_us, "timeline")
    raw = _as_waypoint_dicts(waypoints, role)
    n_future = int(round(expected_frequency_hz * expected_horizon_s))
    if len(raw) not in (n_future, n_future + 1):
        raise TimelineHorizonMismatchError(f"{role} has {len(raw)} waypoints; expected {n_future} future poses or {n_future + 1} including t0")
    includes_t0 = len(raw) == n_future + 1
    # Coordinates are part of the same parser contract and are checked first
    # so a malformed GT is not hidden by a later timestamp-origin diagnostic.
    xs = np.asarray([_coordinate(wp, "x_m", "x", i, role) for i, wp in enumerate(raw)], dtype=float)
    ys = np.asarray([_coordinate(wp, "y_m", "y", i, role) for i, wp in enumerate(raw)], dtype=float)
    parsed = [_extract_timestamp_metadata(wp) for wp in raw]
    present = [p[0] is not None for p in parsed]
    if any(present) and not all(present):
        raise InconsistentWaypointTimelineError(f"{role} waypoints have partial timestamps")
    kinds = {p[1] for p in parsed if p[1] is not None}
    if len(kinds) > 1:
        raise InconsistentWaypointTimelineError(f"{role} mixes relative and absolute timestamp fields")
    dt_s = 1.0 / expected_frequency_hz
    expected_first = 0.0 if includes_t0 else dt_s
    timestamp_kind = "implicit_relative"
    source_fields: List[str] = []
    if kinds:
        kind = next(iter(kinds))
        values = np.asarray([p[0] for p in parsed], dtype=float)
        if np.any(np.diff(values) <= 0):
            raise NonMonotonicWaypointTimelineError(f"{role} timestamps must be strictly increasing")
        if kind == "absolute":
            timestamp_kind = "absolute_us"
            rel_s = (values - t0) / 1_000_000.0
        else:
            timestamp_kind = "relative_s"
            rel_s = values
        if abs(rel_s[0] - expected_first) > 0.001:
            if kind == "absolute":
                raise TimelineOriginMismatchError(f"{role} first timestamp is not anchored to t0+{expected_first:.3f}s: {rel_s[0]:.6f}s")
            raise TimelineOriginMismatchError(f"{role} relative timeline must start at {expected_first:.3f}s, got {rel_s[0]:.6f}s")
        if abs(rel_s[-1] - expected_horizon_s) > 0.001:
            raise TimelineHorizonMismatchError(f"{role} timeline ends at {rel_s[-1]:.6f}s; expected {expected_horizon_s:.6f}s")
        if strict_grid and np.any(np.abs(np.diff(rel_s) - dt_s) > max(0.001, 0.05 * dt_s)):
            raise TimelineHorizonMismatchError(f"{role} timestamp spacing does not match {expected_frequency_hz:g} Hz")
        for p in parsed:
            if p[2]:
                source_fields.append(p[2])
        timestamps_us = np.rint(t0 + rel_s * 1_000_000.0).astype(np.int64)
    else:
        rel_s = np.arange(0.0 if includes_t0 else dt_s, expected_horizon_s + dt_s / 2.0, dt_s)[: len(raw)]
        timestamps_us = np.rint(t0 + rel_s * 1_000_000.0).astype(np.int64)
    if len(timestamps_us) > 1 and np.any(np.diff(timestamps_us) <= 0):
        raise NonMonotonicWaypointTimelineError(f"{role} normalized timestamps are not strictly increasing")
    z_values: List[float] = []
    has_z = False
    for i, wp in enumerate(raw):
        value = wp.get("z_m", wp.get("z"))
        if value is None:
            z_values.append(0.0)
        else:
            if isinstance(value, bool):
                raise TimeContractError(f"Boolean z coordinate at waypoint {i}")
            try:
                zf = float(value)
            except (TypeError, ValueError) as exc:
                raise TimeContractError(f"Non-numeric z coordinate at waypoint {i}") from exc
            if not np.isfinite(zf):
                raise TimeContractError(f"Non-finite z coordinate at waypoint {i}")
            z_values.append(zf)
            has_z = True
    zs = np.asarray(z_values, dtype=float) if has_z else None
    provenance = TimelineProvenance(t0_us=t0, t0_source="resolved", frequency_hz=float(expected_frequency_hz), horizon_s=float(expected_horizon_s), first_waypoint_time_s=float(rel_s[0]), last_waypoint_time_s=float(rel_s[-1]), dt_s=dt_s, is_strict_grid_compliant=True, timestamps_us=timestamps_us.copy(), details={"role": role, "waypoint_count": len(raw)}, role=role, timestamp_kind=timestamp_kind, includes_t0=includes_t0, source_timestamp_fields=sorted(set(source_fields)))
    return NormalizedTrajectoryTimeline(xs, ys, zs, timestamps_us, raw, provenance)


def validate_and_normalize_timeline(raw_waypoints: List[Dict[str, Any]], target_poses: int, expected_frequency_hz: float = 10.0, expected_horizon_s: float = 4.0, allow_fixed_rate_grid: bool = True, strict_grid: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], TimelineProvenance]:
    """Round-4 compatibility wrapper used by legacy callers and audit."""
    if not isinstance(raw_waypoints, list):
        raise TypeError(f"Waypoints must be list, got {type(raw_waypoints)}")
    if len(raw_waypoints) < target_poses:
        raise ValueError(f"InsufficientWaypointsError: Expected at least {target_poses} waypoints, got {len(raw_waypoints)}")
    future = raw_waypoints[:target_poses]
    parsed = [_extract_timestamp_metadata(wp) for wp in future]
    present = [p[0] is not None for p in parsed]
    if any(present) and not all(present):
        raise InconsistentWaypointTimelineError(f"Waypoints have partial timestamps: {sum(present)} present, {len(present)-sum(present)} missing")
    expected_dt = 1.0 / expected_frequency_hz
    if present:
        kinds = {p[1] for p in parsed}
        if len(kinds) > 1:
            raise InconsistentWaypointTimelineError("Waypoints mix relative and absolute timestamp fields")
        values = np.asarray([p[0] for p in parsed], dtype=float)
        if np.any(np.diff(values) <= 0):
            raise NonMonotonicWaypointTimelineError("NonMonotonicWaypointTimelineError: waypoint timestamps must be strictly increasing")
        scale = 1_000_000.0 if next(iter(kinds)) == "absolute" else 1.0
        if strict_grid and np.any(np.abs(np.diff(values) - expected_dt * scale) > 0.05 * expected_dt * scale):
            raise TimelineHorizonMismatchError("Timeline step mismatch")
        observed = (values[-1] - values[0]) / scale + expected_dt
        if strict_grid and abs(observed - expected_horizon_s) > 0.1:
            raise TimelineHorizonMismatchError(f"Timeline horizon mismatch: observed {observed:.2f}s != expected {expected_horizon_s:.2f}s")
        times = values / scale
    else:
        if not allow_fixed_rate_grid:
            raise TimeContractError("Explicit waypoint timestamps required by observation policy")
        times = np.arange(1, target_poses + 1, dtype=float) * expected_dt
    xs = np.asarray([_coordinate(wp, "x_m", "x", i, "prediction") for i, wp in enumerate(future)], dtype=float)
    ys = np.asarray([_coordinate(wp, "y_m", "y", i, "prediction") for i, wp in enumerate(future)], dtype=float)
    provenance = TimelineProvenance(t0_us=0, t0_source="unresolved", frequency_hz=expected_frequency_hz, horizon_s=expected_horizon_s, first_waypoint_time_s=float(times[0]), last_waypoint_time_s=float(times[-1]), dt_s=expected_dt, is_strict_grid_compliant=True, timestamps_us=np.array([], dtype=np.int64))
    return xs, ys, future, provenance
