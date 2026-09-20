"""Observation coverage and obstacle integrity contracts for EPDMS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class ObservationState(str, Enum):
    OBSERVED_WITH_OBJECTS = "OBSERVED_WITH_OBJECTS"
    OBSERVED_EMPTY = "OBSERVED_EMPTY"
    MISSING_OBSERVATION = "MISSING_OBSERVATION"


class CorruptedObservationDataError(ValueError):
    """Raised when an obstacle record violates the data contract (missing timestamp or corrupted geometry)."""
    def __init__(
        self,
        message: str,
        trackline_id: Optional[str] = None,
        field_name: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.trackline_id = trackline_id
        self.field_name = field_name
        self.reason = reason or message


@dataclass
class ObservationCoverageDiagnostics:
    cf_required_frames: int = 0
    cf_observed_frames: int = 0
    cf_confirmed_empty_frames: int = 0
    cf_missing_frames: int = 0
    cf_coverage_ratio: Optional[float] = None

    ttc_required_observations: int = 0
    ttc_observed_observations: int = 0
    ttc_confirmed_empty_observations: int = 0
    ttc_missing_observations: int = 0
    ttc_coverage_ratio: Optional[float] = None

    invalid_obstacle_count: int = 0
    missing_timestamp_obstacle_count: int = 0
    observation_policy: str = "strict_full_coverage"
    is_cf_coverage_adequate: bool = False
    is_ttc_coverage_adequate: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def normalize_and_validate_obstacle(
    obs: Dict[str, Any],
    raise_on_corrupt: bool = True,
) -> Optional[Dict[str, Any]]:
    """Normalizes and validates an obstacle record under strict observation contract.
    
    Supports flat and nested schemas.
    Raises CorruptedObservationDataError if timestamp or geometry is missing or non-finite.
    """
    if not isinstance(obs, dict):
        if raise_on_corrupt:
            raise CorruptedObservationDataError("Obstacle record must be a dict", reason="non_dict_record")
        return None

    data = obs.get("obstacle") if isinstance(obs.get("obstacle"), dict) else obs

    # Extract trackline_id and category
    trackline_id = str(data.get("trackline_id", obs.get("trackline_id", "unknown")))
    category = str(data.get("category", obs.get("category", "unknown")))

    # 1. Extract and validate timestamp_micros
    ts = obs.get("timestamp_micros")
    if ts is None and "key" in obs and isinstance(obs["key"], dict):
        ts = obs["key"].get("timestamp_micros")
    if ts is None and "timestamp_micros" in data:
        ts = data.get("timestamp_micros")

    if ts is None:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) missing required timestamp_micros",
                trackline_id=trackline_id,
                field_name="timestamp_micros",
                reason="missing_timestamp_micros",
            )
        return None

    try:
        ts_val = int(ts)
    except (TypeError, ValueError):
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) has invalid non-integer timestamp: {ts}",
                trackline_id=trackline_id,
                field_name="timestamp_micros",
                reason="invalid_timestamp_type",
            )
        return None

    # 2. Extract and validate Center
    center = data.get("center")
    if not isinstance(center, dict) or "x" not in center or "y" not in center:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) missing center coordinates: {center}",
                trackline_id=trackline_id,
                field_name="center",
                reason="missing_center_coordinates",
            )
        return None
    try:
        ox = float(center["x"])
        oy = float(center["y"])
        oz = float(center.get("z", 0.0))
        if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(oz)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(
                    f"Obstacle (trackline_id={trackline_id}) non-finite center coordinates: ({ox}, {oy}, {oz})",
                    trackline_id=trackline_id,
                    field_name="center",
                    reason="non_finite_center",
                )
            return None
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) non-numeric center coordinates: {ex}",
                trackline_id=trackline_id,
                field_name="center",
                reason="non_numeric_center",
            )
        return None

    # 3. Extract and validate Size
    size = data.get("size")
    if not isinstance(size, dict) or "x" not in size or "y" not in size:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) missing size dimensions: {size}",
                trackline_id=trackline_id,
                field_name="size",
                reason="missing_size",
            )
        return None
    try:
        olength = float(size["x"])
        owidth = float(size["y"])
        oheight = float(size.get("z", 1.5))
        if not (np.isfinite(olength) and np.isfinite(owidth) and np.isfinite(oheight)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(
                    f"Obstacle (trackline_id={trackline_id}) non-finite size: ({olength}, {owidth}, {oheight})",
                    trackline_id=trackline_id,
                    field_name="size",
                    reason="non_finite_size",
                )
            return None
        if olength <= 0.0 or owidth <= 0.0 or oheight <= 0.0:
            if raise_on_corrupt:
                raise CorruptedObservationDataError(
                    f"Obstacle (trackline_id={trackline_id}) non-positive dimensions: l={olength}, w={owidth}, h={oheight}",
                    trackline_id=trackline_id,
                    field_name="size",
                    reason="non_positive_dimensions",
                )
            return None
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) non-numeric size: {ex}",
                trackline_id=trackline_id,
                field_name="size",
                reason="non_numeric_size",
            )
        return None

    # 4. Extract and validate Orientation
    orient = data.get("orientation")
    if not isinstance(orient, dict) or "w" not in orient:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) missing orientation: {orient}",
                trackline_id=trackline_id,
                field_name="orientation",
                reason="missing_orientation",
            )
        return None
    try:
        qw = float(orient["w"])
        qz = float(orient.get("z", 0.0))
        if not (np.isfinite(qw) and np.isfinite(qz)):
            if raise_on_corrupt:
                raise CorruptedObservationDataError(
                    f"Obstacle (trackline_id={trackline_id}) non-finite orientation: ({qw}, {qz})",
                    trackline_id=trackline_id,
                    field_name="orientation",
                    reason="non_finite_orientation",
                )
            return None
        obs_yaw = 2.0 * np.arctan2(qz, qw)
    except (TypeError, ValueError) as ex:
        if raise_on_corrupt:
            raise CorruptedObservationDataError(
                f"Obstacle (trackline_id={trackline_id}) non-numeric orientation: {ex}",
                trackline_id=trackline_id,
                field_name="orientation",
                reason="non_numeric_orientation",
            )
        return None

    return {
        "timestamp_micros": ts_val,
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


_OBSTACLE_INDEX_CACHE: OrderedDict[tuple[int, bool], tuple[Any, Tuple[Dict[int, List[Dict[str, Any]]], np.ndarray, int, int]]] = OrderedDict()


def index_and_filter_obstacles(
    raw_obstacles: Optional[List[Dict[str, Any]]],
    raise_on_corrupt: bool = True,
) -> Tuple[Dict[int, List[Dict[str, Any]]], np.ndarray, int, int]:
    """Normalizes and indexes obstacles by timestamp.
    
    Returns:
        (obs_by_time, all_timestamps_sorted, invalid_count, missing_ts_count)
    """
    if raw_obstacles is None or len(raw_obstacles) == 0:
        return {}, np.array([], dtype=np.int64), 0, 0

    # CF and TTC consume the same immutable obstacle list for every condition
    # of a clip. Keep a small identity-based LRU so repeated conditions do not
    # normalize the same rows again. The original list is retained in each
    # cache entry, preventing id reuse from returning stale data.
    cache_key = (id(raw_obstacles), bool(raise_on_corrupt))
    cached = _OBSTACLE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is raw_obstacles:
        _OBSTACLE_INDEX_CACHE.move_to_end(cache_key)
        return cached[1]

    obs_by_time: Dict[int, List[Dict[str, Any]]] = {}
    invalid_count = 0
    missing_ts_count = 0

    for obs in raw_obstacles:
        try:
            norm = normalize_and_validate_obstacle(obs, raise_on_corrupt=raise_on_corrupt)
            if norm is not None:
                obs_by_time.setdefault(norm["timestamp_micros"], []).append(norm)
            else:
                invalid_count += 1
        except CorruptedObservationDataError as err:
            invalid_count += 1
            if err.field_name == "timestamp_micros":
                missing_ts_count += 1
            if raise_on_corrupt:
                raise

    all_ts = np.array(sorted(obs_by_time.keys()), dtype=np.int64) if obs_by_time else np.array([], dtype=np.int64)
    result = (obs_by_time, all_ts, invalid_count, missing_ts_count)
    _OBSTACLE_INDEX_CACHE[cache_key] = (raw_obstacles, result)
    _OBSTACLE_INDEX_CACHE.move_to_end(cache_key)
    while len(_OBSTACLE_INDEX_CACHE) > 8:
        _OBSTACLE_INDEX_CACHE.popitem(last=False)
    return result


def evaluate_query_coverage(
    query_timestamps_us: np.ndarray,
    obs_by_time: Dict[int, List[Dict[str, Any]]],
    all_obs_timestamps: np.ndarray,
    confirmed_empty_timestamps: Optional[Set[int]] = None,
    half_step_us: int = 50_000,
) -> Tuple[List[ObservationState], int, int, int, float]:
    """Evaluates observation coverage across a set of query timestamps.
    
    Returns:
        (states, observed_count, confirmed_empty_count, missing_count, coverage_ratio)
    """
    confirmed_set = confirmed_empty_timestamps or set()
    states: List[ObservationState] = []
    observed_count = 0
    confirmed_empty_count = 0
    missing_count = 0

    n_queries = len(query_timestamps_us)
    if n_queries == 0:
        return [], 0, 0, 0, 1.0

    for q_ts in query_timestamps_us:
        q_val = int(q_ts)
        # 1. Exact match
        if q_val in obs_by_time:
            states.append(ObservationState.OBSERVED_WITH_OBJECTS)
            observed_count += 1
            continue

        # 2. Nearest neighbor match within half_step tolerance
        matched = False
        if len(all_obs_timestamps) > 0:
            idx = np.searchsorted(all_obs_timestamps, q_val)
            candidates = []
            if idx < len(all_obs_timestamps):
                candidates.append(all_obs_timestamps[idx])
            if idx > 0:
                candidates.append(all_obs_timestamps[idx - 1])
            if candidates:
                best_ts = min(candidates, key=lambda c: abs(c - q_val))
                if abs(best_ts - q_val) <= half_step_us:
                    states.append(ObservationState.OBSERVED_WITH_OBJECTS)
                    observed_count += 1
                    matched = True
                    continue

        # 3. Confirmed empty check
        if q_val in confirmed_set:
            states.append(ObservationState.OBSERVED_EMPTY)
            confirmed_empty_count += 1
            continue

        # 4. Otherwise missing observation
        states.append(ObservationState.MISSING_OBSERVATION)
        missing_count += 1

    coverage_ratio = float((observed_count + confirmed_empty_count) / n_queries)
    return states, observed_count, confirmed_empty_count, missing_count, coverage_ratio


def expand_confirmed_empty_timestamps(
    query_timestamps_us: np.ndarray,
    confirmed_empty_timestamps: Optional[Set[int]] = None,
    confirmed_empty_scene: bool = False,
) -> Set[int]:
    """Combine per-frame attestation with an explicit scene-wide attestation."""
    result = {int(value) for value in (confirmed_empty_timestamps or set())}
    if confirmed_empty_scene:
        result.update(int(value) for value in query_timestamps_us)
    return result


def build_ttc_projection_timestamps(
    timestamps_us: np.ndarray,
    ttc_horizon_s: float,
    step_s: float = 0.2,
) -> np.ndarray:
    """Build the exact deduplicated query grid used by TTC scoring."""
    if len(timestamps_us) == 0:
        return np.array([], dtype=np.int64)
    if ttc_horizon_s < 0 or step_s <= 0:
        raise ValueError("TTC horizon must be non-negative and step must be positive")
    dt_proj_list = [round(float(dt), 2) for dt in np.arange(0.0, float(ttc_horizon_s) + 1e-6, step_s)]
    values = {
        int(timestamp) + int(round(dt * 1_000_000))
        for timestamp in timestamps_us
        for dt in dt_proj_list
    }
    return np.asarray(sorted(values), dtype=np.int64)
