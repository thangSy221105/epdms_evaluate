"""Authoritative NuRec map transform for the frozen EPDMS coordinate contract.

NuRec ClipGT map geometry is stored in the same local-world convention as the
NuRec rig trajectory.  The scorer consumes EGO_AT_T0, therefore map points are
rebased with the inverse of the interpolated ``T_rig_world(t0)`` pose.

This module deliberately does not use ``world_to_nre`` and does not estimate a
correction from predictions or ground truth.  It is a deterministic adapter
around the accepted time/coordinate contracts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .map_loader import inspect_clip_map_status


class MapTransformError(ValueError):
    """Raised when a verified map transform cannot be constructed."""


def _normalize_quaternion(q: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if not math.isfinite(n) or n == 0:
        raise MapTransformError("invalid quaternion in rig trajectory")
    return [float(v) / n for v in q]


def _rotation_from_quaternion(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quaternion_from_rotation(r: np.ndarray) -> list[float]:
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return _normalize_quaternion(q)


def _slerp(a: Sequence[float], b: Sequence[float], amount: float) -> list[float]:
    qa = _normalize_quaternion(a)
    qb = _normalize_quaternion(b)
    dot = sum(x * y for x, y in zip(qa, qb))
    if dot < 0:
        qb = [-v for v in qb]
        dot = -dot
    if dot > 0.9995:
        return _normalize_quaternion([x + amount * (y - x) for x, y in zip(qa, qb)])
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    left = math.sin((1.0 - amount) * theta) / sin_theta
    right = math.sin(amount * theta) / sin_theta
    return _normalize_quaternion([left * x + right * y for x, y in zip(qa, qb)])


def _pose_from_matrix(matrix: Sequence[Sequence[float]]) -> tuple[np.ndarray, list[float]]:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise MapTransformError("invalid 4x4 rig pose")
    return arr[:3, 3].copy(), _quaternion_from_rotation(arr[:3, :3])


def _matrix_from_pose(position: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = _rotation_from_quaternion(quaternion)
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def interpolate_rig_world_pose(samples: Sequence[tuple[int, Sequence[Sequence[float]]]], timestamp_us: int) -> np.ndarray:
    """Interpolate a rig→world pose with linear translation and quaternion SLERP."""

    if not samples:
        raise MapTransformError("rig trajectory has no poses")
    if timestamp_us < samples[0][0] or timestamp_us > samples[-1][0]:
        raise MapTransformError(
            f"t0 outside rig trajectory: {timestamp_us} not in [{samples[0][0]}, {samples[-1][0]}]"
        )
    for left, right in zip(samples, samples[1:]):
        if timestamp_us == left[0]:
            return np.asarray(left[1], dtype=float)
        if left[0] < timestamp_us <= right[0]:
            amount = (timestamp_us - left[0]) / float(right[0] - left[0])
            lp, lq = _pose_from_matrix(left[1])
            rp, rq = _pose_from_matrix(right[1])
            position = lp + amount * (rp - lp)
            return _matrix_from_pose(position, _slerp(lq, rq, amount))
    return np.asarray(samples[-1][1], dtype=float)


def load_rig_world_samples(clip_dir: Path) -> list[tuple[int, list[list[float]]]]:
    path = clip_dir / "rig_trajectories.json"
    if not path.is_file():
        raise MapTransformError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    trajectories = data.get("rig_trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise MapTransformError("rig_trajectories.json has no rig_trajectories")
    item = trajectories[0]
    timestamps = item.get("T_rig_world_timestamps_us")
    poses = item.get("T_rig_worlds")
    if not isinstance(timestamps, list) or not isinstance(poses, list) or len(timestamps) != len(poses):
        raise MapTransformError("T_rig_worlds and timestamps are missing or have different lengths")
    return [(int(timestamp), [[float(value) for value in row] for row in pose]) for timestamp, pose in zip(timestamps, poses)]


def invert_se3(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        raise MapTransformError("SE(3) matrix must be 4x4")
    rotation = arr[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ arr[:3, 3]
    return result


def transform_polygon_to_ego_t0(polygon: np.ndarray, world_to_ego_t0: np.ndarray) -> np.ndarray:
    """Apply the verified world→EGO_AT_T0 transform to a 2D polygon."""

    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 3:
        raise MapTransformError("polygon must contain at least three x/y points")
    homogeneous = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points)), np.ones(len(points))])
    transformed = (world_to_ego_t0 @ homogeneous.T).T
    return transformed[:, :2]


def load_transformed_map_for_clip(
    filtered_dir: Path,
    clip_id: str,
    nurec_t0_us: int,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load and transform a clip's lane/intersection polygons.

    The returned status is strict: partial/corrupt map sources do not become
    scoreable polygons. Raw parquet is read only; no source file is modified.
    """

    status = inspect_clip_map_status(filtered_dir, clip_id)
    if strict and not status.get("usable_for_strict_scoring", False):
        return {**status, "transformed_lane_polygons": [], "transformed_intersection_polygons": [], "transform_status": "BLOCKED_BY_MAP_SOURCE"}
    samples = load_rig_world_samples(filtered_dir / clip_id)
    pose_at_t0 = interpolate_rig_world_pose(samples, int(nurec_t0_us))
    world_to_ego = invert_se3(pose_at_t0)
    lane = [transform_polygon_to_ego_t0(poly, world_to_ego) for poly in status.get("lane_polygons", [])]
    intersection = [transform_polygon_to_ego_t0(poly, world_to_ego) for poly in status.get("intersection_polygons", [])]
    return {
        **status,
        "transformed_lane_polygons": lane,
        "transformed_intersection_polygons": intersection,
        "transform_status": "VERIFIED",
        "map_source_frame": "NCORE_LOCAL_WORLD",
        "map_target_frame": "EGO_AT_T0",
        "nurec_t0_us": int(nurec_t0_us),
        "t_rig_world_at_t0": pose_at_t0.tolist(),
        "world_to_ego_t0": world_to_ego.tolist(),
        "world_to_nre_used": False,
        "fitted_map_correction_used": False,
    }

