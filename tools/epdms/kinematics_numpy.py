"""Kinematic analysis and comfort evaluation using NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ComfortThresholds:
    max_jerk_magnitude: float = 8.37
    max_lat_accel: float = 4.89
    min_lon_accel: float = -4.05
    max_lon_accel: float = 2.40
    max_lon_jerk: float = 4.13
    max_yaw_accel: float = 1.93
    max_yaw_rate: float = 0.95


@dataclass
class KinematicProfile:
    timestamps: np.ndarray
    velocities: np.ndarray
    longitudinal_accels: np.ndarray
    lateral_accels: np.ndarray
    jerk_magnitudes: np.ndarray
    longitudinal_jerks: np.ndarray
    yaw_rates: np.ndarray
    yaw_accels: np.ndarray

    max_abs_longitudinal_accel: float
    max_abs_lateral_accel: float
    max_jerk_magnitude: float
    max_abs_longitudinal_jerk: float
    max_abs_yaw_rate: float
    max_abs_yaw_acceleration: float

    comfort_pass: bool
    failure_reasons: List[str] = field(default_factory=list)


def compute_kinematics(
    x: np.ndarray,
    y: np.ndarray,
    headings: np.ndarray,
    dt: float = 0.1,
    thresholds: ComfortThresholds = ComfortThresholds(),
) -> KinematicProfile:
    """Computes full kinematic profile from a sequence of waypoints at fixed dt."""
    n = len(x)
    if n < 2:
        raise ValueError("Trajectory must have at least 2 points to compute kinematics")

    # Velocities
    dx = np.diff(x)
    dy = np.diff(y)
    speeds = np.hypot(dx, dy) / dt
    speeds = np.insert(speeds, 0, speeds[0])

    # Accelerations
    accels_x = np.gradient(dx / dt, dt)
    accels_x = np.insert(accels_x, 0, accels_x[0])
    accels_y = np.gradient(dy / dt, dt)
    accels_y = np.insert(accels_y, 0, accels_y[0])

    # Yaw rate & acceleration (heading is unwrapped)
    yaw_rates = np.gradient(headings, dt)
    yaw_accels = np.gradient(yaw_rates, dt)

    # Longitudinal & Lateral acceleration in body frame
    # lon_accel = a_x * cos(theta) + a_y * sin(theta)
    # lat_accel = -a_x * sin(theta) + a_y * cos(theta)
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    lon_accels = accels_x * cos_h + accels_y * sin_h
    lat_accels = -accels_x * sin_h + accels_y * cos_h

    # Jerks
    lon_jerks = np.gradient(lon_accels, dt)
    lat_jerks = np.gradient(lat_accels, dt)
    jerk_magnitudes = np.hypot(lon_jerks, lat_jerks)

    max_abs_lon_accel = float(np.max(np.abs(lon_accels)))
    max_abs_lat_accel = float(np.max(np.abs(lat_accels)))
    max_jerk_mag = float(np.max(jerk_magnitudes))
    max_abs_lon_jerk = float(np.max(np.abs(lon_jerks)))
    max_abs_yaw_rate = float(np.max(np.abs(yaw_rates)))
    max_abs_yaw_accel = float(np.max(np.abs(yaw_accels)))

    # Check against NAVSIM comfort limits
    failure_reasons = []
    if max_jerk_mag > thresholds.max_jerk_magnitude:
        failure_reasons.append(f"jerk_magnitude_{max_jerk_mag:.2f}_exceeds_{thresholds.max_jerk_magnitude}")
    if max_abs_lat_accel > thresholds.max_lat_accel:
        failure_reasons.append(f"lateral_accel_{max_abs_lat_accel:.2f}_exceeds_{thresholds.max_lat_accel}")
    if np.min(lon_accels) < thresholds.min_lon_accel:
        failure_reasons.append(f"braking_accel_{np.min(lon_accels):.2f}_below_{thresholds.min_lon_accel}")
    if np.max(lon_accels) > thresholds.max_lon_accel:
        failure_reasons.append(f"longitudinal_accel_{np.max(lon_accels):.2f}_exceeds_{thresholds.max_lon_accel}")
    if max_abs_lon_jerk > thresholds.max_lon_jerk:
        failure_reasons.append(f"lon_jerk_{max_abs_lon_jerk:.2f}_exceeds_{thresholds.max_lon_jerk}")
    if max_abs_yaw_rate > thresholds.max_yaw_rate:
        failure_reasons.append(f"yaw_rate_{max_abs_yaw_rate:.2f}_exceeds_{thresholds.max_yaw_rate}")
    if max_abs_yaw_accel > thresholds.max_yaw_accel:
        failure_reasons.append(f"yaw_accel_{max_abs_yaw_accel:.2f}_exceeds_{thresholds.max_yaw_accel}")

    return KinematicProfile(
        timestamps=np.arange(n) * dt,
        velocities=speeds,
        longitudinal_accels=lon_accels,
        lateral_accels=lat_accels,
        jerk_magnitudes=jerk_magnitudes,
        longitudinal_jerks=lon_jerks,
        yaw_rates=yaw_rates,
        yaw_accels=yaw_accels,
        max_abs_longitudinal_accel=max_abs_lon_accel,
        max_abs_lateral_accel=max_abs_lat_accel,
        max_jerk_magnitude=max_jerk_mag,
        max_abs_longitudinal_jerk=max_abs_lon_jerk,
        max_abs_yaw_rate=max_abs_yaw_rate,
        max_abs_yaw_acceleration=max_abs_yaw_accel,
        comfort_pass=(len(failure_reasons) == 0),
        failure_reasons=failure_reasons,
    )
