"""Coordinate transformation, heading derivation, and angle wrapping using NumPy."""

from __future__ import annotations

import numpy as np


def wrap_angle(angles: np.ndarray | float) -> np.ndarray | float:
    """Wraps angles to [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def derive_heading_from_xy(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes heading angle for a trajectory using finite differences and unwraps it.
    
    Correctly preserves heading when vehicle is stationary at the start, middle, or end,
    guaranteeing rotation invariance.
    """
    n = len(x)
    if n < 2:
        return np.zeros_like(x)

    dx_fwd = np.diff(x)
    dy_fwd = np.diff(y)
    step_norms = np.hypot(dx_fwd, dy_fwd)
    moving_steps = step_norms > eps

    # If the vehicle never moves throughout the trajectory, heading is 0
    if not np.any(moving_steps):
        return np.zeros(n, dtype=float)

    segment_headings = np.zeros(n - 1, dtype=float)
    valid_indices = np.where(moving_steps)[0]

    for idx in valid_indices:
        segment_headings[idx] = np.arctan2(dy_fwd[idx], dx_fwd[idx])

    # Forward fill stationary segments
    last_valid = segment_headings[valid_indices[0]]
    for i in range(n - 1):
        if moving_steps[i]:
            last_valid = segment_headings[i]
        else:
            segment_headings[i] = last_valid

    # Backward fill stationary segments before the first moving step
    first_valid = segment_headings[valid_indices[0]]
    for i in range(valid_indices[0] - 1, -1, -1):
        segment_headings[i] = first_valid

    headings = np.zeros(n, dtype=float)
    headings[0] = segment_headings[0]
    headings[-1] = segment_headings[-1]

    for i in range(1, n - 1):
        h_prev = segment_headings[i - 1]
        h_next = segment_headings[i]
        avg_x = np.cos(h_prev) + np.cos(h_next)
        avg_y = np.sin(h_prev) + np.sin(h_next)
        if np.hypot(avg_x, avg_y) > 1e-4:
            headings[i] = np.arctan2(avg_y, avg_x)
        else:
            headings[i] = h_prev

    return np.unwrap(headings)


def local_to_global(
    x_local: np.ndarray,
    y_local: np.ndarray,
    heading_local: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_heading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms local vehicle coordinates (rear axle frame) to global/world coordinates."""
    c = np.cos(origin_heading)
    s = np.sin(origin_heading)
    x_global = origin_x + x_local * c - y_local * s
    y_global = origin_y + x_local * s + y_local * c
    heading_global = wrap_angle(origin_heading + heading_local)
    return x_global, y_global, heading_global


def global_to_local(
    x_global: np.ndarray,
    y_global: np.ndarray,
    heading_global: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_heading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms global/world coordinates to local vehicle frame at origin."""
    dx = x_global - origin_x
    dy = y_global - origin_y
    c = np.cos(origin_heading)
    s = np.sin(origin_heading)
    x_local = dx * c + dy * s
    y_local = -dx * s + dy * c
    heading_local = wrap_angle(heading_global - origin_heading)
    return x_local, y_local, heading_local
