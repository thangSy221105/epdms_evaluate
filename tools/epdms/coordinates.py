"""Coordinate transformation, heading derivation, and angle wrapping using NumPy."""

from __future__ import annotations

import numpy as np


def wrap_angle(angles: np.ndarray | float) -> np.ndarray | float:
    """Wraps angles to [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def derive_heading_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes heading angle for a trajectory using finite differences and unwraps it.
    
    x, y: 1D arrays of trajectory coordinates.
    Returns: 1D array of unwrapped heading angles in radians.
    """
    n = len(x)
    if n < 2:
        return np.zeros_like(x)

    headings = np.zeros(n, dtype=float)
    # Forward difference at t=0
    headings[0] = np.arctan2(y[1] - y[0], x[1] - x[0])
    # Backward difference at t=n-1
    headings[-1] = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])

    if n > 2:
        # Central difference for interior points
        dx = x[2:] - x[:-2]
        dy = y[2:] - y[:-2]
        # Check for stationary consecutive points
        norms = np.hypot(dx, dy)
        eps = 1e-6
        interior = np.arctan2(dy, dx)
        for i in range(1, n - 1):
            if norms[i - 1] > eps:
                headings[i] = interior[i - 1]
            else:
                headings[i] = headings[i - 1]

    # Unwrap to avoid +/- pi boundary jumping
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
