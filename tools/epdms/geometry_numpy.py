"""Pure NumPy geometric algorithms: SAT collision detection and Point-in-Polygon."""

from __future__ import annotations

import numpy as np


def get_oriented_box_corners(
    center_x: float,
    center_y: float,
    heading_rad: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Returns 4 corners of an oriented bounding box as shape (4, 2)."""
    hl = length / 2.0
    hw = width / 2.0
    # Local box corners in body frame
    local_corners = np.array([
        [hl, hw],
        [-hl, hw],
        [-hl, -hw],
        [hl, -hw],
    ], dtype=float)

    c = np.cos(heading_rad)
    s = np.sin(heading_rad)
    rot = np.array([[c, -s], [s, c]])

    world_corners = (rot @ local_corners.T).T
    world_corners[:, 0] += center_x
    world_corners[:, 1] += center_y
    return world_corners


def get_ego_box_corners(
    rear_axle_x: float,
    rear_axle_y: float,
    heading_rad: float,
    length: float,
    width: float,
    rear_axle_to_center: float,
) -> np.ndarray:
    """Calculates corners of the ego vehicle given its rear axle location."""
    c = np.cos(heading_rad)
    s = np.sin(heading_rad)
    center_x = rear_axle_x + rear_axle_to_center * c
    center_y = rear_axle_y + rear_axle_to_center * s
    return get_oriented_box_corners(center_x, center_y, heading_rad, length, width)


def sat_box_intersection(
    box_a: np.ndarray,
    box_b: np.ndarray,
    touch_is_collision: bool = True,
) -> tuple[bool, float]:
    """Separating Axis Theorem (SAT) for two 2D convex quadrilaterals (oriented rectangles).
    
    box_a, box_b: shape (4, 2) arrays of corner coordinates.
    Returns: (is_intersecting, minimum_clearance).
    If intersecting, minimum_clearance <= 0 (or penetration depth).
    If separated, minimum_clearance > 0 (distance along best separating axis).
    """
    edges_a = np.roll(box_a, -1, axis=0) - box_a
    edges_b = np.roll(box_b, -1, axis=0) - box_b

    # For rectangles, first 2 edges give orthogonal normals
    axes = np.vstack([
        np.array([-edges_a[0, 1], edges_a[0, 0]]),
        np.array([-edges_a[1, 1], edges_a[1, 0]]),
        np.array([-edges_b[0, 1], edges_b[0, 0]]),
        np.array([-edges_b[1, 1], edges_b[1, 0]]),
    ])

    # Normalize axes
    norms = np.hypot(axes[:, 0], axes[:, 1])
    # Avoid zero division
    valid_mask = norms > 1e-8
    if not np.all(valid_mask):
        axes = axes[valid_mask]
        norms = norms[valid_mask]
    axes = axes / norms[:, None]

    min_clearance = -float("inf")
    for axis in axes:
        proj_a = box_a @ axis
        proj_b = box_b @ axis

        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        gap = max(min_a - max_b, min_b - max_a)
        if touch_is_collision:
            if gap > 0:  # Strictly separated
                exact_dist = compute_exact_box_distance(box_a, box_b)
                return False, exact_dist
        else:
            if gap >= 0:  # Touching or separated
                exact_dist = compute_exact_box_distance(box_a, box_b)
                return False, exact_dist

        if gap > min_clearance:
            min_clearance = gap

    # If no separating axis found, boxes intersect
    return True, min_clearance


def is_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, tol: float = 1e-7) -> bool:
    """Checks if point (px, py) lies on line segment (ax, ay)-(bx, by) within tolerance."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    # Cross product for collinearity
    cross = abs(apx * aby - apy * abx)
    seg_len = np.hypot(abx, aby)
    if seg_len < tol:
        return np.hypot(px - ax, py - ay) < tol
    if cross / seg_len > tol:
        return False

    # Dot product for bounding segment
    dot = apx * abx + apy * aby
    if dot < -tol or dot > seg_len * seg_len + tol:
        return False
    return True


def point_in_polygon_ray_casting(px: float, py: float, polygon: np.ndarray, tol: float = 1e-7) -> bool:
    """Tests if (px, py) is inside or on the boundary of a 2D polygon with consistent boundary handling."""
    if not (np.isfinite(px) and np.isfinite(py)):
        raise ValueError(f"Non-finite point coordinates: ({px}, {py})")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("Non-finite polygon coordinates")

    n = len(polygon)
    if n < 3:
        return False

    # 1. Boundary check: If point is on any edge, it is inside
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if is_point_on_segment(px, py, p1[0], p1[1], p2[0], p2[1], tol=tol):
            return True

    # 2. Standard ray casting
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Computes minimum Euclidean distance from point (px, py) to segment AB."""
    abx = bx - ax
    aby = by - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / seg_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return float(np.hypot(px - proj_x, py - proj_y))


def compute_exact_box_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes exact minimum Euclidean distance between two disjoint convex 2D boxes."""
    min_dist = float("inf")
    # Distance from vertices of A to edges of B
    for i in range(len(box_a)):
        v = box_a[i]
        for j in range(len(box_b)):
            e1 = box_b[j]
            e2 = box_b[(j + 1) % len(box_b)]
            d = point_to_segment_dist(v[0], v[1], e1[0], e1[1], e2[0], e2[1])
            if d < min_dist:
                min_dist = d
    # Distance from vertices of B to edges of A
    for i in range(len(box_b)):
        v = box_b[i]
        for j in range(len(box_a)):
            e1 = box_a[j]
            e2 = box_a[(j + 1) % len(box_a)]
            d = point_to_segment_dist(v[0], v[1], e1[0], e1[1], e2[0], e2[1])
            if d < min_dist:
                min_dist = d
    return min_dist


def points_in_any_polygon(points: np.ndarray, polygons: list[np.ndarray], tol: float = 1e-7) -> np.ndarray:
    """Tests an array of points shape (N, 2) against a list of polygons.
    Returns boolean array of shape (N,).
    """
    n_pts = len(points)
    if n_pts == 0 or len(polygons) == 0:
        return np.zeros(n_pts, dtype=bool)

    if not np.all(np.isfinite(points)):
        raise ValueError("Non-finite point coordinates in points_in_any_polygon")

    # Reject points outside each polygon's axis-aligned bounds before the
    # exact ray-casting test.  This is an exact-preserving acceleration: a
    # point can only be inside a polygon if it is inside its bounding box.
    prepared = []
    for poly in polygons:
        polygon = np.asarray(poly)
        if polygon.ndim != 2 or polygon.shape[1] < 2 or len(polygon) == 0:
            prepared.append((polygon, None))
            continue
        prepared.append((
            polygon,
            (
                float(np.min(polygon[:, 0])),
                float(np.max(polygon[:, 0])),
                float(np.min(polygon[:, 1])),
                float(np.max(polygon[:, 1])),
            ),
        ))

    inside_mask = np.zeros(n_pts, dtype=bool)
    for pt_idx in range(n_pts):
        px, py = points[pt_idx, 0], points[pt_idx, 1]
        for poly, bounds in prepared:
            if bounds is None:
                continue
            min_x, max_x, min_y, max_y = bounds
            if px < min_x or px > max_x or py < min_y or py > max_y:
                continue
            if point_in_polygon_ray_casting(px, py, poly, tol=tol):
                inside_mask[pt_idx] = True
                break
    return inside_mask


def project_point_onto_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    """Projects a 2D point onto a polyline.
    Returns:
      (arc_length_along_polyline, lateral_distance_to_polyline)
    """
    if len(polyline) < 2:
        return 0.0, 0.0

    p = point[:2]
    cum_dist = 0.0
    min_dist_sq = float("inf")
    best_s = 0.0

    for i in range(len(polyline) - 1):
        a = polyline[i, :2]
        b = polyline[i + 1, :2]
        ab = b - a
        seg_len_sq = np.dot(ab, ab)
        if seg_len_sq < 1e-8:
            continue
        seg_len = np.sqrt(seg_len_sq)

        t = np.clip(np.dot(p - a, ab) / seg_len_sq, 0.0, 1.0)
        proj = a + t * ab
        dist_sq = np.sum((p - proj) ** 2)

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_s = cum_dist + t * seg_len

        cum_dist += seg_len

    return float(best_s), float(np.sqrt(min_dist_sq))
