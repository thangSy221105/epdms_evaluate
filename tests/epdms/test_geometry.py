"""Unit tests for geometry algorithms (SAT collision and Point-in-Polygon)."""

import unittest
import numpy as np
from tools.epdms.geometry_numpy import (
    get_oriented_box_corners,
    sat_box_intersection,
    point_in_polygon_ray_casting,
    project_point_onto_polyline,
)


class TestGeometry(unittest.TestCase):
    def test_sat_separated_boxes(self):
        # Box A at origin (0, 0), length 4, width 2
        box_a = get_oriented_box_corners(0.0, 0.0, 0.0, 4.0, 2.0)
        # Box B far away at (10, 10)
        box_b = get_oriented_box_corners(10.0, 10.0, 0.0, 4.0, 2.0)
        is_col, clearance = sat_box_intersection(box_a, box_b, touch_is_collision=True)
        self.assertFalse(is_col)
        self.assertGreater(clearance, 0.0)

    def test_sat_overlapping_boxes(self):
        # Box A at origin
        box_a = get_oriented_box_corners(0.0, 0.0, 0.0, 4.0, 2.0)
        # Box B slightly shifted (1.0, 0.5)
        box_b = get_oriented_box_corners(1.0, 0.5, 0.0, 4.0, 2.0)
        is_col, clearance = sat_box_intersection(box_a, box_b, touch_is_collision=True)
        self.assertTrue(is_col)
        self.assertLessEqual(clearance, 0.0)

    def test_sat_rotated_boxes(self):
        # Box A at origin, Box B rotated 45 deg, overlapping corner
        box_a = get_oriented_box_corners(0.0, 0.0, 0.0, 4.0, 2.0)
        box_b = get_oriented_box_corners(2.5, 0.0, np.pi / 4.0, 2.0, 2.0)
        is_col, _ = sat_box_intersection(box_a, box_b, touch_is_collision=True)
        self.assertTrue(is_col)

    def test_point_in_polygon(self):
        # Unit square [0, 2] x [0, 2]
        poly = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        self.assertTrue(point_in_polygon_ray_casting(1.0, 1.0, poly))
        self.assertFalse(point_in_polygon_ray_casting(3.0, 3.0, poly))
        self.assertFalse(point_in_polygon_ray_casting(-1.0, 1.0, poly))

    def test_project_point_onto_polyline(self):
        polyline = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        # Point at (5.0, 1.0) projects to s = 5.0, distance = 1.0
        s, dist = project_point_onto_polyline(np.array([5.0, 1.0]), polyline)
        self.assertAlmostEqual(s, 5.0, places=4)
        self.assertAlmostEqual(dist, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
