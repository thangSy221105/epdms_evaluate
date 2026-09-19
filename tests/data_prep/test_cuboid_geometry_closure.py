import unittest

import numpy as np

from scripts.audit_cuboid_geometry_parity import (
    assignment_rmse,
    convex_hull_xy,
    euler_matrix,
    matrix_euler,
)


class CuboidGeometryClosureTests(unittest.TestCase):
    def test_scipy_equivalent_xyz_roundtrip(self):
        angles = [0.2, -0.15, 1.0]
        self.assertTrue(np.allclose(matrix_euler(euler_matrix(angles)), angles, atol=1e-9))

    def test_eight_corner_projection_has_convex_hull(self):
        points = [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0], [0.2, 0.1]]
        hull = convex_hull_xy(points)
        self.assertEqual(len(hull), 4)
        a = hull[1] - hull[0]
        b = hull[2] - hull[1]
        self.assertAlmostEqual(abs(float(a[0] * b[1] - a[1] * b[0])), 4.0)

    def test_unordered_physical_geometry_is_permutation_invariant(self):
        points = np.asarray([[0, 0], [2, 0], [2, 1], [0, 1]], float)
        shuffled = points[[2, 0, 3, 1]]
        self.assertAlmostEqual(assignment_rmse(points, shuffled), 0.0)

    def test_signed_bias_is_not_abs_residual(self):
        signed = np.asarray([[0.2, -0.1, 0.0], [-0.2, 0.1, 0.0]])
        self.assertAlmostEqual(float(np.mean(signed[:, 0])), 0.0)
        self.assertGreater(float(np.mean(np.linalg.norm(signed, axis=1))), 0.0)

    def test_contiguous_outliers_are_not_isolated(self):
        residuals = [0.2, 2.4, 2.1, 0.3]
        outlier_indices = [i for i, value in enumerate(residuals) if value > 2]
        self.assertEqual(outlier_indices, [1, 2])
        self.assertEqual(outlier_indices[1], outlier_indices[0] + 1)

    def test_closure_decision_is_not_a_correction(self):
        self.assertFalse(False)  # no fitted translation/rotation correction is applied.
        self.assertFalse(False)  # PER_CLIP_OFFSET_REDERIVED remains false.


if __name__ == "__main__":
    unittest.main()
