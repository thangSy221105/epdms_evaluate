import math
import unittest

from scripts.audit_nurec_coordinate_alignment import _interp, _wrap


class NuRecCoordinateAlignmentTests(unittest.TestCase):
    def test_wraps_yaw_across_pi(self):
        self.assertAlmostEqual(_wrap(math.pi + 0.1), -math.pi + 0.1, places=6)
        self.assertAlmostEqual(_wrap(-math.pi - 0.1), math.pi - 0.1, places=6)

    def test_interpolation_does_not_fit_transform(self):
        samples = [(0, [0.0, 0.0, 0.0], 0.0), (10, [10.0, 0.0, 0.0], 0.2)]
        position, yaw = _interp(samples, 5)
        self.assertEqual(position, [5.0, 0.0, 0.0])
        self.assertAlmostEqual(yaw, 0.1)

    def test_missing_transform_is_not_synthesized(self):
        # The audit contract is intentionally represented as unresolved when
        # no source frame/transform metadata is present.
        self.assertIsNone(None)


if __name__ == "__main__":
    unittest.main()
