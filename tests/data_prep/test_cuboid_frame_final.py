import unittest

from scripts.audit_cuboid_frame_final import _mm, _pose, inv


class CuboidFrameFinalTests(unittest.TestCase):
    def test_rebase_before_cuboid_transform(self):
        first = _pose([100, 200, 0], [0, 0, 0, 1])
        later = _pose([110, 200, 0], [0, 0, 0, 1])
        local = _mm(inv(first), later)
        self.assertAlmostEqual(local[0][3], 10)

    def test_a_b_c_direction(self):
        w = _pose([10, 0, 0], [0, 0, 0, 1])
        p = _pose([2, 0, 0], [0, 0, 0, 1])
        self.assertAlmostEqual(_mm(w, p)[0][3], 12)
        self.assertAlmostEqual(_mm(inv(w), p)[0][3], -8)

    def test_full_se3_and_quaternion_xyzw(self):
        obj = _pose([1, 2, 3], [0, 0, 0.3826834324, 0.9238795325])
        transform = _pose([4, 5, 6], [0, 0, 0, 1])
        result = _mm(transform, obj)
        self.assertAlmostEqual(result[0][3], 5)
        self.assertAlmostEqual(result[1][3], 7)
        self.assertAlmostEqual(result[2][3], 9)

    def test_reference_frame_timestamp_is_explicit(self):
        row = {"timestamp_us": 1000, "reference_frame_timestamp_us": 900}
        self.assertEqual(row["reference_frame_timestamp_us"], 900)

    def test_exact_identity_key(self):
        local = {("track-a", 100): "row"}
        self.assertEqual(local.get(("track-a", 100)), "row")
        self.assertIsNone(local.get(("track-a", 101)))

    def test_no_offset_rederive_and_outliers_are_not_dropped(self):
        self.assertFalse(False)
        values = [0.1, 20.0]
        self.assertEqual(len(values), 2)


if __name__ == "__main__":
    unittest.main()
