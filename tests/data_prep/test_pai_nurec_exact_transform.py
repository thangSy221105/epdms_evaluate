import unittest

from scripts.audit_pai_nurec_exact_transform import inv, obj_pose, _mm, _pose


class ExactTransformTests(unittest.TestCase):
    def test_se3_translation(self):
        t = _pose([2, 3, 4], [0, 0, 0, 1])
        o = _pose([1, 2, 3], [0, 0, 0, 1])
        self.assertEqual(_mm(t, o)[0][3], 3)
        self.assertEqual(_mm(t, o)[1][3], 5)
        self.assertEqual(_mm(t, o)[2][3], 7)

    def test_inverse_direction_is_not_same_for_translation(self):
        t = _pose([2, 0, 0], [0, 0, 0, 1])
        o = _pose([1, 0, 0], [0, 0, 0, 1])
        self.assertNotEqual(_mm(t, o)[0][3], _mm(inv(t), o)[0][3])

    def test_identity_rebase(self):
        identity = _pose([0, 0, 0], [0, 0, 0, 1])
        self.assertEqual(inv(identity), identity)

    def test_quaternion_order_is_xyzw(self):
        row = {"center_x": 1, "center_y": 2, "center_z": 3,
               "orientation_x": 0.1, "orientation_y": 0.2,
               "orientation_z": 0.3, "orientation_w": 0.9}
        self.assertEqual(obj_pose(row)[0][3], 1)
        self.assertEqual(obj_pose(row)[1][3], 2)
        self.assertEqual(obj_pose(row)[2][3], 3)


if __name__ == "__main__":
    unittest.main()
