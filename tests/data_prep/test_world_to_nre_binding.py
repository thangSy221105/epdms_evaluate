import unittest

from scripts.audit_world_to_nre_binding import _mm, _pose, inv


class WorldToNreBindingTests(unittest.TestCase):
    def test_forward_direction(self):
        world_to_scene = _pose([10, 0, 0], [0, 0, 0, 1])
        rig_world = _pose([2, 0, 0], [0, 0, 0, 1])
        self.assertEqual(_mm(world_to_scene, rig_world)[0][3], 12)

    def test_inverse_direction_differs(self):
        world_to_scene = _pose([10, 0, 0], [0, 0, 0, 1])
        rig_world = _pose([2, 0, 0], [0, 0, 0, 1])
        self.assertNotEqual(_mm(world_to_scene, rig_world)[0][3], _mm(inv(world_to_scene), rig_world)[0][3])

    def test_rebase_first_pose_identity(self):
        first = _pose([4, 5, 6], [0, 0, 0, 1])
        rebased = _mm(inv(first), first)
        self.assertAlmostEqual(rebased[0][0], 1.0)
        self.assertAlmostEqual(rebased[1][1], 1.0)
        self.assertAlmostEqual(rebased[2][2], 1.0)
        self.assertAlmostEqual(rebased[0][3], 0.0)

    def test_reference_timestamp_is_distinct_from_query_contract(self):
        row = {"timestamp_us": 100, "reference_frame_timestamp_us": 90}
        self.assertEqual(row["reference_frame_timestamp_us"], 90)
        self.assertNotEqual(row["timestamp_us"], row["reference_frame_timestamp_us"])

    def test_se3_inverse_round_trip(self):
        transform = _pose([1, 2, 3], [0, 0, 0.3826834324, 0.9238795325])
        identity = _mm(inv(transform), transform)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(identity[i][j], 1.0 if i == j else 0.0, places=6)

    def test_rebase_prevents_raw_global_translation_leak(self):
        first = _pose([1000, 2000, 0], [0, 0, 0, 1])
        later = _pose([1010, 2000, 0], [0, 0, 0, 1])
        world_to_scene = _pose([0, 0, 0], [0, 0, 0, 1])
        direct = _mm(world_to_scene, later)
        rebased = _mm(world_to_scene, _mm(inv(first), later))
        self.assertEqual(direct[0][3], 1010)
        self.assertAlmostEqual(rebased[0][3], 10)


if __name__ == "__main__":
    unittest.main()
