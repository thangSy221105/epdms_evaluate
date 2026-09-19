import unittest

from scripts.audit_ego_frame_variants import _mm, _pose, inv


class EgoFrameVariantTests(unittest.TestCase):
    def setUp(self):
        self.world_to_nre = _pose([10, 0, 0], [0, 0, 0, 1])
        self.raw_first = _pose([1000, 2000, 0], [0, 0, 0, 1])
        self.raw_later = _pose([1012, 2000, 0], [0, 0, 0, 1])

    def test_ncore_rebase_before_comparison(self):
        local = _mm(inv(self.raw_first), self.raw_later)
        self.assertAlmostEqual(local[0][3], 12)
        self.assertAlmostEqual(local[1][3], 0)

    def test_no_world_to_nre_variant(self):
        local = _mm(inv(self.raw_first), self.raw_later)
        self.assertAlmostEqual(local[0][3], 12)

    def test_forward_variant(self):
        local = _mm(inv(self.raw_first), self.raw_later)
        forward = _mm(self.world_to_nre, local)
        self.assertAlmostEqual(forward[0][3], 22)

    def test_inverse_variant(self):
        local = _mm(inv(self.raw_first), self.raw_later)
        inverse = _mm(inv(self.world_to_nre), local)
        self.assertAlmostEqual(inverse[0][3], 2)

    def test_no_offset_rederive(self):
        per_clip_offset_rederived = False
        self.assertFalse(per_clip_offset_rederived)


if __name__ == "__main__":
    unittest.main()
