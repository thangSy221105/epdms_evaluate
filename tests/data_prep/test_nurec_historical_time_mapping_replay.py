import unittest

from scripts.replay_nurec_historical_time_mapping_full300 import (
    HISTORICAL_OFFSETS,
    historical_pairs,
)


class HistoricalReplayContractTests(unittest.TestCase):
    def test_known_historical_clip_reproduces_exact_offset(self):
        pai = [{"timestamp": 0}, {"timestamp": 20_000_000}]
        nurec = [HISTORICAL_OFFSETS["028508ba-ef59-48d3-a95b-94eb92e3b063"], HISTORICAL_OFFSETS["028508ba-ef59-48d3-a95b-94eb92e3b063"] + 20_000_000]
        pairs, diagnostics = historical_pairs(pai, nurec)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(diagnostics["offsets_us"], [3_033_653_000, 3_033_653_000])
        self.assertTrue(diagnostics["offset_equal"])

    def test_two_semantic_boundaries_are_required(self):
        with self.assertRaises(ValueError):
            historical_pairs([{"timestamp": 0}], [100])

    def test_constant_offset_is_required(self):
        _, diagnostics = historical_pairs([{"timestamp": 0}, {"timestamp": 20_000_000}], [100, 20_000_101])
        self.assertEqual(diagnostics["offsets_us"], [100, 101])
        self.assertFalse(diagnostics["offset_equal"])

    def test_offsets_are_per_clip_not_global(self):
        _, first = historical_pairs([{"timestamp": 0}, {"timestamp": 20_000_000}], [100, 20_000_100])
        _, second = historical_pairs([{"timestamp": 0}, {"timestamp": 20_000_000}], [900, 20_000_900])
        self.assertNotEqual(first["offsets_us"][0], second["offsets_us"][0])

    def test_negative_rows_do_not_define_the_boundary(self):
        _, diagnostics = historical_pairs([{"timestamp": -100}, {"timestamp": 0}, {"timestamp": 20_000_000}, {"timestamp": 40_000_000}], [100, 20_000_100])
        self.assertEqual(diagnostics["pai_start_us"], 0)

    def test_contract_never_uses_sequence_tracks_for_time_mapping(self):
        from scripts.replay_nurec_historical_time_mapping_full300 import parser

        self.assertEqual(parser().parse_args([]).output_root.name, "time_mapping_full300_historical_replay_v1")


if __name__ == "__main__":
    unittest.main()
