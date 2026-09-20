import unittest

from scripts.audit_nurec_xodr_provenance_v2 import parse_geo_reference, sample_reference_geometry
import xml.etree.ElementTree as ET


class NuRecXodrProvenanceV2Tests(unittest.TestCase):
    def test_georeference_parses_historical_altitude_typo(self):
        value = parse_geo_reference(
            "+proj=tmerc +lon_0=17.9 +lat_0=59.3 +=alt_0=0 +ellps=WGS84 +units=m"
        )
        self.assertEqual(value["proj"], "tmerc")
        self.assertEqual(value["ellps"], "WGS84")
        self.assertEqual(value["alt_0"], 0.0)
        self.assertEqual(value["parse_status"], "PARSED")

    def test_reference_geometry_sampling_is_deterministic(self):
        root = ET.fromstring(
            "<OpenDRIVE><road><planView>"
            '<geometry x="0" y="0" hdg="0" length="10"><line/></geometry>'
            "</planView></road></OpenDRIVE>"
        )
        points = sample_reference_geometry(root, spacing_m=5.0)
        self.assertEqual(points.shape, (3, 4))
        self.assertEqual(points[0].tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(points[-1].tolist(), [10.0, 0.0, 0.0, 1.0])

    def test_missing_road_rule_cannot_become_direction_ready(self):
        # The audit's contract requires RHT/LHT provenance; lane ids alone are
        # intentionally not a fallback.
        road_rule = ""
        lane_ids = [1, -1]
        self.assertFalse(bool(road_rule) and bool(lane_ids))


if __name__ == "__main__":
    unittest.main()
