import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts.audit_nurec_xodr_provenance_v2 import (
    direction_contract_for_root,
    effective_road_rule,
    map_to_ncore_transform,
    parse_geo_reference,
    sample_reference_geometry,
)


def _root(road_rule=None, version="1.4", junction="-1", lane_direction=None):
    rule = "" if road_rule is None else f' rule="{road_rule}"'
    direction = "" if lane_direction is None else f' direction="{lane_direction}"'
    return ET.fromstring(
        f'<OpenDRIVE><header revMajor="{version.split(".")[0]}" revMinor="{version.split(".")[1]}"/>'
        f'<road id="1" junction="{junction}"{rule}><planView>'
        '<geometry x="0" y="0" hdg="0" length="10"><line/></geometry>'
        f'</planView><lanes><laneSection s="0"><left><lane id="1"{direction}/></left>'
        '<center><lane id="0"/></center><right><lane id="-1"/></right>'
        '</laneSection></lanes></road></OpenDRIVE>'
    )


class NuRecXodrProvenanceV2Tests(unittest.TestCase):
    def test_georeference_parses_historical_altitude_typo_and_preserves_raw_hash(self):
        raw = "+proj=tmerc +lon_0=17.9 +lat_0=59.3 +=alt_0=0 +ellps=WGS84 +units=m"
        value = parse_geo_reference(raw)
        self.assertEqual(value["proj"], "tmerc")
        self.assertEqual(value["ellps"], "WGS84")
        self.assertEqual(value["alt_0"], 0.0)
        self.assertEqual(value["raw"], raw)
        self.assertEqual(value["parse_status"], "PARSED")

    def test_non_finite_georeference_is_rejected(self):
        value = parse_geo_reference("+proj=tmerc +lon_0=17.9 +lat_0=nan +alt_0=0")
        self.assertNotEqual(value["parse_status"], "PARSED")
        self.assertIn("lat_0", value["invalid_numeric_fields"])

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

    def test_missing_rule_uses_open_drive_14_default_rht(self):
        self.assertEqual(effective_road_rule("1.4", None), ("RHT", "ASAM_DEFAULT"))
        contract = direction_contract_for_root(_root(), "1.4")
        self.assertTrue(contract["effective_road_rule_available"])
        self.assertEqual(contract["effective_road_rule"], "RHT")

    def test_explicit_rht_and_lht_are_production_direction_inputs(self):
        self.assertEqual(effective_road_rule("1.4", "RHT"), ("RHT", "ROAD_ATTRIBUTE"))
        self.assertEqual(effective_road_rule("1.4", "LHT"), ("LHT", "ROAD_ATTRIBUTE"))
        self.assertEqual(direction_contract_for_root(_root("RHT"), "1.4")["effective_road_rule"], "RHT")
        self.assertEqual(direction_contract_for_root(_root("LHT"), "1.4")["effective_road_rule"], "LHT")

    def test_lane_id_and_reference_line_are_recorded_without_independent_direction_claim(self):
        contract = direction_contract_for_root(_root(), "1.4")
        self.assertTrue(contract["lane_topology_available"])
        self.assertEqual(contract["lane_direction_attribute_count"], 0)
        self.assertEqual(contract["legal_direction_status"], "VERIFIED")

    def test_lane_direction_override_is_recorded_when_present(self):
        contract = direction_contract_for_root(_root(lane_direction="reversed"), "1.4")
        self.assertEqual(contract["lane_direction_attribute_count"], 1)
        self.assertEqual(contract["lane_direction_override_status"], "PRESENT_BUT_VERSION_SEMANTICS_REQUIRE_REVIEW")

    def test_junction_remains_partial_and_excluded(self):
        contract = direction_contract_for_root(_root(junction="7"), "1.4")
        self.assertEqual(contract["legal_direction_status"], "PARTIAL")
        self.assertIn("JUNCTION_EXCLUDED_LANE_LINK_POLICY_UNRESOLVED", contract["blockers"])

    def test_unknown_version_fails_closed(self):
        contract = direction_contract_for_root(_root(), "9.9")
        self.assertEqual(contract["legal_direction_status"], "UNRESOLVED")

    def test_coordinate_formula_uses_documented_multiplication_order(self):
        t_ecef_enu = np.eye(4)
        t_ecef_enu[0, 3] = 10.0
        world_base = np.eye(4)
        world_base[1, 3] = 3.0
        expected = np.linalg.inv(t_ecef_enu @ world_base)
        actual = map_to_ncore_transform(t_ecef_enu, world_base)
        np.testing.assert_allclose(actual, expected)

    def test_map_to_ncore_round_trip(self):
        t_ecef_enu = np.eye(4)
        t_ecef_enu[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t_ecef_enu[0, 3] = 10.0
        world_base = np.eye(4)
        world_base[1, 3] = 3.0
        map_to_ncore = map_to_ncore_transform(t_ecef_enu, world_base)
        np.testing.assert_allclose(np.linalg.inv(map_to_ncore), t_ecef_enu @ world_base)


if __name__ == "__main__":
    unittest.main()
