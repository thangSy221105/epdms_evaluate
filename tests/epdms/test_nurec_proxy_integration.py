import copy
import unittest

import numpy as np

from tools.epdms.nurec_inputs import ObservationReadiness, decorate_inputs
from tools.epdms.schemas import VehicleParameters
from tools.epdms.score_record import evaluate_single_condition


class TestNuRecProxyIntegration(unittest.TestCase):
    def setUp(self):
        self.t0 = 5_100_000
        self.vehicle = VehicleParameters()
        self.contract = {
            "status": "VERIFIED_FROZEN_CONTRACT",
            "common_evaluation_frame": "EGO_AT_T0",
            "anchor_semantics": {
                "prediction": "ego_t0_reference",
                "ground_truth": "ego_t0_reference",
                "obstacle": "cuboid_center",
                "map": "drivable_geometry",
                "alignment_status": "VERIFIED_BY_ACCEPTED_PRIOR_CONTRACT",
            },
            "obstacle_source_frame": "NCORE_LOCAL_WORLD",
            "map_source_frame": "NCORE_LOCAL_WORLD",
            "obstacle_transform_status": "VERIFIED",
        }
        self.readiness = ObservationReadiness("clip", 41, 41, 0, 0, 51, 51, 0, 0)

    def _wps(self):
        return [{"t_s": 0.1 * (i + 1), "x_m": i * 0.1, "y_m": 0.0} for i in range(40)]

    def _rows(self):
        pred = {"clip_id": "clip", "mode": "cross_scene", "alpha": 0.0, "t0_us": self.t0, "clean_waypoints": self._wps()}
        gt = {"clip_id": "clip", "t0_us": self.t0, "expert_future": self._wps()}
        context = {"clip_id": "clip", "semantic_context": {"obstacle": {"all_obstacles": []}}}
        return pred, gt, context

    def test_frozen_contract_adapter_passes_coordinate_gate_without_world_to_nre(self):
        pred, gt, context = self._rows()
        p, g, c = decorate_inputs(pred, gt, context, self.contract, {"offset_us": 0, "verified": True}, self.readiness)
        self.assertTrue(c["coordinate_alignment_verified"])
        self.assertFalse(c["coordinate_contract"]["world_to_nre_applied_to_ego"])
        self.assertEqual(p["coordinate_frame"], "EGO_AT_T0")
        self.assertEqual(g["coordinate_frame"], "EGO_AT_T0")
        self.assertEqual(pred.get("coordinate_frame"), None)

    def test_missing_observation_emits_null_metrics_and_explicit_status(self):
        pred, gt, context = self._rows()
        missing = ObservationReadiness("clip", 41, 40, 0, 1, 51, 51, 0, 0)
        p, g, c = decorate_inputs(pred, gt, context, self.contract, {"offset_us": 0, "verified": True}, missing)
        record = evaluate_single_condition(p, c, g, self.vehicle, strict_mode=True, lane_polygons=[], map_status="OK")
        self.assertFalse(record.valid)
        self.assertEqual(record.overall_score_status, "INCOMPLETE_OBSERVATION_EVIDENCE")
        self.assertIsNone(record.collision_free_proxy)
        self.assertIsNone(record.ttc_proxy)
        self.assertIsNone(record.nurec_safety_proxy_v1)

    def test_missing_map_emits_null_dac_and_proxy(self):
        pred, gt, context = self._rows()
        p, g, c = decorate_inputs(pred, gt, context, self.contract, {"offset_us": 0, "verified": True}, self.readiness)
        # Explicit empties make the observation gate pass; the map gate must
        # still fail closed rather than substituting DAC=1.
        c["confirmed_empty_scene"] = True
        record = evaluate_single_condition(p, c, g, self.vehicle, strict_mode=True, lane_polygons=[], map_status="MAP_TRANSFORM_MISSING")
        self.assertFalse(record.valid)
        self.assertEqual(record.overall_score_status, "MAP_NOT_READY")
        self.assertIsNone(record.dac_proxy)
        self.assertIsNone(record.nurec_safety_proxy_v1)

    def test_formula_and_range_fields_remain_available_when_scored(self):
        pred, gt, context = self._rows()
        c = copy.deepcopy(context)
        p, g, c = decorate_inputs(pred, gt, c, self.contract, {"offset_us": 0, "verified": True}, self.readiness)
        c["confirmed_empty_scene"] = True
        polygon = [np.asarray([[-20.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-20.0, 20.0]])]
        record = evaluate_single_condition(p, c, g, self.vehicle, strict_mode=True, lane_polygons=polygon, map_status="OK")
        self.assertTrue(record.valid, record.failure_reason)
        expected = record.collision_free_proxy * record.dac_proxy * (5 * record.ttc_proxy + 5 * record.progress_gt_proxy + 2 * record.future_comfort_proxy) / 12
        self.assertAlmostEqual(record.nurec_safety_proxy_v1, expected, places=12)


if __name__ == "__main__":
    unittest.main()
