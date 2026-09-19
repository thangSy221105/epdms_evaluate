import math
import unittest
from scripts.audit_nurec_obstacle_provenance import _crosscheck, _public_trace
from scripts.prepare_nurec_obstacles import classify_queries, _wrap

class ObstacleProvenanceTests(unittest.TestCase):
    def test_public_chain_contains_frame_and_world_steps(self):
        trace=_public_trace(); path=trace["public_instant_nurec_track_transform_chain"]["code_path"]
        self.assertTrue(any("evaluate_poses" in x for x in path)); self.assertTrue(any("T_world_world_base" in x for x in path))
    def test_exact_track_crosscheck_preserves_dimensions(self):
        row={"track_id":"a","timestamp_us":1,"center":[0,0,0],"quaternion":[0,0,0,1],"dimensions":[4,2,1]}; pairs,summary=_crosscheck([row],[row]); self.assertEqual(summary["pair_count"],1); self.assertEqual(pairs[0]["dimension_abs_diff"],0)
    def test_nearby_object_is_not_complete_frame(self):
        row=classify_queries([100],[95],50)[0]; self.assertIn(row["observation_state"],["NEAREST_OBJECT_WITHIN_TOLERANCE"]); self.assertFalse(row["attestation_verified"])
    def test_missing_rows_are_not_empty(self):
        row=classify_queries([100],[],50)[0]; self.assertNotEqual(row["observation_state"],"EXACT_CONFIRMED_EMPTY")
    def test_sensor_like_timeline_does_not_attest_empty(self):
        rows=classify_queries([100,200],[100],50); self.assertFalse(any(r["attestation_verified"] for r in rows))
    def test_yaw_wrap_handles_plus_minus_179_degrees(self):
        delta=abs(_wrap(math.radians(179)-math.radians(-179)))
        self.assertAlmostEqual(math.degrees(delta),2.0,places=6)
    def test_yaw_wrap_identical_angles_are_zero(self):
        self.assertEqual(_wrap(0.0),0.0)
        self.assertEqual(_wrap(math.pi),math.pi)
    def test_crosscheck_uses_wrapped_yaw_residual(self):
        seq={"track_id":"a","timestamp_us":1,"center":[0,0,0],"quaternion":[0,0,math.sin(math.radians(89.5)),math.cos(math.radians(89.5))],"dimensions":[4,2,1]}
        obs={"track_id":"a","timestamp_us":1,"center":[0,0,0],"quaternion":[0,0,math.sin(math.radians(-89.5)),math.cos(math.radians(-89.5))],"dimensions":[4,2,1]}
        _,summary=_crosscheck([seq],[obs])
        self.assertLess(summary["yaw_rmse_deg"],2.0)

if __name__=="__main__": unittest.main()
