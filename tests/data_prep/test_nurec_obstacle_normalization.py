import json, math, tempfile, unittest
from pathlib import Path
from scripts.prepare_nurec_obstacles import classify_queries, load_sequence_tracks, transform_object_to_ego_t0, validate_quaternion, coverage_summary, pose_yaw

def pose(x=0,y=0,z=0,yaw=0):
    c,s=math.cos(yaw),math.sin(yaw); return [[c,-s,0,x],[s,c,0,y],[0,0,1,z],[0,0,0,1]]

class ObstacleNormalizationTests(unittest.TestCase):
    def test_world_to_ego_identity_and_yaw(self):
        out=transform_object_to_ego_t0(pose(x=1),pose()); self.assertAlmostEqual(out[0][3],1)
        out=transform_object_to_ego_t0(pose(y=1,yaw=math.pi/2),pose(yaw=math.pi/2)); self.assertAlmostEqual(out[0][3],1,places=6)
    def test_full_orientation_and_dimensions_are_rigid(self):
        out=transform_object_to_ego_t0(pose(z=1,yaw=math.pi/2),pose(yaw=math.pi/2)); self.assertAlmostEqual(out[2][3],1); self.assertAlmostEqual(pose_yaw(out),0,places=6)
    def test_quaternion_validation(self):
        self.assertEqual(validate_quaternion([0,0,0,2]),[0,0,0,1])
        for q in ([0,0,0,0],[math.nan,0,0,1]):
            with self.assertRaises(ValueError): validate_quaternion(q)
    def test_coverage_object_nearest_unknown_and_empty(self):
        rows=classify_queries([100,200,300],[100,195],50); self.assertEqual(rows[0]["observation_state"],"EXACT_OBJECT"); self.assertEqual(rows[1]["observation_state"],"NEAREST_OBJECT_WITHIN_TOLERANCE"); self.assertEqual(rows[2]["observation_state"],"OUT_OF_RANGE")
        self.assertEqual(classify_queries([100],[],50)[0]["observation_state"],"OUT_OF_RANGE")
        self.assertEqual(classify_queries([100],[100],50,[200])[0]["observation_state"],"EXACT_OBJECT")
        self.assertEqual(coverage_summary(rows,"CF")["CF_REQUIRED_QUERY_COUNT"],3)
    def test_sequence_schema_and_bad_pose(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"sequence_tracks.json"; p.write_text(json.dumps({"tracks_data":{"tracks_id":["a"],"tracks_poses":[[[1,2,3,0,0,0,1],[1,2,3,0,0,0,0]]],"tracks_timestamps_us":[[1,0]],"tracks_label_class":["car"],"tracks_flags":["NONE"]},"cuboidtracks_data":{"cuboids_dims":[[4,2,1]]}}))
            rows,errors,schema=load_sequence_tracks(p); self.assertEqual(schema["pose_representation"],"[x,y,z,qx,qy,qz,qw]"); self.assertEqual(len(rows),1); self.assertTrue(errors)
    def test_no_empty_attestation_from_missing_rows(self):
        row=classify_queries([100],[],50)[0]; self.assertNotEqual(row["observation_state"],"EXACT_CONFIRMED_EMPTY"); self.assertFalse(row["attestation_verified"])

if __name__=="__main__": unittest.main()
