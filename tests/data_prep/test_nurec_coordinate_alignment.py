import json, math, tempfile, unittest
from pathlib import Path
from scripts.audit_nurec_coordinate_alignment import PoseInterpolationOutOfRangeError, interpolate_pose, inverse_pose, load_time_record, select_prediction, _mm, _pose, inspect_map
from scripts.audit_nurec_coordinate_multiclip import _numerical_status

def pose(x=0,y=0,z=0,roll=0,pitch=0,yaw=0):
    cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    r=[[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
    return [r[0]+[x],r[1]+[y],r[2]+[z],[0.,0.,0.,1.]]

class NuRecCoordinateAlignmentTests(unittest.TestCase):
    def test_endpoints_and_out_of_range(self):
        s=[(0,pose()),(10,pose(x=10))]; self.assertEqual(interpolate_pose(s,0),s[0][1]); self.assertEqual(interpolate_pose(s,10),s[1][1])
        for t in (-1,11):
            with self.assertRaises(PoseInterpolationOutOfRangeError): interpolate_pose(s,t)
    def test_linear_translation(self): self.assertAlmostEqual(interpolate_pose([(0,pose()),(10,pose(x=10))],5)[0][3],5.)
    def test_shortest_yaw_slerp(self):
        r=interpolate_pose([(0,pose(yaw=math.radians(179))),(10,pose(yaw=math.radians(-179)))],5); self.assertAlmostEqual(abs(math.degrees(math.atan2(r[1][0],r[0][0]))),180.,places=4)
    def test_non_yaw_slerp(self):
        r=interpolate_pose([(0,pose(roll=0,pitch=0)),(10,pose(roll=math.pi/2,pitch=math.pi/4))],5); self.assertTrue(abs(r[2][1])>0.1)
    def test_full_se3_yaw_direction(self):
        t0=pose(yaw=math.pi/2); t=pose(y=1,yaw=math.pi/2); local=_mm(inverse_pose(t0),t); self.assertAlmostEqual(local[0][3],1.,places=6); self.assertAlmostEqual(local[1][3],0.,places=6); self.assertNotAlmostEqual(_mm(t,inverse_pose(t0))[0][3],1.,places=3)
    def test_full_se3_roll_and_pitch(self):
        roll_local=_mm(inverse_pose(pose(roll=math.pi/2)),pose(z=1)); self.assertAlmostEqual(roll_local[1][3],1.,places=6)
        pitch_local=_mm(inverse_pose(pose(pitch=math.pi/2)),pose(z=1)); self.assertAlmostEqual(pitch_local[0][3],-1.,places=6)
    def _sidecar(self, value):
        td=tempfile.TemporaryDirectory(); path=Path(td.name)/"time.jsonl"; path.write_text(json.dumps(value)+"\n"); return td,path
    def test_time_sidecar_strict_validation(self):
        good={"clip_id":"a","physicalai_t0_us":10,"nurec_t0_us":110,"offset_us":100,"mapping_type":"PER_CLIP_REBASE","verified":True}; td,path=self._sidecar(good); self.assertEqual(load_time_record(path,"a",10)["offset_us"],100); td.cleanup()
        cases=[{**good,"verified":False},{**good,"mapping_type":"BAD"},{**good,"physicalai_t0_us":11},{**good,"offset_us":99},{k:v for k,v in good.items() if k!="offset_us"},{**good,"offset_us":True}]
        for case in cases:
            td,path=self._sidecar(case)
            with self.assertRaises(ValueError): load_time_record(path,"a",10)
            td.cleanup()
    def test_sidecar_duplicate_and_per_clip_isolation(self):
        td=tempfile.TemporaryDirectory(); path=Path(td.name)/"time.jsonl"; a={"clip_id":"a","physicalai_t0_us":10,"nurec_t0_us":110,"offset_us":100,"mapping_type":"PER_CLIP_REBASE","verified":True}; b={**a,"clip_id":"b","nurec_t0_us":210,"offset_us":200}; path.write_text("\n".join(map(json.dumps,[a,b,a]))+"\n")
        with self.assertRaises(ValueError): load_time_record(path,"a",10)
        path.write_text("\n".join(map(json.dumps,[a,b]))+"\n"); self.assertEqual(load_time_record(path,"b",10)["offset_us"],200); td.cleanup()
    def test_prediction_selector_and_statuses(self):
        td=tempfile.TemporaryDirectory(); path=Path(td.name)/"pred.jsonl"; path.write_text('{"clip_id":"a","mode":"x","alpha":0}\n{"clip_id":"a","mode":"y","alpha":1}\n')
        with self.assertRaises(ValueError): select_prediction(path,"a")
        row,status=select_prediction(path,"a",mode="x",alpha=0); self.assertEqual(status,"UNRESOLVED"); self.assertEqual(row["mode"],"x")
        path.write_text('{"clip_id":"a","mode":"x","alpha":0,"coordinate_frame":"EGO"}\n'); self.assertEqual(select_prediction(path,"a")[1],"DECLARED_UNVERIFIED")
        path.write_text('{"clip_id":"a","coordinate_frame":"EGO","anchor":"rear_axle","provenance":"manifest"}\n'); self.assertEqual(select_prediction(path,"a")[1],"VERIFIED_FROM_PROVENANCE")
        self.assertEqual(select_prediction(None,"a")[1],"NOT_AVAILABLE"); td.cleanup()
    def test_numerical_gate_and_map_fail_closed(self):
        good={"xy_rmse_m":0.1,"xy_p95_m":0.2,"yaw_rmse_deg":0.5}; bad={"xy_rmse_m":1.0,"xy_p95_m":0.2,"yaw_rmse_deg":0.5}
        self.assertEqual(_numerical_status(good),"PASS_NUMERICAL_SUPPORT"); self.assertEqual(_numerical_status(bad),"FAIL_NUMERICAL_MISMATCH")
        with tempfile.TemporaryDirectory() as td: self.assertEqual(inspect_map(Path(td)),("MISSING_T_WORLD_BASE","MISSING"))

if __name__=="__main__": unittest.main()
