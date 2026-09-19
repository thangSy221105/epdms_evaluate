import math, json, tempfile, unittest
from pathlib import Path
from scripts.audit_nurec_coordinate_alignment import PoseInterpolationOutOfRangeError, interpolate_pose, inverse_pose, load_time_record, select_prediction

def pose(x,y,z,yaw=0.0):
    c,s=math.cos(yaw),math.sin(yaw); return [[c,-s,0.,x],[s,c,0.,y],[0.,0.,1.,z],[0.,0.,0.,1.]]

class NuRecCoordinateAlignmentTests(unittest.TestCase):
    def test_exact_endpoints_allowed(self):
        samples=[(0,pose(0,0,0)),(10,pose(10,0,0))]; self.assertEqual(interpolate_pose(samples,0),samples[0][1]); self.assertEqual(interpolate_pose(samples,10),samples[1][1])
    def test_out_of_range_is_rejected(self):
        samples=[(0,pose(0,0,0)),(10,pose(10,0,0))]
        with self.assertRaises(PoseInterpolationOutOfRangeError): interpolate_pose(samples,-1)
        with self.assertRaises(PoseInterpolationOutOfRangeError): interpolate_pose(samples,11)
    def test_translation_is_linear(self):
        self.assertAlmostEqual(interpolate_pose([(0,pose(0,0,0)),(10,pose(10,0,0))],5)[0][3],5.)
    def test_rotation_uses_shortest_slerp(self):
        r=interpolate_pose([(0,pose(0,0,0,math.radians(179))),(10,pose(0,0,0,math.radians(-179)))],5); self.assertAlmostEqual(abs(math.degrees(math.atan2(r[1][0],r[0][0]))),180.,places=4)
    def test_full_se3_keeps_roll_pitch_translation(self):
        first=pose(0,0,0,math.pi/2); second=pose(0,1,0,math.pi/2); local=inverse_pose(first); self.assertAlmostEqual(local[0][3],0.,places=6); self.assertAlmostEqual(second[1][3],1.,places=6)
    def test_time_sidecar_is_consumed_and_validated(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"time.jsonl"; path.write_text(json.dumps({"clip_id":"a","physicalai_t0_us":5100000,"nurec_t0_us":100,"offset_us":-5099900,"mapping_type":"PER_CLIP_REBASE","verified":True})+"\n")
            self.assertEqual(load_time_record(path,"a",5100000)["offset_us"],-5099900)
            with self.assertRaises(ValueError): load_time_record(path,"b",5100000)
    def test_prediction_duplicate_requires_selector(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"pred.jsonl"; path.write_text('{"clip_id":"a","mode":"x","alpha":0}\n{"clip_id":"a","mode":"y","alpha":1}\n')
            with self.assertRaises(ValueError): select_prediction(path,"a")
            row,status=select_prediction(path,"a",mode="x",alpha=0); self.assertEqual(row["mode"],"x"); self.assertEqual(status,"UNRESOLVED")
    def test_missing_prediction_is_nonfatal_and_unresolved(self):
        row,status=select_prediction(None,"a"); self.assertIsNone(row); self.assertEqual(status,"NOT_AVAILABLE")

if __name__=="__main__": unittest.main()
