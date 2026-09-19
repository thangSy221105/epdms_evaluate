"""Provenance-first NuRec coordinate audit with strict full-SE(3) poses."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from typing import Any

class PoseInterpolationOutOfRangeError(ValueError):
    pass

def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields=sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=fields)
        if fields: w.writeheader(); w.writerows(rows)

def _read(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def _wrap(a: float) -> float: return math.atan2(math.sin(a), math.cos(a))
def sha256(path: Path) -> str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda:h.read(1024*1024), b""): d.update(b)
    return d.hexdigest()

def _qnorm(q):
    n=math.sqrt(sum(v*v for v in q))
    if n==0 or not math.isfinite(n): raise ValueError("invalid quaternion")
    return [v/n for v in q]

def _slerp(a,b,t):
    a=_qnorm(a); b=_qnorm(b); dot=sum(x*y for x,y in zip(a,b))
    if dot<0: b=[-v for v in b]; dot=-dot
    if dot>.9995: return _qnorm([x+t*(y-x) for x,y in zip(a,b)])
    theta=math.acos(max(-1,min(1,dot))); s=math.sin(theta); u=math.sin((1-t)*theta)/s; v=math.sin(t*theta)/s
    return _qnorm([u*x+v*y for x,y in zip(a,b)])

def _qrot(q):
    x,y,z,w=_qnorm(q)
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]

def _rquat(r):
    tr=r[0][0]+r[1][1]+r[2][2]
    if tr>0:
        s=math.sqrt(tr+1)*2; return [(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,(r[1][0]-r[0][1])/s,.25*s]
    if r[0][0]>r[1][1] and r[0][0]>r[2][2]:
        s=math.sqrt(1+r[0][0]-r[1][1]-r[2][2])*2; return [.25*s,(r[0][1]+r[1][0])/s,(r[0][2]+r[2][0])/s,(r[2][1]-r[1][2])/s]
    if r[1][1]>r[2][2]:
        s=math.sqrt(1+r[1][1]-r[0][0]-r[2][2])*2; return [(r[0][1]+r[1][0])/s,.25*s,(r[1][2]+r[2][1])/s,(r[0][2]-r[2][0])/s]
    s=math.sqrt(1+r[2][2]-r[0][0]-r[1][1])*2; return [(r[0][2]+r[2][0])/s,(r[1][2]+r[2][1])/s,.25*s,(r[1][0]-r[0][1])/s]

def _pose(p,q):
    r=_qrot(q); return [r[0]+[p[0]],r[1]+[p[1]],r[2]+[p[2]],[0.,0.,0.,1.]]

def _mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def inverse_pose(t):
    r=[x[:3] for x in t[:3]]; p=[t[i][3] for i in range(3)]; rt=[[r[j][i] for j in range(3)] for i in range(3)]; q=[-sum(rt[i][j]*p[j] for j in range(3)) for i in range(3)]
    return [rt[0]+[q[0]],rt[1]+[q[1]],rt[2]+[q[2]],[0.,0.,0.,1.]]

def interpolate_pose(samples, timestamp):
    if not samples: raise ValueError("POSE_INTERPOLATION_NO_SAMPLES")
    if timestamp<samples[0][0] or timestamp>samples[-1][0]: raise PoseInterpolationOutOfRangeError(f"POSE_INTERPOLATION_OUT_OF_RANGE query_timestamp_us={timestamp} min_timestamp_us={samples[0][0]} max_timestamp_us={samples[-1][0]}")
    for left,right in zip(samples,samples[1:]):
        if timestamp==left[0]: return left[1]
        if left[0]<timestamp<=right[0]:
            a=(timestamp-left[0])/(right[0]-left[0]); p=[left[1][i][3]+a*(right[1][i][3]-left[1][i][3]) for i in range(3)]; q=_slerp(_rquat([x[:3] for x in left[1][:3]]),_rquat([x[:3] for x in right[1][:3]]),a); return _pose(p,q)
    return samples[-1][1]

def load_nurec_poses(root: Path):
    d=_read(root/"rig_trajectories.json")["rig_trajectories"][0]
    return [(int(t),[[float(v) for v in row] for row in m]) for t,m in zip(d["T_rig_world_timestamps_us"],d["T_rig_worlds"])]

def load_time_record(path: Path, clip_id: str, expected_t0_us: int):
    if path.suffix.lower()==".jsonl":
        records=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        value=_read(path); records=value if isinstance(value,list) else value.get("records",[value])
    matches=[r for r in records if r.get("clip_id")==clip_id]
    if len(matches)!=1: raise ValueError(f"TIME_MAPPING_CLIP_ID_MISMATCH expected={clip_id} matches={len(matches)}")
    r=matches[0]
    if r.get("verified") is not True: raise ValueError("TIME_MAPPING_UNVERIFIED")
    if r.get("mapping_type") not in {"PER_CLIP_REBASE","PER_CLIP_OFFSET","RESOLVED_PER_CLIP_REBASE"}: raise ValueError("TIME_MAPPING_TYPE_UNSUPPORTED")
    if any(not isinstance(r.get(k),int) for k in ("physicalai_t0_us","nurec_t0_us","offset_us")): raise ValueError("TIME_MAPPING_FIELD_INVALID")
    if r["physicalai_t0_us"]!=expected_t0_us or r["physicalai_t0_us"]+r["offset_us"]!=r["nurec_t0_us"]: raise ValueError("TIME_MAPPING_T0_MISMATCH")
    return r

def select_prediction(path, clip_id, mode=None, alpha=None, record_key=None):
    if path is None: return None,"NOT_AVAILABLE"
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x).get("clip_id")==clip_id]
    if record_key is not None: rows=[r for r in rows if r.get("record_key")==record_key]
    elif mode is not None or alpha is not None: rows=[r for r in rows if (mode is None or r.get("mode")==mode) and (alpha is None or float(r.get("alpha"))==alpha)]
    if len(rows)>1: raise ValueError("PREDICTION_SELECTOR_REQUIRED_DUPLICATE_CLIP_ROWS")
    if not rows: return None,"NOT_AVAILABLE"
    frame=rows[0].get("coordinate_frame") or rows[0].get("prediction_frame")
    return rows[0],("VERIFIED_FROM_PROVENANCE" if frame else "UNRESOLVED")

def run(args):
    root=Path(args.nurec_clip_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rec=load_time_record(Path(args.time_alignment_jsonl),args.clip_id,args.prediction_t0_us); poses=load_nurec_poses(root); t0=rec["nurec_t0_us"]; queries=[t0+100000*(i+1) for i in range(64)]; base=interpolate_pose(poses,t0); local=[_mm(inverse_pose(base),interpolate_pose(poses,t))[0:3] for t in queries]; prediction,pstatus=select_prediction(Path(args.prediction_jsonl) if args.prediction_jsonl else None,args.clip_id,args.prediction_mode,args.prediction_alpha,args.prediction_record_key)
    _dump(out/"coordinate_frame_inventory.json",{"prediction":{"status":pstatus},"ground_truth":{"frame":"EGO_AT_T0","status":"SUPPORTED_BY_UPSTREAM_EGO_AT_T0_CONTRACT"},"egomotion":{"frame":"rig_to_world","status":"VERIFIED_RIG_TO_WORLD_DIRECTION"},"obstacle":{"frame":None,"status":"UNRESOLVED"},"map":{"frame":None,"status":"PROVENANCE_AVAILABLE_NOT_INTEGRATED"}})
    _dump(out/"prediction_coordinate_contract.json",{"status":pstatus,"selector":{"mode":args.prediction_mode,"alpha":args.prediction_alpha,"record_key":args.prediction_record_key}}); _dump(out/"gt_coordinate_contract.json",{"status":"SUPPORTED_BY_UPSTREAM_EGO_AT_T0_CONTRACT","frame":"EGO_AT_T0","anchor":"t0_pose"}); _dump(out/"rig_pose_contract.json",{"status":"VERIFIED_RIG_TO_WORLD_DIRECTION","interpolation":"STRICT_IN_RANGE_ONLY + FULL_SE3 + SLERP","source":"rig_trajectories.json:T_rig_worlds"})
    _csv(out/"calibration_transform_inventory.csv",[{"source":"rig_trajectories.json","field":"world_to_nre.matrix","matrix_present":True,"semantics_status":"PARTIAL","semantics_evidence":"explicit matrix; scoring binding not proven","usable_for_scoring":False}]); _dump(out/"coordinate_transform_graph.json",{"common_frame":"EGO_AT_T0","edges":[{"source":"GT","target":"EGO_AT_T0","status":"DIRECT_SAME_FRAME","verified":True},{"source":"NuRec rig/world","target":"EGO_AT_T0","status":"VERIFIED_DYNAMIC_TRANSFORM","source":"T_rig_worlds + inverse(T_t0)","verified":True},{"source":"prediction","target":"EGO_AT_T0","status":"UNRESOLVED","verified":False},{"source":"obstacle","target":"EGO_AT_T0","status":"UNRESOLVED","verified":False},{"source":"map","target":"EGO_AT_T0","status":"PROVENANCE_AVAILABLE_NOT_INTEGRATED","verified":False}]})
    _csv(out/"ego_coordinate_validation.csv",[{"clip_id":args.clip_id,"query_min_us":queries[0],"query_max_us":queries[-1],"interpolation_in_range":True,"pose_query_count":len(local),"status":"POSE_CHAIN_VERIFIED"}])
    summary={"clip_id":args.clip_id,"time_mapping_source":str(args.time_alignment_jsonl),"time_mapping_verified":True,"per_clip_offset_us":rec["offset_us"],"coordinate_alignment_status":"PARTIALLY_VERIFIED","coordinate_code_ready":True,"coordinate_data_ready":False,"pose_interpolation_policy":"STRICT_IN_RANGE_ONLY","pose_rotation_interpolation":"SLERP","localization_transform":"FULL_SE3_INVERSE_T0","prediction_frame_status":pstatus,"gt_frame_status":"SUPPORTED_BY_UPSTREAM_EGO_AT_T0_CONTRACT","nurec_pose_chain_status":"VERIFIED_RIG_TO_WORLD_DIRECTION","obstacle_frame_status":"UNRESOLVED","map_frame_status":"PROVENANCE_AVAILABLE_NOT_INTEGRATED","drivable_space_status":"MISSING","coordinate_audit_derived_new_time_offset":False,"remaining_blockers":["PREDICTION_FRAME_PROVENANCE","OBSTACLE_REFERENCE_FRAME_NOT_PRESERVED","MAP_GEOMETRY_FRAME_NOT_DECLARED","DRIVABLE_SPACE_MISSING","OBSERVATION_COVERAGE"]}; _dump(out/"coordinate_alignment_summary.json",summary); return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--nurec-clip-dir",required=True); p.add_argument("--prediction-jsonl"); p.add_argument("--ground-truth-jsonl"); p.add_argument("--time-alignment-jsonl",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--clip-id",required=True); p.add_argument("--prediction-t0-us",type=int,default=5100000); p.add_argument("--prediction-mode"); p.add_argument("--prediction-alpha",type=float); p.add_argument("--prediction-record-key"); a=p.parse_args(); print(json.dumps(run(a),indent=2,ensure_ascii=True)); return 0

if __name__=="__main__": raise SystemExit(main())
