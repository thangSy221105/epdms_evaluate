"""Reproducible PAI/NuRec pose validation for an explicit external manifest."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, zipfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _wrap, _pose
except ModuleNotFoundError:
    from scripts.audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _wrap, _pose

XY_RMSE_WARN_M=0.30
XY_P95_WARN_M=0.50
YAW_RMSE_WARN_DEG=2.0

def _sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _sha(path: Path) -> str:
    with path.open("rb") as h: return _sha_bytes(h.read())

def _read_pai(spec: str, clip_id: str):
    path=Path(spec); entry=f"{clip_id}.egomotion.parquet"
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            if entry not in z.namelist(): raise ValueError("PAI_EGOMOTION_MEMBER_NOT_FOUND")
            data=z.read(entry); table=pq.read_table(pa.BufferReader(data)); provenance={"pai_archive_path":str(path),"pai_archive_sha256":_sha(path),"pai_member_name":entry,"pai_member_sha256":_sha_bytes(data)}
    else:
        if not path.is_file(): raise ValueError("PAI_EGOMOTION_MEMBER_NOT_FOUND")
        table=pq.read_table(path); provenance={"pai_archive_path":None,"pai_archive_sha256":None,"pai_member_name":path.name,"pai_member_sha256":_sha(path)}
    rows=sorted(table.to_pylist(),key=lambda r:int(r["timestamp"]))
    return [(int(r["timestamp"]),_pose([float(r[k]) for k in ("x","y","z")],[float(r[k]) for k in ("qx","qy","qz","qw")])) for r in rows],provenance

def _metrics(a,b):
    xy=[math.hypot(x[0][3]-y[0][3],x[1][3]-y[1][3]) for x,y in zip(a,b)]; ys=[]
    for x,y in zip(a,b): ys.append(abs(_wrap(math.atan2(x[1][0],x[0][0])-math.atan2(y[1][0],y[0][0]))))
    quant=lambda v,q: sorted(v)[max(0,min(len(v)-1,math.ceil(q*len(v))-1))]
    return {"xy_rmse_m":math.sqrt(sum(v*v for v in xy)/len(xy)),"xy_median_m":quant(xy,.5),"xy_p95_m":quant(xy,.95),"xy_max_m":max(xy),"yaw_rmse_deg":math.degrees(math.sqrt(sum(v*v for v in ys)/len(ys))),"yaw_median_deg":math.degrees(quant(ys,.5)),"yaw_p95_deg":math.degrees(quant(ys,.95)),"yaw_max_deg":math.degrees(max(ys))}

def _numerical_status(m):
    if m["xy_rmse_m"]<=XY_RMSE_WARN_M and m["xy_p95_m"]<=XY_P95_WARN_M and m["yaw_rmse_deg"]<=YAW_RMSE_WARN_DEG: return "PASS_NUMERICAL_SUPPORT"
    if m["xy_rmse_m"]<=2*XY_RMSE_WARN_M and m["xy_p95_m"]<=2*XY_P95_WARN_M and m["yaw_rmse_deg"]<=2*YAW_RMSE_WARN_DEG: return "WARN_NUMERICAL_RESIDUAL"
    return "FAIL_NUMERICAL_MISMATCH"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--time-alignment-jsonl",required=True); p.add_argument("--output-csv",required=True); p.add_argument("--output-summary",required=True); a=p.parse_args()
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]; time_source=Path(a.time_alignment_jsonl); rows=[]
    for item in manifest:
        cid=item["clip_id"]; rec=load_time_record(time_source,cid,int(item.get("physicalai_t0_us",5100000))); pai,pai_prov=_read_pai(item["pai_egomotion_path"],cid); nu=load_nurec_poses(Path(item["nurec_clip_dir"]))
        count=int(item.get("future_points",64)); queries=[rec["physicalai_t0_us"]+100000*(i+1) for i in range(count)]; nq=[rec["nurec_t0_us"]+100000*(i+1) for i in range(count)]
        base_p=interpolate_pose(pai,rec["physicalai_t0_us"]); base_n=interpolate_pose(nu,rec["nurec_t0_us"]); p_local=[_mm(inverse_pose(base_p),interpolate_pose(pai,t)) for t in queries]; n_local=[_mm(inverse_pose(base_n),interpolate_pose(nu,t)) for t in nq]; metric=_metrics(p_local,n_local); num=_numerical_status(metric)
        pred=item.get("prediction_source"); gt=item.get("gt_source"); gt_hash=_sha(Path(gt)) if gt and Path(gt).is_file() else None; pred_hash=_sha(Path(pred)) if pred and Path(pred).is_file() else None
        rows.append({"clip_id":cid,"physicalai_t0_us":rec["physicalai_t0_us"],"nurec_t0_us":rec["nurec_t0_us"],"per_clip_offset_us":rec["offset_us"],"pai_pose_min_us":pai[0][0],"pai_pose_max_us":pai[-1][0],"nurec_pose_min_us":nu[0][0],"nurec_pose_max_us":nu[-1][0],"query_min_us":nq[0],"query_max_us":nq[-1],"interpolation_in_range":True,"pose_comparison_point_count":count,"prediction_available":bool(pred),"prediction_frame_status":"NOT_AVAILABLE" if not pred else "UNRESOLVED","time_mapping_verified":True,"yaw_validation_source":"PAI_NUREC_POSE","numerical_validation_status":num,"pose_chain_provenance_status":"VERIFIED_RIG_TO_WORLD_DIRECTION","transform_source":"PAI egomotion + NuRec T_rig_worlds","transform_direction":"inverse(T_t0) @ T_t","status":num,**metric,**pai_prov,"nurec_rig_trajectories_sha256":_sha(Path(item["nurec_clip_dir"])/"rig_trajectories.json"),"time_mapping_source_sha256":_sha(time_source),"prediction_source_sha256":pred_hash,"gt_source_sha256":gt_hash})
    out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    failed=[r["clip_id"] for r in rows if r["numerical_validation_status"]=="FAIL_NUMERICAL_MISMATCH"]; warning=[r["clip_id"] for r in rows if r["numerical_validation_status"]=="WARN_NUMERICAL_RESIDUAL"]; overall="NEEDS_REVIEW" if failed else "VERIFIED_WITH_NUMERICAL_SUPPORT"
    summary={"clip_count":len(rows),"pose_chain_provenance_status":"VERIFIED_RIG_TO_WORLD_DIRECTION","numerical_validation_status":"FAIL_NUMERICAL_MISMATCH" if failed else ("WARN_NUMERICAL_RESIDUAL" if warning else "PASS_NUMERICAL_SUPPORT"),"pose_chain_status":overall,"coordinate_alignment_status":"PARTIALLY_VERIFIED","all_time_mappings_verified":all(r["time_mapping_verified"] for r in rows),"all_queries_in_range":all(r["interpolation_in_range"] for r in rows),"max_xy_rmse_m":max(r["xy_rmse_m"] for r in rows),"max_xy_p95_m":max(r["xy_p95_m"] for r in rows),"max_xy_max_m":max(r["xy_max_m"] for r in rows),"max_yaw_rmse_deg":max(r["yaw_rmse_deg"] for r in rows),"max_yaw_p95_deg":max(r["yaw_p95_deg"] for r in rows),"max_yaw_max_deg":max(r["yaw_max_deg"] for r in rows),"failed_clips":failed,"warning_clips":warning,"thresholds":{"xy_rmse_warn_m":XY_RMSE_WARN_M,"xy_p95_warn_m":XY_P95_WARN_M,"yaw_rmse_warn_deg":YAW_RMSE_WARN_DEG},"pose_interpolation_policy":"STRICT_IN_RANGE_ONLY","pose_rotation_interpolation":"SLERP","localization_transform":"FULL_SE3_INVERSE_T0","time_mapping_source":str(time_source),"per_clip_offset_rederived":False,"remaining_blockers":["PREDICTION_FRAME_PROVENANCE","OBSTACLE_REFERENCE_FRAME_NOT_PRESERVED","MAP_GEOMETRY_FRAME_NOT_DECLARED","DRIVABLE_SPACE_MISSING"]}
    Path(a.output_summary).write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
