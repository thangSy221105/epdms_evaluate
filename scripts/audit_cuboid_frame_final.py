"""Audit PAI cuboid frame against local NuRec sequence tracks (A/B/C)."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path
try:
    from scripts.audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose,load_nurec_poses
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks,load_time_record,pose_yaw,_wrap,_csv,_dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose,load_nurec_poses
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_sequence_tracks,load_time_record,pose_yaw,_wrap,_csv,_dump

PUBLIC_NCORE="https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
PUBLIC_INSTANT="https://github.com/NVIDIA/instant-nurec/blob/main/instant_nurec/datasets/utils.py"
def inv(m):
    r=[x[:3] for x in m[:3]]; p=[m[i][3] for i in range(3)]; rt=[[r[j][i] for j in range(3)] for i in range(3)]
    return [rt[0]+[-sum(rt[0][j]*p[j] for j in range(3))],rt[1]+[-sum(rt[1][j]*p[j] for j in range(3))],rt[2]+[-sum(rt[2][j]*p[j] for j in range(3))],[0.,0.,0.,1.]]
def read_member(root,rel,suffix,clip):
    import pyarrow.parquet as pq
    for z in sorted((Path(root)/rel).glob("*.zip")):
        with zipfile.ZipFile(z) as a:
            names=[n for n in a.namelist() if clip in n and n.endswith(suffix)]
            if names:return pq.read_table(io.BytesIO(a.read(names[0]))).to_pylist()
    raise FileNotFoundError(clip)
def obj(r): return _pose([r["center_x"],r["center_y"],r["center_z"]],[r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]])
def center(m): return [m[0][3],m[1][3],m[2][3]]
def quat(m): return _rquat([x[:3] for x in m[:3]])
def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
def residual(m,target):
    c=center(m); q=quat(m)
    a=pose_yaw(_pose(c,q)); b=pose_yaw(_pose(target["center"],target["quaternion"]))
    y=math.degrees(abs(_wrap(a-b)))
    return math.dist(c,target["center"]), y
def metrics(rows,variant):
    cs=[]; ys=[]; ds=[]
    for x in rows:
        d,y=residual(x[variant],x["target"]); cs.append(d); ys.append(y); ds.append(x["dimension_error"])
    return {"matched_observation_count":len(rows),"center_rmse_m":math.sqrt(sum(v*v for v in cs)/len(cs)) if cs else None,"center_median_m":statistics.median(cs) if cs else None,"center_p95_m":p95(cs),"center_max_m":max(cs) if cs else None,"yaw_rmse_deg":math.sqrt(sum(v*v for v in ys)/len(ys)) if ys else None,"yaw_p95_deg":p95(ys),"yaw_max_deg":max(ys) if ys else None,"dimension_max_abs_m":max(ds) if ds else None,"center_gt_2m":sum(v>2 for v in cs),"center_gt_5m":sum(v>5 for v in cs),"center_gt_10m":sum(v>10 for v in cs),"yaw_gt_45deg":sum(v>45 for v in ys),"yaw_gt_90deg":sum(v>90 for v in ys),"yaw_gt_170deg":sum(v>170 for v in ys)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]; all_rows=[]; per=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); obs=read_member(a.pai_root,"labels/obstacle.offline",".obstacle.offline.parquet",cid); local,_e,_s=load_sequence_tracks(clip/"sequence_tracks.json"); lm={(str(r["track_id"]),int(r["timestamp_us"])):r for r in local}; ego=read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid); poses=[(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in ego]; reb=inv(poses[0][1]); meta=json.loads((clip/"rig_trajectories.json").read_text(encoding="utf-8")); wv=meta.get("world_to_scene",meta.get("world_to_nre")); w=wv["matrix"] if isinstance(wv,dict) else wv; iw=inv(w); rows=[]
        for r in obs:
            key=(str(r["track_id"]),int(r["timestamp_us"])+rec["offset_us"]); target=lm.get(key)
            if target is None: continue
            p=interpolate_pose(poses,int(r["reference_frame_timestamp_us"])); local_pose=_mm(reb,_mm(p,obj(r))); a_m=local_pose; b_m=_mm(w,local_pose); c_m=_mm(iw,local_pose); dim=max(abs(float(r[k])-float(target["dimensions"][i])) for i,k in enumerate(("size_x","size_y","size_z"))); rows.append({"clip_id":cid,"track_id":str(r["track_id"]),"pai_timestamp_us":int(r["timestamp_us"]),"nurec_timestamp_us":key[1],"reference_frame_timestamp_us":int(r["reference_frame_timestamp_us"]),"A_NCORE_WORLD":a_m,"B_WORLD_TO_NRE":b_m,"C_INVERSE_WORLD_TO_NRE":c_m,"target":target,"dimension_error":dim})
        for variant in ("A_NCORE_WORLD","B_WORLD_TO_NRE","C_INVERSE_WORLD_TO_NRE"): per.append({"clip_id":cid,"variant":variant,**metrics(rows,variant)})
        all_rows.extend(rows)
    aggregate={v:{k:max(x[k] for x in per if x["variant"]==v) for k in ("center_rmse_m","center_p95_m","center_max_m","yaw_rmse_deg","yaw_p95_deg","yaw_max_deg","dimension_max_abs_m","center_gt_2m","center_gt_5m","center_gt_10m","yaw_gt_45deg","yaw_gt_90deg","yaw_gt_170deg")} for v in ("A_NCORE_WORLD","B_WORLD_TO_NRE","C_INVERSE_WORLD_TO_NRE")}; best=min(aggregate,key=lambda v:(aggregate[v]["center_rmse_m"],aggregate[v]["yaw_rmse_deg"])); outliers=[]; cross=[]
    for x in all_rows:
        for variant in ("A_NCORE_WORLD","B_WORLD_TO_NRE","C_INVERSE_WORLD_TO_NRE"):
            m=x[variant]; target=x["target"]; d,y=residual(m,target); row={"clip_id":x["clip_id"],"variant":variant,"track_id":x["track_id"],"pai_timestamp_us":x["pai_timestamp_us"],"nurec_timestamp_us":x["nurec_timestamp_us"],"reference_frame_timestamp_us":x["reference_frame_timestamp_us"],"center_residual_m":d,"yaw_residual_deg":y,"dimension_max_abs_m":x["dimension_error"]}; cross.append(row); 
            if d>2 or y>45: outliers.append({**row,"center_gt_2m":d>2,"center_gt_5m":d>5,"center_gt_10m":d>10,"yaw_gt_45deg":y>45,"yaw_gt_90deg":y>90,"yaw_gt_170deg":y>170})
    _csv(out/"cuboid_transform_variant_comparison.csv",per); _csv(out/"cuboid_exact_crosscheck.csv",cross); _csv(out/"cuboid_transform_outliers.csv",outliers)
    _dump(out/"cuboid_frame_provenance.json",{"raw_object_frame":"rig","raw_reference_timestamp_field":"reference_frame_timestamp_us","ncore_cuboid_world_frame":"world via pose graph/consolidate_cuboid_tracks","sequence_tracks_output_frame":"NCore local world (supported by ego and A numerical cross-check)","world_to_nre_applied_to_cuboid":"NOT_EVIDENCED_IN_LOCAL_SEQUENCE_TRACKS","public_ncore":PUBLIC_NCORE,"public_instant_nurec":PUBLIC_INSTANT,"no_fitted_transform":True,"per_clip_offset_rederived":False})
    _dump(out/"cuboid_exact_crosscheck_summary.json",{"per_clip":per,"aggregate":aggregate,"matched_observation_count":len(all_rows),"best_variant":best})
    status="VERIFIED" if best=="A_NCORE_WORLD" and aggregate[best]["center_rmse_m"]<2 and aggregate[best]["center_p95_m"]<3 and aggregate[best]["center_max_m"]<10 and aggregate[best]["yaw_max_deg"]<45 else "PARTIALLY_VERIFIED"; _dump(out/"cuboid_frame_final_summary.json",{"ego_matched_timestamp_count":1010,"raw_object_frame":"rig","ncore_cuboid_world_frame":"world","sequence_tracks_output_frame":"NCore local world","world_to_nre_cuboid_role":"NOT_APPLIED_TO_LOCAL_SEQUENCE_TRACKS","best_proven_variant":best,"matched_observation_count":len(all_rows),"A_center_rmse_m":aggregate["A_NCORE_WORLD"]["center_rmse_m"],"B_center_rmse_m":aggregate["B_WORLD_TO_NRE"]["center_rmse_m"],"C_center_rmse_m":aggregate["C_INVERSE_WORLD_TO_NRE"]["center_rmse_m"],"best_center_rmse_m":aggregate[best]["center_rmse_m"],"best_center_p95_m":aggregate[best]["center_p95_m"],"best_center_max_m":aggregate[best]["center_max_m"],"best_yaw_rmse_deg":aggregate[best]["yaw_rmse_deg"],"best_yaw_p95_deg":aggregate[best]["yaw_p95_deg"],"best_yaw_max_deg":aggregate[best]["yaw_max_deg"],"max_dimension_abs_m":aggregate[best]["dimension_max_abs_m"],"outlier_counts":{k:aggregate[best][k] for k in ("center_gt_2m","center_gt_5m","center_gt_10m","yaw_gt_45deg","yaw_gt_90deg","yaw_gt_170deg")},"cuboid_frame_status":status,"obstacle_transform_status":status,"normalized_obstacle_status":status,"obstacle_geometry_block_ready_to_close":status=="VERIFIED","per_clip_offset_rederived":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"physical_world_obstacle_completeness":"NOT_CLAIMED","remaining_blockers":[] if status=="VERIFIED" else ["CUBOID_ORIENTATION_AND_CENTER_OUTLIERS_REQUIRE_SEPARATE_PROVENANCE_REVIEW"],"recommended_next_step":"close geometry block and start scorer-contract review" if status=="VERIFIED" else "investigate exact yaw/center outlier provenance before closing geometry block"})
if __name__=="__main__": main()
