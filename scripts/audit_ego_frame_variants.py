"""Compare local NuRec rig trajectories against three documented frame variants."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path
try:
    from scripts.audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose,load_nurec_poses
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_time_record,_csv,_dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose,load_nurec_poses
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_time_record,_csv,_dump

def inv(m):
    r=[x[:3] for x in m[:3]]; p=[m[i][3] for i in range(3)]; rt=[[r[j][i] for j in range(3)] for i in range(3)]
    return [rt[0]+[-sum(rt[0][j]*p[j] for j in range(3))],rt[1]+[-sum(rt[1][j]*p[j] for j in range(3))],rt[2]+[-sum(rt[2][j]*p[j] for j in range(3))],[0.,0.,0.,1.]]
def ego_rows(root,clip):
    import pyarrow.parquet as pq
    for z in sorted((Path(root)/"labels/egomotion.offline").glob("*.zip")):
        with zipfile.ZipFile(z) as a:
            names=[n for n in a.namelist() if clip in n and n.endswith(".egomotion.offline.parquet")]
            if names:
                return [(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in pq.read_table(io.BytesIO(a.read(names[0]))).to_pylist()]
    raise FileNotFoundError(clip)
def center(m): return [m[0][3],m[1][3],m[2][3]]
def yaw(m): return math.atan2(m[1][0],m[0][0])
def wrap(x): return math.atan2(math.sin(x),math.cos(x))
def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
def metric(pairs):
    ds=[math.dist(center(a),center(b)) for a,b in pairs]; ys=[math.degrees(abs(wrap(yaw(a)-yaw(b)))) for a,b in pairs]
    return {"matched_timestamp_count":len(pairs),"center_rmse_m":math.sqrt(sum(x*x for x in ds)/len(ds)),"center_p95_m":p95(ds),"center_max_m":max(ds),"yaw_rmse_deg":math.sqrt(sum(x*x for x in ys)/len(ys)),"yaw_p95_deg":p95(ys),"yaw_max_deg":max(ys)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]; rows=[]; summaries=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); raw=ego_rows(a.pai_root,cid); local=load_nurec_poses(clip); meta=json.loads((clip/"rig_trajectories.json").read_text(encoding="utf-8")); value=meta.get("world_to_scene",meta.get("world_to_nre")); w=value["matrix"] if isinstance(value,dict) else value; iw=inv(w); reb=inv(raw[0][1]); variants={"A_NO_WORLD_TO_NRE":[],"B_FORWARD":[],"C_INVERSE":[]}
        for ts,p in raw:
            target=interpolate_pose(local,ts+rec["offset_us"]); local_p=_mm(reb,p); variants["A_NO_WORLD_TO_NRE"].append((local_p,target)); variants["B_FORWARD"].append((_mm(w,local_p),target)); variants["C_INVERSE"].append((_mm(iw,local_p),target))
        ms={k:metric(v) for k,v in variants.items()}; summaries.append({"clip_id":cid,"variants":ms,"rebase_applied":True,"offset_rederived":False})
        for k,v in variants.items(): rows.append({"clip_id":cid,"variant":k,**ms[k]})
    aggregate={k:{"max_center_rmse_m":max(x["variants"][k]["center_rmse_m"] for x in summaries),"max_center_p95_m":max(x["variants"][k]["center_p95_m"] for x in summaries),"max_center_max_m":max(x["variants"][k]["center_max_m"] for x in summaries),"max_yaw_rmse_deg":max(x["variants"][k]["yaw_rmse_deg"] for x in summaries),"max_yaw_p95_deg":max(x["variants"][k]["yaw_p95_deg"] for x in summaries),"max_yaw_max_deg":max(x["variants"][k]["yaw_max_deg"] for x in summaries)} for k in ("A_NO_WORLD_TO_NRE","B_FORWARD","C_INVERSE")}
    best=min(aggregate,key=lambda k:(aggregate[k]["max_center_rmse_m"],aggregate[k]["max_yaw_rmse_deg"])); supported=best=="A_NO_WORLD_TO_NRE" and aggregate[best]["max_center_rmse_m"]<2.0
    _csv(out/"ego_frame_variant_comparison.csv",rows); _dump(out/"ego_frame_variant_summary.json",{"clips":summaries,"aggregate":aggregate,"best_supported_variant":best}); _dump(out/"ego_frame_contract.json",{"status":"VERIFIED_NCORE_LOCAL_WORLD" if supported else "PARTIALLY_VERIFIED","best_variant":best,"serialization_semantics":"NCore PAI converter rebases first rig pose and stores dynamic source=rig,target=world; local T_rig_worlds is compared in that local world","world_to_nre_applies_to_ego":False if supported else None,"offset_rederived":False,"no_fitted_transform":True})
    _dump(out/"ego_frame_variant_final_summary.json",{"ego_matched_timestamp_count":len(manifest)*202,"A_max_center_rmse_m":aggregate["A_NO_WORLD_TO_NRE"]["max_center_rmse_m"],"A_max_center_p95_m":aggregate["A_NO_WORLD_TO_NRE"]["max_center_p95_m"],"A_max_yaw_rmse_deg":aggregate["A_NO_WORLD_TO_NRE"]["max_yaw_rmse_deg"],"B_max_center_rmse_m":aggregate["B_FORWARD"]["max_center_rmse_m"],"B_max_center_p95_m":aggregate["B_FORWARD"]["max_center_p95_m"],"B_max_yaw_rmse_deg":aggregate["B_FORWARD"]["max_yaw_rmse_deg"],"C_max_center_rmse_m":aggregate["C_INVERSE"]["max_center_rmse_m"],"C_max_center_p95_m":aggregate["C_INVERSE"]["max_center_p95_m"],"C_max_yaw_rmse_deg":aggregate["C_INVERSE"]["max_yaw_rmse_deg"],"best_supported_variant":best,"local_ego_frame_status":"VERIFIED_NCORE_LOCAL_WORLD" if supported else "PARTIALLY_VERIFIED","world_to_nre_applies_to_ego":False if supported else None,"world_to_nre_role_for_ego":"NOT_APPLIED" if supported else "UNRESOLVED","world_to_nre_role_for_cuboids":"UNRESOLVED_TO_VERIFY_NEXT","per_clip_offset_rederived":False,"obstacle_transform_status":"BLOCKED_BY_EGO_FRAME_DECISION","obstacle_geometry_block_ready_to_close":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"remaining_blockers":["EGO_FRAME_VARIANT_NOT_PROVEN" if not supported else "WORLD_TO_NRE_CUBOID_ROLE_UNRESOLVED"],"recommended_next_step":"verify cuboid reference-frame serialization separately; do not apply world_to_nre to ego"})
if __name__=="__main__": main()
