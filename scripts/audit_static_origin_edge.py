"""Characterize a possible static NuRec origin edge from ego poses only."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, inverse_pose, interpolate_pose, load_nurec_poses
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_time_record, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, inverse_pose, interpolate_pose, load_nurec_poses
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_time_record, _csv, _dump

def read_ego(root, clip):
    import pyarrow.parquet as pq
    for z in sorted((Path(root)/"labels/egomotion.offline").glob("*.zip")):
        with zipfile.ZipFile(z) as a:
            names=[n for n in a.namelist() if clip in n and n.endswith(".egomotion.offline.parquet")]
            if names:
                rows=pq.read_table(io.BytesIO(a.read(names[0]))).to_pylist()
                return [(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in rows]
    raise FileNotFoundError(clip)

def center(m): return [m[0][3],m[1][3],m[2][3]]
def yaw(m):
    r=m; return math.atan2(r[1][0],r[0][0])
def wrap(x): return math.atan2(math.sin(x),math.cos(x))
def inv(m): return inverse_pose(m)
def stats(v):
    return {"mean":statistics.mean(v) if v else None,"std":statistics.pstdev(v) if len(v)>1 else 0.0 if v else None,"min":min(v) if v else None,"max":max(v) if v else None}
def static_decision(max_translation_std, max_yaw_std, max_yaw):
    return max_translation_std < 0.5 and max_yaw_std < 1.0 and max_yaw < 2.0

def world_to_nre(meta):
    value=meta.get("world_to_scene",meta.get("world_to_nre"))
    return value["matrix"] if isinstance(value,dict) else value

def camera_end(meta):
    values=[]
    for seq in meta.get("rig_trajectories",[]):
        for frames in seq.get("cameras_frame_timestamps_us",{}).values():
            if frames: values.append(int(frames[0][1]))
    return min(values) if values else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]
    delta_rows=[]; summaries=[]; refs=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); raw=read_ego(a.pai_root,cid); local=load_nurec_poses(clip); meta=json.loads((clip/"rig_trajectories.json").read_text(encoding="utf-8")); w=world_to_nre(meta); first=raw[0][1]; rebase=inv(first); candidates=[]
        for ts,p in raw:
            local_time=ts+rec["offset_us"]; candidate=_mm(w,_mm(rebase,p)); local_pose=interpolate_pose(local,local_time); delta=_mm(local_pose,inv(candidate)); c=center(delta); yd=math.degrees(abs(wrap(yaw(delta)))); candidates.append((ts,local_time,local_pose,candidate,delta,c,yd))
            delta_rows.append({"clip_id":cid,"pai_timestamp_us":ts,"nurec_timestamp_us":local_time,"delta_x":c[0],"delta_y":c[1],"delta_z":c[2],"delta_translation_norm":math.dist(c,[0,0,0]),"delta_yaw_deg":yd})
        xs=[x[5][0] for x in candidates]; ys=[x[5][1] for x in candidates]; zs=[x[5][2] for x in candidates]; norms=[math.dist(x[5],[0,0,0]) for x in candidates]; yaws=[x[6] for x in candidates]; mid=candidates[len(candidates)//2]
        def snap(x): return {"pai_timestamp_us":x[0],"nurec_timestamp_us":x[1],"T_local":x[2],"T_candidate":x[3],"delta_T":x[4]}
        t0_pose=interpolate_pose(local,rec["nurec_t0_us"]); t0_candidate=_mm(w,_mm(rebase,raw[0][1]+[])) if False else _mm(w,rebase)
        first_context=camera_end(meta)
        refs.append({"clip_id":cid,"pai_first_ego_timestamp_us":raw[0][0],"ncore_first_rebased_timestamp_us":raw[0][0],"local_nurec_first_rig_timestamp_us":local[0][0],"first_context_camera_end_us":first_context,"nurec_t0_us":rec["nurec_t0_us"],"T_local_first":local[0][1],"T_candidate_first":candidates[0][3],"delta_first":candidates[0][4],"T_local_at_nurec_t0":t0_pose,"T_candidate_at_nurec_t0":_mm(w,interpolate_pose([(t,rebase and _mm(rebase,p)) for t,p in raw],rec["nurec_t0_us"]-rec["offset_us"])),"delta_at_nurec_t0":_mm(t0_pose,inv(_mm(w,interpolate_pose([(t,_mm(rebase,p)) for t,p in raw],rec["nurec_t0_us"]-rec["offset_us"]))))})
        summaries.append({"clip_id":cid,"matched_timestamp_count":len(candidates),"delta_translation_mean_xyz":[statistics.mean(xs),statistics.mean(ys),statistics.mean(zs)],"delta_translation_std_xyz":[statistics.pstdev(xs),statistics.pstdev(ys),statistics.pstdev(zs)],"delta_translation_norm":stats(norms),"delta_yaw_mean_deg":statistics.mean(yaws),"delta_yaw_std_deg":statistics.pstdev(yaws),"delta_yaw_max_deg":max(yaws),"snapshots":{"first":snap(candidates[0]),"middle":snap(mid),"last":snap(candidates[-1])}})
    max_std=max(max(x["delta_translation_std_xyz"]) for x in summaries); max_range=max(max(x["delta_translation_norm"]["max"]-x["delta_translation_norm"]["min"] for x in summaries),0); max_yaw_std=max(x["delta_yaw_std_deg"] for x in summaries); max_yaw=max(x["delta_yaw_max_deg"] for x in summaries); static=static_decision(max_std,max_yaw_std,max_yaw)
    _csv(out/"ego_delta_transform.csv",delta_rows); _csv(out/"reference_timestamp_inventory.csv",refs)
    _dump(out/"ego_delta_transform_summary.json",{"clips":summaries,"matched_timestamp_count":len(delta_rows),"max_delta_translation_std_m":max_std,"max_delta_translation_range_m":max_range,"max_delta_yaw_std_deg":max_yaw_std,"max_delta_yaw_max_deg":max_yaw,"static_origin_edge_status":"STRONGLY_SUPPORTED" if static else "REJECTED"})
    _dump(out/"static_origin_edge_contract.json",{"status":"STRONGLY_SUPPORTED" if static else "REJECTED","formula":"delta_T(t)=T_local_nurec(t) @ inverse(world_to_nre @ T_rig_world_local(t))","no_fitted_transform":True,"offset_rederived":False,"decision_thresholds":{"translation_std_m":0.5,"yaw_std_deg":1.0,"yaw_max_deg":2.0}})
    _dump(out/"static_origin_edge_final_summary.json",{"ego_matched_timestamp_count":len(delta_rows),"static_origin_edge_status":"STRONGLY_SUPPORTED" if static else "REJECTED","max_delta_translation_std_m":max_std,"max_delta_translation_range_m":max_range,"max_delta_yaw_std_deg":max_yaw_std,"max_delta_yaw_max_deg":max_yaw,"reference_timestamp_origin_status":"RECORDED_NOT_PROVEN","obstacle_transform_status":"BLOCKED_BY_ORIGIN_EDGE","obstacle_geometry_block_ready_to_close":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"per_clip_offset_rederived":False,"remaining_blockers":["STATIC_ORIGIN_EDGE_NOT_PROVEN" if not static else "ORIGIN_EDGE_NOT_APPLIED_OR_PROVEN_FOR_OBSTACLES"],"recommended_next_step":"review timestamp-origin evidence and obtain explicit serializer contract before applying any origin edge"})

if __name__=="__main__": main()
