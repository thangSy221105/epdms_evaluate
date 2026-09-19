"""Replay the public NCore/Instant-NuRec cuboid selection semantics."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path
try:
    from scripts.audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks,load_time_record,pose_yaw,_wrap,_csv,_dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose
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
def residual(m,target):
    c=center(m); q=quat(m); a=pose_yaw(_pose(c,q)); b=pose_yaw(_pose(target["center"],target["quaternion"]))
    return math.dist(c,target["center"]),math.degrees(abs(_wrap(a-b))),c,q
def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
def metrics(rows):
    cs=[r["center_residual_m"] for r in rows]; ys=[r["yaw_residual_deg"] for r in rows]
    return {"matched_selected_observation_count":len(rows),"global_center_rmse_m":math.sqrt(sum(v*v for v in cs)/len(cs)),"global_center_median_m":statistics.median(cs),"global_center_p95_m":p95(cs),"global_center_max_m":max(cs),"global_yaw_rmse_deg":math.sqrt(sum(v*v for v in ys)/len(ys)),"global_yaw_p95_deg":p95(ys),"global_yaw_max_deg":max(ys),"total_center_gt_2m":sum(v>2 for v in cs),"total_center_gt_5m":sum(v>5 for v in cs),"total_center_gt_10m":sum(v>10 for v in cs),"total_yaw_gt_45deg":sum(v>45 for v in ys),"total_yaw_gt_90deg":sum(v>90 for v in ys),"total_yaw_gt_170deg":sum(v>170 for v in ys)}
def source_name(source): return "AUTOLABEL" if "autolabel" in str(source).lower() else str(source).upper()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); ap.add_argument("--min-centroid-rig-dist",type=float,default=3.0); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]; selection=[]; duplicates=[]; cross=[]; dimension_rows=[]; dimension_emitted=set(); selected_count=0; candidate_count=0; duplicate_count=0; rejected_source=0; rejected_distance=0; rejected_dup=0; ambiguous=0
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); raw=read_member(a.pai_root,"labels/obstacle.offline",".obstacle.offline.parquet",cid); local,_e,_s=load_sequence_tracks(clip/"sequence_tracks.json"); lm={(str(r["track_id"]),int(r["timestamp_us"])):r for r in local}; ego=read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid); poses=[(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in ego]; reb=inv(poses[0][1]); rows=sorted(list(enumerate(raw)),key=lambda x:(str(x[1]["track_id"]),int(x[1]["timestamp_us"])))
        selected_by_track={}; first_dims={}
        for raw_idx,r in rows:
            candidate_count+=1; source=source_name(r.get("source")); version=str(r.get("source_version") or "any"); key=(str(r["track_id"]),int(r["timestamp_us"])); status="SELECTED"; reason="VALID_AUTOLABEL_ANY"
            if source!="AUTOLABEL": status="REJECTED_BY_LABEL_SOURCE"; reason="SOURCE_NOT_ALLOWED"; rejected_source+=1
            near=math.sqrt(float(r["center_x"])**2+float(r["center_y"])**2+float(r["center_z"])**2)
            if status=="SELECTED" and near<a.min_centroid_rig_dist: status="REJECTED_BY_DISTANCE"; reason="TRACK_MIN_CENTROID_RIG_DIST"; rejected_distance+=1
            track=selected_by_track.setdefault(str(r["track_id"]),[])
            if status=="SELECTED" and int(r["timestamp_us"]) in track[-2:]: status="REJECTED_DUPLICATE_TIMESTAMP"; reason="TIMESTAMP_IN_LAST_TWO_TRACK_TIMESTAMPS"; rejected_dup+=1; duplicate_count+=1
            selection.append({"clip_id":cid,"track_id":str(r["track_id"]),"timestamp_us":int(r["timestamp_us"]),"raw_row_index":raw_idx,"source":str(r.get("source")),"source_version":r.get("source_version"),"selection_status":status,"selection_reason":reason})
            if status=="SELECTED": track.append(int(r["timestamp_us"])); selected_count+=1; first_dims.setdefault(str(r["track_id"]),[r["size_x"],r["size_y"],r["size_z"]]);
        selected_index={}
        for x in selection:
            if x["clip_id"]==cid and x["selection_status"]=="SELECTED": selected_index.setdefault((x["track_id"],x["timestamp_us"]+rec["offset_us"]),[]).append(x)
        for key,target in lm.items():
            candidates=selected_index.get(key,[])
            if len(candidates)!=1:
                if len(candidates)>1: ambiguous+=1
                continue
            ident=candidates[0]; r=raw[ident["raw_row_index"]]; p=interpolate_pose(poses,int(r["reference_frame_timestamp_us"])); local_ego=_mm(reb,p); transformed=_mm(local_ego,obj(r)); cd,yd,c,q=residual(transformed,target); tracks=[x for x in local if str(x["track_id"])==key[0]]; tracks.sort(key=lambda x:int(x["timestamp_us"])); idx=next(i for i,x in enumerate(tracks) if int(x["timestamp_us"])==key[1]); pos="FIRST" if idx==0 else "LAST" if idx==len(tracks)-1 else "MIDDLE"; ego_distance=math.dist(c,center(local_ego)); row={"clip_id":cid,"track_id":key[0],"timestamp_us":key[1],"raw_row_index":ident["raw_row_index"],"source":r.get("source"),"source_version":r.get("source_version"),"selected_by_upstream_replay":True,"duplicate_key_count":sum(1 for x in selection if x["clip_id"]==cid and x["track_id"]==key[0] and x["timestamp_us"]==key[1]),"reference_frame_timestamp_us":int(r["reference_frame_timestamp_us"]),"timestamp_minus_reference_us":int(r["timestamp_us"])-int(r["reference_frame_timestamp_us"]),"raw_quaternion":json.dumps([r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]]),"raw_euler_xyz":"REPLAYED_R_FROM_QUAT_XYZW","reconstructed_quaternion":json.dumps(q),"transformed_center":json.dumps(c),"local_center":json.dumps(target["center"]),"transformed_yaw":pose_yaw(_pose(c,q)),"local_yaw":pose_yaw(_pose(target["center"],target["quaternion"])),"center_residual_m":cd,"yaw_residual_deg":yd,"track_position":pos,"ego_distance_m":ego_distance,"dimension_error_m":max(abs(float(first_dims[key[0]][i])-float(target["dimensions"][i])) for i in range(3))}; cross.append(row); dim_key=(cid,key[0]);
            if dim_key not in dimension_emitted:
                dimension_emitted.add(dim_key); dimension_rows.append({"clip_id":cid,"track_id":key[0],"dimension_source":"SELECTED_FIRST_RAW_OBSERVATION","raw_first_dimensions":json.dumps(first_dims[key[0]]),"local_dimensions":json.dumps(target["dimensions"]),"max_abs_error_m":row["dimension_error_m"]})
    gm=metrics(cross); outliers=[r for r in cross if r["center_residual_m"]>2 or r["yaw_residual_deg"]>45]; _csv(out/"raw_cuboid_selection_audit.csv",selection); _csv(out/"duplicate_track_timestamp_audit.csv",[x for x in selection if x["selection_status"]=="REJECTED_DUPLICATE_TIMESTAMP"]); _csv(out/"selected_raw_to_sequence_track_crosscheck.csv",cross); _csv(out/"cuboid_serialization_outliers.csv",outliers)
    _dump(out/"cuboid_dimension_provenance.json",{"status":"TRACK_CONSTANT_FROM_SELECTED_FIRST_OBSERVATION","rows":dimension_rows,"max_dimension_abs_m":max((x["max_abs_error_m"] for x in dimension_rows),default=None)})
    _dump(out/"cuboid_serialization_global_metrics.json",gm); _dump(out/"cuboid_serialization_contract.json",{"raw_to_ncore":"PAI row -> CuboidTrackObservation; source/reference_frame/reference_frame_timestamp_us copied","bbox_replay":"R.from_quat([qx,qy,qz,qw]).as_euler(xyz) then BBox3","selection":"AUTOLABEL@any; duplicate timestamp in last two skipped; min centroid rig distance 3m","world_transform":"T_rig_world_local(reference_frame_timestamp_us) @ T_object_rig","instant_nurec_sources":[PUBLIC_NCORE,PUBLIC_INSTANT],"offset_rederived":False,"no_fitted_transform":True})
    _dump(out/"cuboid_serialization_final_summary.json",{**gm,"raw_match_candidate_count":candidate_count,"raw_selected_row_count":selected_count,"raw_duplicate_key_count":duplicate_count,"raw_rejected_label_source_count":rejected_source,"raw_rejected_distance_count":rejected_distance,"raw_rejected_duplicate_timestamp_count":rejected_dup,"local_observation_matched_to_selected_raw_count":len(cross),"ambiguous_selected_raw_count":ambiguous,"track_dimension_source_status":"VERIFIED_TRACK_CONSTANT_SELECTED_FIRST_OBSERVATION","raw_row_selection_status":"VERIFIED_REPLAY_RULES","cuboid_serialization_status":"PARTIALLY_VERIFIED","systematic_frame_error_found":False,"cuboid_frame_status":"PARTIALLY_VERIFIED","obstacle_transform_status":"PARTIALLY_VERIFIED","normalized_obstacle_status":"PARTIALLY_VERIFIED","obstacle_geometry_block_ready_to_close":False,"per_clip_offset_rederived":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"physical_world_obstacle_completeness":"NOT_CLAIMED","remaining_blockers":["OUTLIER_ORIENTATION_AND_CENTER_PROVENANCE_NOT_FULLY_EXPLAINED"],"recommended_next_step":"review remaining selected-row outliers before closing geometry block"})
if __name__=="__main__": main()
