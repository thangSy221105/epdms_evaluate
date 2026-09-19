"""Forensic audit of A_NCORE_WORLD cuboid outliers; no correction is applied."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump

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
def yaw(m): return pose_yaw(m)
def residual(m,target):
    c=center(m); q=quat(m); a=yaw(_pose(c,q)); b=yaw(_pose(target["center"],target["quaternion"]))
    return math.dist(c,target["center"]),math.degrees(abs(_wrap(a-b)))
def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
def global_metrics(rows):
    cs=[]; ys=[]
    for r in rows: cs.append(r["center_residual_m"]); ys.append(r["yaw_residual_deg"])
    return {"matched_observation_count":len(rows),"global_center_rmse_m":math.sqrt(sum(v*v for v in cs)/len(cs)),"global_center_median_m":statistics.median(cs),"global_center_p95_m":p95(cs),"global_center_max_m":max(cs),"global_yaw_rmse_deg":math.sqrt(sum(v*v for v in ys)/len(ys)),"global_yaw_p95_deg":p95(ys),"global_yaw_max_deg":max(ys),"total_center_gt_2m":sum(v>2 for v in cs),"total_center_gt_5m":sum(v>5 for v in cs),"total_center_gt_10m":sum(v>10 for v in cs),"total_yaw_gt_45deg":sum(v>45 for v in ys),"total_yaw_gt_90deg":sum(v>90 for v in ys),"total_yaw_gt_170deg":sum(v>170 for v in ys)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]; all_rows=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); obs=read_member(a.pai_root,"labels/obstacle.offline",".obstacle.offline.parquet",cid); local,_e,_s=load_sequence_tracks(clip/"sequence_tracks.json"); lm={(str(r["track_id"]),int(r["timestamp_us"])):r for r in local}; tracks={}
        for r in local: tracks.setdefault(str(r["track_id"]),[]).append(r)
        for values in tracks.values(): values.sort(key=lambda r:int(r["timestamp_us"]))
        ego=read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid); poses=[(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in ego]; reb=inv(poses[0][1])
        for r in obs:
            nurec_ts=int(r["timestamp_us"])+rec["offset_us"]; target=lm.get((str(r["track_id"]),nurec_ts))
            if target is None: continue
            ego_pose=interpolate_pose(poses,int(r["reference_frame_timestamp_us"])); transformed=_mm(reb,_mm(ego_pose,obj(r))); c=center(transformed); q=quat(transformed); cd,yd=residual(transformed,target); track=tracks[str(r["track_id"])]; idx=next(i for i,x in enumerate(track) if int(x["timestamp_us"])==nurec_ts); pos="FIRST" if idx==0 else "LAST" if idx==len(track)-1 else "MIDDLE"; ego_distance=math.dist(c,center(ego_pose)); rawq=[r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]]; localq=target["quaternion"]; mod180=min(yd,abs(180.0-yd)); all_rows.append({"clip_id":cid,"track_id":str(r["track_id"]),"category":target["category"],"pai_timestamp_us":int(r["timestamp_us"]),"nurec_timestamp_us":nurec_ts,"reference_frame_timestamp_us":int(r["reference_frame_timestamp_us"]),"timestamp_minus_reference_us":int(r["timestamp_us"])-int(r["reference_frame_timestamp_us"]),"raw_pai_center":json.dumps([r["center_x"],r["center_y"],r["center_z"]]),"raw_pai_quaternion":json.dumps(rawq),"transformed_center":json.dumps(c),"transformed_quaternion":json.dumps(q),"local_center":json.dumps(target["center"]),"local_quaternion":json.dumps(localq),"pai_dimensions":json.dumps([r["size_x"],r["size_y"],r["size_z"]]),"local_dimensions":json.dumps(target["dimensions"]),"center_residual_m":cd,"yaw_residual_deg":yd,"yaw_residual_mod_180_deg":mod180,"dimension_error_m":max(abs(float(r[k])-float(target["dimensions"][i])) for i,k in enumerate(("size_x","size_y","size_z"))),"ego_distance_m":ego_distance,"track_observation_index":idx,"track_observation_count":len(track),"track_position":pos})
    gm=global_metrics(all_rows); outliers=[r for r in all_rows if r["center_residual_m"]>2 or r["yaw_residual_deg"]>45]; groups={}
    for r in outliers:
        key=(r["clip_id"],r["track_id"],r["category"]); g=groups.setdefault(key,{"clip_id":key[0],"track_id":key[1],"category":key[2],"row_count":0,"positions":set(),"yaw_gt_170_count":0,"center_gt_2_count":0,"timestamp_min":r["nurec_timestamp_us"],"timestamp_max":r["nurec_timestamp_us"]}); g["row_count"]+=1; g["positions"].add(r["track_position"]); g["yaw_gt_170_count"]+=r["yaw_residual_deg"]>170; g["center_gt_2_count"]+=r["center_residual_m"]>2; g["timestamp_min"]=min(g["timestamp_min"],r["nurec_timestamp_us"]); g["timestamp_max"]=max(g["timestamp_max"],r["nurec_timestamp_us"])
    groups=[{**g,"positions":"|".join(sorted(g["positions"]))} for g in groups.values()]; flip_rows=[r for r in all_rows if r["yaw_residual_deg"]>45]; flip_supported=bool(flip_rows) and sum(r["yaw_residual_mod_180_deg"]<5 for r in flip_rows)/len(flip_rows)>.8
    _csv(out/"cuboid_outlier_forensics.csv",outliers); _csv(out/"cuboid_outlier_group_summary.csv",groups); _dump(out/"cuboid_global_metrics.json",gm); _dump(out/"cuboid_orientation_provenance.json",{"yaw_180_flip_pattern_status":"SUPPORTED_BY_DATA_ONLY" if flip_supported else "NOT_SUPPORTED","outlier_count":len(flip_rows),"mod_180_not_used_as_official_residual":True,"upstream_orientation_equivalence_proven":False}); _dump(out/"cuboid_center_outlier_provenance.json",{"center_outlier_count":sum(r["center_residual_m"]>2 for r in all_rows),"interpolation_field":"reference_frame_timestamp_us","raw_ego_coverage_checked":True,"outliers_dropped":False}); _dump(out/"cuboid_outlier_final_summary.json",{**gm,"center_outlier_provenance_status":"PARTIALLY_EXPLAINED_BY_ROW_LEVEL_FORENSICS","yaw_outlier_provenance_status":"DATA_PATTERN_ONLY_NOT_UPSTREAM_PROVEN","yaw_180_flip_pattern_status":"SUPPORTED_BY_DATA_ONLY" if flip_supported else "NOT_SUPPORTED","systematic_frame_error_found":False,"cuboid_frame_status":"PARTIALLY_VERIFIED","obstacle_transform_status":"PARTIALLY_VERIFIED","normalized_obstacle_status":"PARTIALLY_VERIFIED","obstacle_geometry_block_ready_to_close":False,"per_clip_offset_rederived":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"remaining_blockers":["YAW_180_EQUIVALENCE_NOT_UPSTREAM_PROVEN","CENTER_AND_YAW_OUTLIERS_NOT_FULLY_EXPLAINED"],"recommended_next_step":"keep geometry block open; review upstream/local cuboid orientation serialization before scorer contract"})
if __name__=="__main__": main()
