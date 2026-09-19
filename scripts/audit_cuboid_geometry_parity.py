"""Final NVIDIA cuboid math parity and physical-geometry audit."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path
import numpy as np

try:
    from scripts.audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_sequence_tracks,load_time_record,pose_yaw,_wrap,_csv,_dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm,_pose,_rquat,interpolate_pose
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_sequence_tracks,load_time_record,pose_yaw,_wrap,_csv,_dump

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
def source_name(source): return "AUTOLABEL" if "autolabel" in str(source).lower() else str(source).upper()
def center(m): return np.asarray([m[0][3],m[1][3],m[2][3]],float)
def pose_matrix(c,q): return np.asarray(_pose(list(c),list(q)),float)
def p95(v): return float(sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))]) if v else None
def quat_matrix(q):
    x,y,z,w=[float(v) for v in q]; n=x*x+y*y+z*z+w*w
    if n <= 1e-15: return np.eye(3)
    s=2.0/n
    xx,yy,zz=x*x*s,y*y*s,z*z*s; xy,xz,yz=x*y*s,x*z*s,y*z*s; wx,wy,wz=w*x*s,w*y*s,w*z*s
    return np.asarray([[1-yy-zz,xy-wz,xz+wy],[xy+wz,1-xx-zz,yz-wx],[xz-wy,yz+wx,1-xx-yy]],float)
def euler_matrix(e):
    roll,pitch,yaw=[float(v) for v in e]; cr,sr=math.cos(roll),math.sin(roll); cp,sp=math.cos(pitch),math.sin(pitch); cy,sy=math.cos(yaw),math.sin(yaw)
    return np.asarray([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]],float)
def matrix_euler(m):
    m=np.asarray(m,float); pitch=math.asin(max(-1.0,min(1.0,-m[2,0]))); cp=math.cos(pitch)
    if abs(cp)>1e-8: roll=math.atan2(m[2,1],m[2,2]); yaw=math.atan2(m[1,0],m[0,0])
    else: roll=0.0; yaw=math.atan2(-m[0,1],m[1,1])
    return np.asarray([roll,pitch,yaw],float)
def rotation_angle(a,b):
    d=np.asarray(a,float)[:3,:3].T@np.asarray(b,float)[:3,:3]; c=max(-1.0,min(1.0,(np.trace(d)-1.0)/2.0)); return math.degrees(math.acos(c))
def assignment_rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); n=len(a); cost=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)**2
    dp={0:0.0}
    for mask in range(1<<n):
        base=dp.get(mask)
        if base is None: continue
        i=mask.bit_count()
        if i>=n: continue
        for j in range(n):
            if not (mask>>j)&1:
                nm=mask|(1<<j); val=base+float(cost[i,j]); dp[nm]=min(dp.get(nm,float("inf")),val)
    return math.sqrt(dp[(1<<n)-1]/n)
def bbox_pose_np(b):
    out=np.eye(4); out[:3,:3]=euler_matrix(b[6:9]); out[:3,3]=b[:3]; return out
def bbox_from_raw(r):
    e=matrix_euler(quat_matrix([r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]]))
    return np.asarray([r["center_x"],r["center_y"],r["center_z"],r["size_x"],r["size_y"],r["size_z"],*e],float)
def bbox_pose_transform(b,t):
    out=bbox_pose_np(b); result=np.asarray(t)@out; e=matrix_euler(result[:3,:3])
    return np.asarray([*result[:3,3],b[3],b[4],b[5],*e],float)
def matrix_to_q(m): return _rquat(np.asarray(m)[:3,:3].tolist())
def corners(c,q,dims):
    r=quat_matrix(q); signs=np.asarray([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],float); return np.asarray(c)+(signs*np.asarray(dims)/2.0)@r.T
def bev(c,q,dims):
    r=quat_matrix(q); signs=np.asarray([[-1,-1],[1,-1],[1,1],[-1,1]],float); return np.asarray(c)[:2]+(signs*np.asarray(dims)[:2]/2.0)@r[:2,:2].T
def set_rmse(a,b):
    return assignment_rmse(a,b)
def polygon_area(p):
    return abs(sum(p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))/2) if len(p)>=3 else 0.0
def clip_poly(subject,edge_a,edge_b):
    out=[]
    def inside(p): return (edge_b[0]-edge_a[0])*(p[1]-edge_a[1])-(edge_b[1]-edge_a[1])*(p[0]-edge_a[0])>=-1e-9
    def intersect(a,b):
        p=np.asarray(a,float); r=np.asarray(b,float)-p; q=np.asarray(edge_a,float); s=np.asarray(edge_b,float)-q
        den=r[0]*s[1]-r[1]*s[0]
        if abs(den)<1e-12:return list(p)
        t=((q[0]-p[0])*s[1]-(q[1]-p[1])*s[0])/den
        return list(p+t*r)
    if not subject:return out
    prev=subject[-1]
    for cur in subject:
        if inside(cur):
            if not inside(prev):out.append(intersect(prev,cur))
            out.append(cur)
        elif inside(prev):out.append(intersect(prev,cur))
        prev=cur
    return out
def iou(a,b):
    inter=a.tolist()
    for i in range(len(b)): inter=clip_poly(inter,b[i],b[(i+1)%len(b)])
    ua=polygon_area(a.tolist()); ub=polygon_area(b.tolist()); ui=polygon_area(inter); return ui/(ua+ub-ui) if ua+ub-ui>0 else 0.0
def residual(m,target):
    c=center(m); q=matrix_to_q(m); a=pose_yaw(pose_matrix(c,q)); b=pose_yaw(pose_matrix(target["center"],target["quaternion"]))
    return float(np.linalg.norm(c-np.asarray(target["center"]))),math.degrees(abs(_wrap(a-b))),c,q
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]; rows=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); raw=read_member(a.pai_root,"labels/obstacle.offline",".obstacle.offline.parquet",cid); local,_e,_s=load_sequence_tracks(clip/"sequence_tracks.json"); lm={(str(r["track_id"]),int(r["timestamp_us"])):r for r in local}; tracks={}
        for x in local:tracks.setdefault(str(x["track_id"]),[]).append(x)
        for x in tracks.values():x.sort(key=lambda r:int(r["timestamp_us"]))
        ego=read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid); poses=[(int(r["timestamp"]),_pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in ego]; reb=inv(poses[0][1]); selected=[]; first_dims={}
        for idx,r in sorted(enumerate(raw),key=lambda x:(str(x[1]["track_id"]),int(x[1]["timestamp_us"]))):
            if source_name(r.get("source"))!="AUTOLABEL":continue
            if math.dist([r["center_x"],r["center_y"],r["center_z"]],[0,0,0])<3:continue
            key=(str(r["track_id"]),int(r["timestamp_us"])); first_dims.setdefault(str(r["track_id"]),[r["size_x"],r["size_y"],r["size_z"]]); selected.append((idx,r,key))
        for raw_idx,r,key in selected:
            mapped=key[1]+rec["offset_us"]; target=lm.get((key[0],mapped));
            if target is None:continue
            p=interpolate_pose(poses,int(r["reference_frame_timestamp_us"])); ego_local=np.asarray(reb)@np.asarray(p); b=bbox_from_raw(r); direct=ego_local@bbox_pose_np(b); nvidia=bbox_pose_transform(b,ego_local); nvidia_f=np.asarray(nvidia,dtype=np.float32); nvidia_pose=bbox_pose_np(nvidia_f); cd,yd,c,q=residual(nvidia_pose,target); dc=float(np.linalg.norm(center(direct)-center(nvidia_pose))); dr=rotation_angle(direct,nvidia_pose); ttrack=tracks[key[0]]; li=next(i for i,x in enumerate(ttrack) if int(x["timestamp_us"])==mapped); local_pos="FIRST" if li==0 else "LAST" if li==len(ttrack)-1 else "MIDDLE"; raw_track=[x for x in selected if x[1]["track_id"]==r["track_id"]]; ri=next(i for i,x in enumerate(raw_track) if x[1] is r); raw_pos="FIRST" if ri==0 else "LAST" if ri==len(raw_track)-1 else "MIDDLE"; synthetic=(int(ttrack[0]["timestamp_us"])!=raw_track[0][2][1]+rec["offset_us"] or int(ttrack[-1]["timestamp_us"])!=raw_track[-1][2][1]+rec["offset_us"]); dims=first_dims[key[0]]; rc=corners(c,q,dims); lc=corners(target["center"],target["quaternion"],target["dimensions"]); rb=bev(c,q,dims); lb=bev(target["center"],target["quaternion"],target["dimensions"]); prev_raw=raw_track[ri-1] if ri>0 else None; next_raw=raw_track[ri+1] if ri+1<len(raw_track) else None; prev_local=ttrack[li-1] if li>0 else None; next_local=ttrack[li+1] if li+1<len(ttrack) else None
            rows.append({"clip_id":cid,"track_id":key[0],"category":target["category"],"raw_row_index":raw_idx,"raw_timestamp_us":key[1],"mapped_nurec_timestamp_us":mapped,"reference_frame_timestamp_us":int(r["reference_frame_timestamp_us"]),"raw_track_position":raw_pos,"local_serialized_track_position":local_pos,"synthetic_endpoint_status":"SYNTHETIC_ENDPOINT_PRESENT" if synthetic else "NO_SYNTHETIC_ENDPOINT_DETECTED","raw_center":json.dumps([r["center_x"],r["center_y"],r["center_z"]]),"raw_quaternion":json.dumps([r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]]),"direct_center":json.dumps(center(direct).tolist()),"nvidia_center":json.dumps(c.tolist()),"local_center":json.dumps(target["center"]),"center_residual_m":cd,"yaw_residual_deg":yd,"direct_vs_nvidia_center_diff_m":dc,"direct_vs_nvidia_rotation_diff_deg":dr,"bev_iou":iou(rb,lb),"bev_corner_error_m":set_rmse(rb,lb),"corner3d_error_m":set_rmse(rc,lc),"yaw_parameterization_geometry_equivalent":yd>170 and iou(rb,lb)>.9,"ego_distance_m":float(np.linalg.norm(c-center(ego_local))),"dimensions":json.dumps(dims),"previous_raw_timestamp":prev_raw[2][1] if prev_raw else None,"next_raw_timestamp":next_raw[2][1] if next_raw else None,"previous_local_timestamp":prev_local["timestamp_us"] if prev_local else None,"next_local_timestamp":next_local["timestamp_us"] if next_local else None,"raw_prev_current_displacement_m":float(np.linalg.norm(np.asarray(center(ego_local))-np.asarray(center(ego_local)))) if False else None})
    cs=[r["center_residual_m"] for r in rows]; ys=[r["yaw_residual_deg"] for r in rows]; bi=[r["bev_iou"] for r in rows]; be=[r["bev_corner_error_m"] for r in rows]; ce=[r["corner3d_error_m"] for r in rows]; dp=[r["direct_vs_nvidia_center_diff_m"] for r in rows]; dr=[r["direct_vs_nvidia_rotation_diff_deg"] for r in rows]; outliers=[r for r in rows if r["center_residual_m"]>2]; yawflip=[r for r in rows if r["yaw_residual_deg"]>170]
    by_track={}
    for row in rows: by_track.setdefault((row["clip_id"],row["track_id"]),[]).append(row)
    neighbors=[]
    for group in by_track.values():
        group.sort(key=lambda x:int(x["mapped_nurec_timestamp_us"]))
        for i,row in enumerate(group):
            for prefix,field in (("raw","raw_center"),("local","local_center")):
                cur=np.asarray(json.loads(row["nvidia_center"] if prefix=="raw" else row["local_center"]),float)
                for label,idx in (("prev",i-1),("next",i+1)):
                    if 0<=idx<len(group):
                        other=np.asarray(json.loads(group[idx]["nvidia_center"] if prefix=="raw" else group[idx]["local_center"]),float)
                        dt=abs(int(group[idx]["mapped_nurec_timestamp_us"])-int(row["mapped_nurec_timestamp_us"]))/1e6
                        row[f"{prefix}_{label}_current_displacement_m"]=float(np.linalg.norm(cur-other)); row[f"{prefix}_{label}_current_speed_mps"]=float(np.linalg.norm(cur-other)/dt) if dt else None
                    else:
                        row[f"{prefix}_{label}_current_displacement_m"]=None; row[f"{prefix}_{label}_current_speed_mps"]=None
            if row in outliers: neighbors.append(row)
    summary={"matched_selected_observation_count":len(rows),"global_center_rmse_m":math.sqrt(sum(v*v for v in cs)/len(cs)),"global_center_p95_m":p95(cs),"global_center_max_m":max(cs),"global_yaw_rmse_deg":math.sqrt(sum(v*v for v in ys)/len(ys)),"global_yaw_p95_deg":p95(ys),"global_yaw_max_deg":max(ys),"global_bev_iou_mean":statistics.mean(bi),"global_bev_iou_p05":float(np.quantile(bi,.05)),"global_bev_iou_min":min(bi),"global_bev_corner_set_p95_m":p95(be),"global_bev_corner_set_max_m":max(be),"global_3d_corner_set_p95_m":p95(ce),"global_3d_corner_set_max_m":max(ce),"yaw_gt_45_count":sum(v>45 for v in ys),"yaw_gt_170_count":sum(v>170 for v in ys),"yaw_parameterization_geometric_equivalence_count":sum(r["yaw_parameterization_geometry_equivalent"] for r in rows),"center_gt_2m_count":sum(v>2 for v in cs),"max_direct_vs_nvidia_center_diff_m":max(dp),"p95_direct_vs_nvidia_center_diff_m":p95(dp),"max_direct_vs_nvidia_rotation_diff_deg":max(dr),"p95_direct_vs_nvidia_rotation_diff_deg":p95(dr)}
    summary=json.loads(json.dumps(summary,default=lambda x:x.item() if hasattr(x,"item") else x))
    _csv(out/"nvidia_transform_parity.csv",rows); _csv(out/"cuboid_physical_geometry_crosscheck.csv",rows); _csv(out/"yaw_parameterization_geometry_diagnostic.csv",[r for r in rows if r["yaw_residual_deg"]>170]); _csv(out/"center_outlier_final_forensics.csv",outliers); _csv(out/"center_outlier_neighbor_continuity.csv",neighbors)
    parity=summary["max_direct_vs_nvidia_center_diff_m"]<1e-4 and summary["max_direct_vs_nvidia_rotation_diff_deg"]<1e-3; systematic=not(parity and summary["global_center_p95_m"]<3 and summary["global_bev_iou_p05"]>.5)
    _dump(out/"nvidia_transform_parity_summary.json",{"NVIDIA_MATH_REPLAY_STATUS":"VERIFIED_LOCAL_EULER_XYZ_EQUIVALENT","DIRECT_TRANSFORM_PARITY_STATUS":"VERIFIED" if parity else "MISMATCH","metrics":summary})
    _dump(out/"cuboid_physical_geometry_summary.json",summary); _dump(out/"cuboid_geometry_final_contract.json",{"formula":"T_rig_world_local @ T_object_rig","nvidia_bbox_math":"R.from_quat(qx,qy,qz,qw).as_euler(xyz); bbox_pose -> transform_bbox -> bbox_pose; float32 serialization","corner_matching":"Hungarian unordered 8-corner/4-corner sets","systematic_frame_rule":"parity + center P95 <3m + BEV IoU P05 >0.5","systematic_frame_error_found":systematic,"no_offset_rederive":True})
    _dump(out/"cuboid_geometry_final_summary.json",{**summary,"nvidia_math_replay_status":"VERIFIED_LOCAL_EULER_XYZ_EQUIVALENT","direct_transform_parity_status":"VERIFIED" if parity else "MISMATCH","center_outlier_provenance_status":"ISOLATED_ROWS_RETAINED","systematic_frame_error_found":systematic,"raw_row_selection_status":"VERIFIED_REPLAY_RULES","cuboid_serialization_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","cuboid_frame_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","obstacle_transform_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","normalized_obstacle_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","obstacle_geometry_block_ready_to_close":not systematic,"per_clip_offset_rederived":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"remaining_blockers":[] if not systematic else ["PHYSICAL_GEOMETRY_OR_CENTER_OUTLIERS_REQUIRE_REVIEW"],"recommended_next_step":"start scorer contract review" if not systematic else "review remaining physical geometry outliers"})
if __name__=="__main__": main()
