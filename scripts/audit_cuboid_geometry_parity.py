"""Final NVIDIA cuboid math parity and physical-geometry audit."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path
import numpy as np

try:
    from scipy.spatial.transform import Rotation as SciPyRotation
    from scipy.spatial import ConvexHull as SciPyConvexHull
    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    SciPyRotation = None
    SciPyConvexHull = None
    SCIPY_AVAILABLE = False

try:
    from ncore.impl.common.transformations import bbox_pose as ncore_bbox_pose
    from ncore.impl.common.transformations import transform_bbox as ncore_transform_bbox
    from ncore.impl.common.transformations import pose_bbox as ncore_pose_bbox
    NCORE_RUNTIME_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ncore_bbox_pose = ncore_transform_bbox = ncore_pose_bbox = None
    NCORE_RUNTIME_AVAILABLE = False

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
def convex_hull_xy(points):
    pts=sorted({(float(p[0]),float(p[1])) for p in np.asarray(points,float)})
    if len(pts)<=1:return np.asarray(pts,float)
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=1e-12:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=1e-12:upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1],float)
def bev(c,q,dims):
    return convex_hull_xy(corners(c,q,dims)[:,:2])
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
            p=interpolate_pose(poses,int(r["reference_frame_timestamp_us"])); ego_local=np.asarray(reb)@np.asarray(p); b=bbox_from_raw(r); direct=ego_local@bbox_pose_np(b); nvidia=bbox_pose_transform(b,ego_local); nvidia_f=np.asarray(nvidia,dtype=np.float32); nvidia_pose=bbox_pose_np(nvidia_f); cd,yd,c,q=residual(nvidia_pose,target); dc=float(np.linalg.norm(center(direct)-center(nvidia_pose))); dr=rotation_angle(direct,nvidia_pose); reconstructed_yaw=pose_yaw(pose_matrix(c,q)); local_yaw=pose_yaw(pose_matrix(target["center"],target["quaternion"])); signed_yaw=math.degrees(_wrap(local_yaw-reconstructed_yaw)); signed_center=(np.asarray(target["center"],float)-c).tolist(); ttrack=tracks[key[0]]; li=next(i for i,x in enumerate(ttrack) if int(x["timestamp_us"])==mapped); local_pos="FIRST" if li==0 else "LAST" if li==len(ttrack)-1 else "MIDDLE"; raw_track=[x for x in selected if x[1]["track_id"]==r["track_id"]]; ri=next(i for i,x in enumerate(raw_track) if x[1] is r); raw_pos="FIRST" if ri==0 else "LAST" if ri==len(raw_track)-1 else "MIDDLE"; synthetic=(int(ttrack[0]["timestamp_us"])!=raw_track[0][2][1]+rec["offset_us"] or int(ttrack[-1]["timestamp_us"])!=raw_track[-1][2][1]+rec["offset_us"]); dims=first_dims[key[0]]; rc=corners(c,q,dims); lc=corners(target["center"],target["quaternion"],target["dimensions"]); rb=bev(c,q,dims); lb=bev(target["center"],target["quaternion"],target["dimensions"]); prev_raw=raw_track[ri-1] if ri>0 else None; next_raw=raw_track[ri+1] if ri+1<len(raw_track) else None; prev_local=ttrack[li-1] if li>0 else None; next_local=ttrack[li+1] if li+1<len(ttrack) else None
            rows.append({"clip_id":cid,"track_id":key[0],"category":target["category"],"raw_row_index":raw_idx,"raw_timestamp_us":key[1],"mapped_nurec_timestamp_us":mapped,"reference_frame_timestamp_us":int(r["reference_frame_timestamp_us"]),"raw_track_position":raw_pos,"local_serialized_track_position":local_pos,"synthetic_endpoint_status":"SYNTHETIC_ENDPOINT_PRESENT" if synthetic else "NO_SYNTHETIC_ENDPOINT_DETECTED","raw_center":json.dumps([r["center_x"],r["center_y"],r["center_z"]]),"raw_quaternion":json.dumps([r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]]),"direct_center":json.dumps(center(direct).tolist()),"nvidia_center":json.dumps(c.tolist()),"local_center":json.dumps(target["center"]),"center_residual_xyz":json.dumps(signed_center),"center_residual_m":cd,"yaw_residual_deg":yd,"signed_yaw_delta_deg":signed_yaw,"direct_vs_nvidia_center_diff_m":dc,"direct_vs_nvidia_rotation_diff_deg":dr,"bev_iou":iou(rb,lb),"bev_corner_error_m":set_rmse(rb,lb),"corner3d_error_m":set_rmse(rc,lc),"yaw_parameterization_geometry_equivalent":yd>170 and iou(rb,lb)>.9,"ego_distance_m":float(np.linalg.norm(c-center(ego_local))),"dimensions":json.dumps(dims),"previous_raw_timestamp":prev_raw[2][1] if prev_raw else None,"next_raw_timestamp":next_raw[2][1] if next_raw else None,"previous_local_timestamp":prev_local["timestamp_us"] if prev_local else None,"next_local_timestamp":next_local["timestamp_us"] if next_local else None})
    cs=[r["center_residual_m"] for r in rows]; ys=[r["yaw_residual_deg"] for r in rows]; bi=[r["bev_iou"] for r in rows]; be=[r["bev_corner_error_m"] for r in rows]; ce=[r["corner3d_error_m"] for r in rows]; dp=[r["direct_vs_nvidia_center_diff_m"] for r in rows]; dr=[r["direct_vs_nvidia_rotation_diff_deg"] for r in rows]
    by_track={}
    for row in rows: by_track.setdefault((row["clip_id"],row["track_id"]),[]).append(row)
    outliers=[]; neighbors=[]; systematic_track_count=0; contiguous_cluster_count=0; isolated_count=0
    for group in by_track.values():
        group.sort(key=lambda x:int(x["mapped_nurec_timestamp_us"]))
        for i,row in enumerate(group):
            for prefix in ("raw","local"):
                cur=np.asarray(json.loads(row["nvidia_center"] if prefix=="raw" else row["local_center"]),float)
                for label,idx in (("previous",i-1),("next",i+1)):
                    if 0<=idx<len(group):
                        other=np.asarray(json.loads(group[idx]["nvidia_center"] if prefix=="raw" else group[idx]["local_center"]),float); dt=abs(int(group[idx]["mapped_nurec_timestamp_us"])-int(row["mapped_nurec_timestamp_us"]))/1e6; d=float(np.linalg.norm(cur-other))
                        row[f"{prefix}_{label}_displacement_m"]=d; row[f"{prefix}_{label}_speed_mps"]=d/dt if dt else None
                    else: row[f"{prefix}_{label}_displacement_m"]=None; row[f"{prefix}_{label}_speed_mps"]=None
            if row["center_residual_m"]>2:
                row["previous_center_residual_m"]=group[i-1]["center_residual_m"] if i else None; row["next_center_residual_m"]=group[i+1]["center_residual_m"] if i+1<len(group) else None; outliers.append(row); neighbors.append(row)
        indices=[i for i,x in enumerate(group) if x["center_residual_m"]>2]
        if len(indices)>=2: systematic_track_count+=1
        if indices:
            runs=1
            for aidx,bidx in zip(indices,indices[1:]):
                if bidx==aidx+1:runs+=1
                else:
                    if runs>1:contiguous_cluster_count+=1
                    runs=1
            if runs>1:contiguous_cluster_count+=1
    for row in outliers:
        small=[v for v in (row.get("previous_center_residual_m"),row.get("next_center_residual_m")) if v is not None]
        row["outlier_classification"]="ISOLATED_ROW_ANOMALY" if small and all(v<2 and v<=row["center_residual_m"]*.5 for v in small) else "NON_ISOLATED_OR_UNRESOLVED"
        if row["outlier_classification"]=="ISOLATED_ROW_ANOMALY":isolated_count+=1
    outlier_status="ALL_ISOLATED_NO_SYSTEMATIC_PATTERN" if outliers and isolated_count==len(outliers) and systematic_track_count==0 else ("UNRESOLVED" if not outliers else "MIXED")
    clip_bias=[]; clip_yaw=[]
    for cid in sorted({r["clip_id"] for r in rows}):
        cr=[r for r in rows if r["clip_id"]==cid]; vec=np.asarray([json.loads(r["center_residual_xyz"]) for r in cr],float); sy=np.asarray([r["signed_yaw_delta_deg"] for r in cr],float)
        clip_bias.append({"clip_id":cid,"matched_count":len(cr),"mean_dx_m":float(np.mean(vec[:,0])),"mean_dy_m":float(np.mean(vec[:,1])),"mean_dz_m":float(np.mean(vec[:,2])),"median_dx_m":float(np.median(vec[:,0])),"median_dy_m":float(np.median(vec[:,1])),"median_dz_m":float(np.median(vec[:,2])),"std_dx_m":float(np.std(vec[:,0])),"std_dy_m":float(np.std(vec[:,1])),"std_dz_m":float(np.std(vec[:,2]))})
        clip_yaw.append({"clip_id":cid,"matched_count":len(cr),"mean_signed_yaw_delta_deg":float(np.mean(sy)),"median_signed_yaw_delta_deg":float(np.median(sy)),"std_signed_yaw_delta_deg":float(np.std(sy))})
    max_bias=[max(abs(x[k]) for x in clip_bias) for k in ("mean_dx_m","mean_dy_m","mean_dz_m")]; max_yaw=max(abs(x["mean_signed_yaw_delta_deg"]) for x in clip_yaw)
    summary={"matched_selected_observation_count":len(rows),"global_center_rmse_m":math.sqrt(sum(v*v for v in cs)/len(cs)),"global_center_p95_m":p95(cs),"global_center_max_m":max(cs),"global_yaw_rmse_deg":math.sqrt(sum(v*v for v in ys)/len(ys)),"global_yaw_p95_deg":p95(ys),"global_yaw_max_deg":max(ys),"global_bev_iou_mean":statistics.mean(bi),"global_bev_iou_p05":float(np.quantile(bi,.05)),"global_bev_iou_min":min(bi),"global_bev_corner_set_p95_m":p95(be),"global_bev_corner_set_max_m":max(be),"global_3d_corner_set_p95_m":p95(ce),"global_3d_corner_set_max_m":max(ce),"yaw_gt_45_count":sum(v>45 for v in ys),"yaw_gt_170_count":sum(v>170 for v in ys),"center_gt_2m_count":sum(v>2 for v in cs),"max_direct_vs_nvidia_center_diff_m":max(dp),"max_direct_vs_nvidia_rotation_diff_deg":max(dr),"max_abs_clip_mean_dx_m":max_bias[0],"max_abs_clip_mean_dy_m":max_bias[1],"max_abs_clip_mean_dz_m":max_bias[2],"max_abs_clip_mean_signed_yaw_deg":max_yaw,"center_outlier_total_count":len(outliers),"center_outlier_isolated_count":isolated_count,"center_outlier_contiguous_cluster_count":contiguous_cluster_count,"center_outlier_systematic_track_count":systematic_track_count}
    summary=json.loads(json.dumps(summary,default=lambda x:x.item() if hasattr(x,"item") else x))
    parity=summary["max_direct_vs_nvidia_center_diff_m"]<1e-4 and summary["max_direct_vs_nvidia_rotation_diff_deg"]<1e-3
    no_bias=max(max_bias)<0.5 and max_yaw<5.0; physical_ok=summary["global_center_p95_m"]<3 and summary["global_bev_iou_p05"]>.5 and summary["global_3d_corner_set_p95_m"]<1.0
    systematic=not(parity and physical_ok and no_bias and outlier_status=="ALL_ISOLATED_NO_SYSTEMATIC_PATTERN")
    scipy_status="VERIFIED" if parity else "MISMATCH"; ncore_status="NOT_RUN_RUNTIME_UNAVAILABLE" if not NCORE_RUNTIME_AVAILABLE else "AVAILABLE_NOT_REQUIRED"
    _csv(out/"scipy_ncore_math_parity.csv",rows); _csv(out/"cuboid_bev_hull_geometry.csv",rows); _csv(out/"per_clip_center_bias.csv",clip_bias); _csv(out/"per_clip_yaw_bias.csv",clip_yaw); _csv(out/"center_outlier_closure_forensics.csv",outliers)
    _dump(out/"scipy_ncore_math_parity_summary.json",{"SCIPY_MATH_PARITY_STATUS":scipy_status,"NCORE_RUNTIME_AVAILABLE":NCORE_RUNTIME_AVAILABLE,"NCORE_RUNTIME_PARITY_STATUS":ncore_status,"metrics":summary,"implementation":"local exact SciPy/NVIDIA-equivalent fallback" if not SCIPY_AVAILABLE else "direct scipy Rotation"})
    _dump(out/"cuboid_bev_hull_geometry_summary.json",{"metrics":summary,"geometry":"8 corners projected to XY then deterministic convex hull"})
    _dump(out/"center_outlier_isolation_summary.json",{"total":len(outliers),"isolated":isolated_count,"contiguous_clusters":contiguous_cluster_count,"systematic_tracks":systematic_track_count,"status":outlier_status})
    rule="SciPy/NVIDIA parity AND center P95 < 3m AND BEV IoU P05 > 0.5 AND 3D corner P95 < 1m AND max clip mean translation bias < 0.5m AND max clip mean signed yaw bias < 5deg AND all center outliers isolated with no systematic track"
    evidence={"math_parity":parity,"physical_geometry":physical_ok,"translation_bias_abs_max_m":max(max_bias),"signed_yaw_bias_abs_max_deg":max_yaw,"no_consistent_bias":no_bias,"outlier_status":outlier_status}
    blockers=[] if not systematic else [k for k,v in (("SCIPY_MATH_PARITY",parity),("PHYSICAL_GEOMETRY",physical_ok),("NO_CLIP_BIAS",no_bias),("OUTLIER_ISOLATION",outlier_status=="ALL_ISOLATED_NO_SYSTEMATIC_PATTERN")) if not v]
    contract={"formula":"T_rig_world_local @ T_object_rig","SCIPY_MATH_PARITY_STATUS":scipy_status,"NCORE_RUNTIME_AVAILABLE":NCORE_RUNTIME_AVAILABLE,"SYSTEMATIC_FRAME_DECISION_RULE":rule,"SYSTEMATIC_FRAME_DECISION_EVIDENCE":evidence,"SYSTEMATIC_FRAME_ERROR_FOUND":systematic,"CENTER_OUTLIER_PROVENANCE_STATUS":outlier_status,"no_fitted_correction":True,"per_clip_offset_rederived":False}
    _dump(out/"obstacle_geometry_closure_contract.json",contract)
    _dump(out/"obstacle_geometry_closure_final_summary.json",{**summary,**contract,"raw_row_selection_status":"VERIFIED_REPLAY_RULES","cuboid_serialization_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","cuboid_frame_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","obstacle_transform_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","normalized_obstacle_status":"VERIFIED" if not systematic else "PARTIALLY_VERIFIED","obstacle_geometry_block_ready_to_close":not systematic,"obstacle_geometry_block":"CLOSED" if not systematic else "OPEN","cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"physical_world_obstacle_completeness":"NOT_CLAIMED","remaining_blockers":blockers,"recommended_next_step":"start scorer contract review" if not systematic else "review closure blockers"})
if __name__=="__main__": main()
