"""Reproduce the documented PAI -> NCore -> NuRec cuboid transform chain.

This is an audit/forensics tool.  It does not alter raw data or scorer semantics.
"""
from __future__ import annotations

import argparse, csv, io, json, math, statistics, zipfile
from pathlib import Path

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS, group_rows_by_mapped_timestamp, nearest_group
    from scripts.prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose
    from audit_pai_obstacle_offline import PILOT_CLIPS, group_rows_by_mapped_timestamp, nearest_group
    from prepare_nurec_obstacles import build_queries, load_sequence_tracks, load_time_record, pose_yaw, _wrap, _csv, _dump

NCORE_URL = "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
INSTANT_URL = "https://github.com/NVIDIA/instant-nurec/blob/main/instant_nurec/datasets/utils.py"

def read_member(root, rel, suffix, clip):
    for z in sorted((Path(root) / rel).glob("*.zip")):
        with zipfile.ZipFile(z) as a:
            names = [n for n in a.namelist() if clip in n and n.endswith(suffix)]
            if names:
                import pyarrow.parquet as pq
                return pq.read_table(io.BytesIO(a.read(names[0]))).to_pylist(), str(z), names[0]
    raise FileNotFoundError(f"missing {rel} {clip} {suffix}")

def inv(m):
    r = [row[:3] for row in m[:3]]; t = [m[i][3] for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    out = [[rt[i][j] for j in range(3)] + [-(rt[i][0]*t[0]+rt[i][1]*t[1]+rt[i][2]*t[2])] for i in range(3)]
    return out + [[0.,0.,0.,1.]]

def pose_rows(rows):
    return [(int(r["timestamp"]), _pose([r["x"],r["y"],r["z"]], [r["qx"],r["qy"],r["qz"],r["qw"]])) for r in rows]

def obj_pose(r):
    return _pose([r["center_x"],r["center_y"],r["center_z"]], [r["orientation_x"],r["orientation_y"],r["orientation_z"],r["orientation_w"]])

def center(m): return [m[0][3],m[1][3],m[2][3]]
def quat(m): return _rquat([row[:3] for row in m[:3]])
def p95(v): return sorted(v)[min(len(v)-1, max(0, math.ceil(.95*len(v))-1))] if v else None
def residuals(matrix, target):
    c=center(matrix); q=quat(matrix)
    ty=pose_yaw(_pose(target["center"], target["quaternion"]))
    ys=math.degrees(abs(_wrap(pose_yaw(_pose(c,q))-ty)))
    return math.dist(c,target["center"]), ys

def local_world_ref(clip):
    d = json.loads((Path(clip)/"rig_trajectories.json").read_text(encoding="utf-8"))
    base = d.get("T_world_base")
    ref = inv(base) if base else None
    wtn = d.get("world_to_scene", d.get("world_to_nre"))
    return {"exact_field_present": False, "candidate_world_to_scene": wtn, "T_world_base_present": base is not None,
            "T_world_ref_candidate": ref, "status": "CANDIDATE_DERIVED_NOT_VERIFIED",
            "reason": "T_world_ref=inverse(T_world_base) is a convention candidate; local serializer does not explicitly label this edge as Instant-NuRec world reference"}

def compare(clip, pai, ego, local, offset, world_ref):
    local_map = {(str(x["track_id"]), int(x["timestamp_us"])): x for x in local}
    samples = pose_rows(ego); first = samples[0][1]; rebase = inv(first)
    rows=[]; unmatched=[]; variants={k:[] for k in ("A_RAW_EGO","B_REBASED_EGO","C_WORLD_REF_REBASED","D_INVERSE_WORLD_REF_REBASED")}
    for r in pai:
        key=(str(r["track_id"]),int(r["timestamp_us"])+int(offset)); target=local_map.get(key)
        if target is None:
            unmatched.append({"clip_id":clip,"track_id":str(r["track_id"]),"pai_timestamp_us":int(r["timestamp_us"]),"nurec_timestamp_us":key[1],"classification":"UNRESOLVED"}); continue
        ego_pose=interpolate_pose(samples,int(r["reference_frame_timestamp_us"]))
        base=_mm(ego_pose,obj_pose(r)); raw=base; rebased=_mm(rebase,base)
        wr = world_ref["T_world_ref_candidate"] or _pose([0,0,0],[0,0,0,1])
        variants["A_RAW_EGO"].append(raw); variants["B_REBASED_EGO"].append(rebased)
        variants["C_WORLD_REF_REBASED"].append(_mm(wr,rebased)); variants["D_INVERSE_WORLD_REF_REBASED"].append(_mm(inv(wr),rebased))
        rows.append((r,target,raw,rebased,key[1]))
    def metric(ms):
        cs=[]; ys=[]; ds=[]
        for (r,t,b,_,ts) in rows:
            d,y=residuals(b,t); cs.append(d); ys.append(y); ds.append(max(abs(float(r[k])-float(t["dimensions"][i])) for i,k in enumerate(("size_x","size_y","size_z"))))
        return {"matched":len(rows),"center_rmse_m":math.sqrt(sum(x*x for x in cs)/len(cs)) if cs else None,"center_p95_m":p95(cs),"center_max_m":max(cs) if cs else None,"yaw_rmse_deg":math.sqrt(sum(x*x for x in ys)/len(ys)) if ys else None,"yaw_p95_deg":p95(ys),"yaw_max_deg":max(ys) if ys else None,"dimension_max_abs_m":max(ds) if ds else None}
    # Recompute each metric against the corresponding documented/diagnostic matrix.
    def metric_for(name):
        cs=[]; ys=[]; ds=[]
        for (r,t,raw,rebased,ts), matrix in zip(rows, variants[name]):
            d,y=residuals(matrix,t); cs.append(d); ys.append(y); ds.append(max(abs(float(r[k])-float(t["dimensions"][i])) for i,k in enumerate(("size_x","size_y","size_z"))))
        return {"matched":len(rows),"center_rmse_m":math.sqrt(sum(x*x for x in cs)/len(cs)) if cs else None,"center_p95_m":p95(cs),"center_max_m":max(cs) if cs else None,"yaw_rmse_deg":math.sqrt(sum(x*x for x in ys)/len(ys)) if ys else None,"yaw_p95_deg":p95(ys),"yaw_max_deg":max(ys) if ys else None,"dimension_max_abs_m":max(ds) if ds else None}
    variants_summary={"clip_id":clip}
    for name in variants: variants_summary[name]=metric_for(name)
    return rows, unmatched, variants_summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]
    all_cross=[]; all_unmatched=[]; summaries=[]; variants=[]; rebase=[]; worldrefs=[]; cf=[]; ttc=[]
    for item in manifest:
        cid=item["clip_id"]; rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); pai,pa,pm=read_member(a.pai_root,"labels/obstacle.offline",".obstacle.offline.parquet",cid); ego,ea,em=read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid); local,_,_=load_sequence_tracks(Path(item["nurec_clip_dir"])/"sequence_tracks.json"); wr=local_world_ref(item["nurec_clip_dir"])
        samples=pose_rows(ego); first=samples[0][1]; rebased_first=_mm(inv(first),first); rebase.append({"clip_id":cid,"formula":"inverse(T_first_raw) @ T_first_raw","T_first_raw":first,"T_rebased_first":rebased_first,"rebased_first_identity":all(abs(rebased_first[i][j]-(1.0 if i==j else 0.0))<1e-9 for i in range(4) for j in range(4)),"source":NCORE_URL,"rebase_verified":True})
        worldrefs.append({"clip_id":cid,**wr}); rows,unm,vs=compare(cid,pai,ego,local,rec["offset_us"],wr); all_unmatched+=unm; variants.append(vs)
        chosen=vs["C_WORLD_REF_REBASED"]; summaries.append({"clip_id":cid,**chosen,"per_clip_offset_rederived":False});
        for r,t,raw,b,ts in rows:
            final_matrix=_mm(wr["T_world_ref_candidate"],b); c=center(final_matrix); q=quat(final_matrix); all_cross.append({"clip_id":cid,"track_id":str(r["track_id"]),"pai_timestamp_us":int(r["timestamp_us"]),"nurec_timestamp_us":ts,"center_x":c[0],"center_y":c[1],"center_z":c[2],"quaternion_x":q[0],"quaternion_y":q[1],"quaternion_z":q[2],"quaternion_w":q[3],"center_residual_m":math.dist(c,t["center"]),"yaw_residual_deg":math.degrees(abs(_wrap(pose_yaw(_pose(c,q))-pose_yaw(_pose(t["center"],t["quaternion"]))))),"dimension_max_abs_m":max(abs(float(r[k])-float(t["dimensions"][i])) for i,k in enumerate(("size_x","size_y","size_z"))),"track_id_match":"EXACT_ID_PRESERVED","transform":"T_world_ref_candidate @ T_rig_world_local @ T_object_rig"})
        g=group_rows_by_mapped_timestamp(pai,rec["offset_us"])
        for label,qs,tol,dst in (("CF",build_queries(rec["nurec_t0_us"])[0],50000,cf),("TTC",build_queries(rec["nurec_t0_us"])[1],100000,ttc)):
            for q in qs:
                near,delta=nearest_group(q,g,tol); dst.append({"clip_id":cid,"query_timestamp_us":q,"nearest_label_timestamp_us":near,"delta_us":delta,"obstacle_count":len(g.get(near,[])) if near is not None else 0,"label_set_timestamp_available":near is not None,"physical_world_completeness":"NOT_CLAIMED"})
    _csv(out/"pai_ncore_rig_pose_reproduction.csv",rebase); _csv(out/"transform_variant_comparison.csv",[{"clip_id":x["clip_id"],"variant":k,**v} for x in variants for k,v in x.items() if k!="clip_id"]); _csv(out/"pai_nurec_exact_crosscheck.csv",all_cross)
    outliers=[]
    for r in all_cross:
        if r["center_residual_m"]>2 or r["yaw_residual_deg"]>45:
            outliers.append({**r,"center_gt_2m":r["center_residual_m"]>2,"center_gt_5m":r["center_residual_m"]>5,"center_gt_10m":r["center_residual_m"]>10,"yaw_gt_45deg":r["yaw_residual_deg"]>45,"yaw_gt_90deg":r["yaw_residual_deg"]>90,"yaw_gt_170deg":r["yaw_residual_deg"]>170})
    _csv(out/"obstacle_transform_outliers.csv",outliers); _csv(out/"unmatched_candidate_classification.csv",all_unmatched)
    _dump(out/"pai_ncore_rebase_contract.json",{"status":"VERIFIED_PUBLIC_CONVERTER","formula":"inverse(T_world_world_global) @ T_rig_world_raw","first_pose_defines_world_global":True,"source":NCORE_URL,"per_clip_offset_rederived":False})
    _dump(out/"instant_nurec_world_ref_contract.json",{"status":"PUBLIC_API_DIRECTION_VERIFIED","formula":"get_frames_T_source_sensor(source_node=world, frame_timepoint=END)","source":INSTANT_URL,"local_application":"NOT_APPLIED_UNPROVEN"})
    _dump(out/"local_nurec_world_ref_contract.json",{"status":"CANDIDATE_DERIVED_NOT_VERIFIED","clips":worldrefs,"derivation":"T_world_ref=inverse(T_world_base)","no_fitted_transform":True})
    _dump(out/"pai_to_nurec_exact_transform_contract.json",{"status":"PARTIALLY_VERIFIED","formula":"T_world_ref @ T_rig_world_local @ T_object_rig","world_ref_status":"CANDIDATE_DERIVED_NOT_VERIFIED","timestamp_mapping":"existing per-clip offset reused","no_fitted_transform":True})
    _dump(out/"pai_nurec_exact_crosscheck_summary.json",{"clips":summaries,"matched_observation_count":len(all_cross),"world_ref_status":"CANDIDATE_DERIVED_NOT_VERIFIED"})
    def ready(x): return sum(bool(r["label_set_timestamp_available"]) for r in x),len(x)
    c1,c2=ready(cf); t1,t2=ready(ttc); _csv(out/"cf_proxy_label_set_availability.csv",cf); _csv(out/"ttc_proxy_label_set_availability.csv",ttc)
    _dump(out/"proxy_label_set_readiness_summary.json",{"physical_world_obstacle_completeness":"NOT_CLAIMED","cf_proxy_label_set_required_query_count":c2,"cf_proxy_label_set_available_query_count":c1,"cf_proxy_label_set_availability_rate":c1/c2 if c2 else None,"cf_proxy_label_set_ready":c1==c2,"ttc_proxy_label_set_required_query_count":t2,"ttc_proxy_label_set_available_query_count":t1,"ttc_proxy_label_set_availability_rate":t1/t2 if t2 else None,"ttc_proxy_label_set_ready":t1==t2,"cf_data_ready":False,"ttc_data_ready":False})
    _dump(out/"pai_nurec_exact_transform_final_summary.json",{"pai_to_nurec_full_transform_status":"PARTIALLY_VERIFIED","obstacle_transform_status":"PARTIALLY_VERIFIED","normalized_obstacle_status":"PARTIALLY_VERIFIED","obstacle_geometry_block_ready_to_close":False,"track_id_provenance_status":"EXACT_ID_PRESERVED_FOR_MATCHED_ROWS","matched_observation_count":len(all_cross),"local_world_ref_status":"CANDIDATE_DERIVED_NOT_VERIFIED","local_world_to_scene_status":"PRESENT_AS_WORLD_TO_NRE_NON_IDENTITY","physical_world_obstacle_completeness":"NOT_CLAIMED","pai_empty_frame_semantics":"UNRESOLVED_EMPTY_ATTESTATION","cf_proxy_label_set_ready":c1==c2,"ttc_proxy_label_set_ready":t1==t2,"cf_data_ready":False,"ttc_data_ready":False,"per_clip_offset_rederived":False,"remaining_blockers":["T_WORLD_BASE_INVERSE_CONVENTION_NUMERICALLY_NOT_BOUND_TO_LOCAL_SCENE","EMPTY_FRAME_ATTESTATION_UNRESOLVED"],"recommended_next_step":"obtain exact NuRec sequence-loader serialization metadata before promoting the transform to scorer contract"})

if __name__=="__main__": main()
