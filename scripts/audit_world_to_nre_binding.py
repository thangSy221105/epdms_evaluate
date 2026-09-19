"""Verify NuRec world-to-scene binding using ego trajectory before cuboids."""
from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path

try:
    from scripts.audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose, load_nurec_poses
    from scripts.audit_pai_obstacle_offline import PILOT_CLIPS
    from scripts.prepare_nurec_obstacles import load_time_record, pose_yaw, _wrap, _csv, _dump
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import _mm, _pose, _rquat, interpolate_pose, load_nurec_poses
    from audit_pai_obstacle_offline import PILOT_CLIPS
    from prepare_nurec_obstacles import load_time_record, pose_yaw, _wrap, _csv, _dump

NCORE_FRAME_CONVERSION = "NCore FrameConversion.matrix semantics: source -> target"

def read_member(root, rel, suffix, clip):
    import pyarrow.parquet as pq
    for z in sorted((Path(root)/rel).glob("*.zip")):
        with zipfile.ZipFile(z) as a:
            names=[n for n in a.namelist() if clip in n and n.endswith(suffix)]
            if names: return pq.read_table(io.BytesIO(a.read(names[0]))).to_pylist()
    raise FileNotFoundError(clip)

def pose_rows(rows):
    return [(int(r["timestamp"]), _pose([r["x"],r["y"],r["z"]],[r["qx"],r["qy"],r["qz"],r["qw"]])) for r in rows]

def inv(m):
    r=[x[:3] for x in m[:3]]; p=[m[i][3] for i in range(3)]; rt=[[r[j][i] for j in range(3)] for i in range(3)]
    return [rt[0]+[-sum(rt[0][j]*p[j] for j in range(3))],rt[1]+[-sum(rt[1][j]*p[j] for j in range(3))],rt[2]+[-sum(rt[2][j]*p[j] for j in range(3))],[0.,0.,0.,1.]]

def center(m): return [m[0][3],m[1][3],m[2][3]]
def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
def metrics(pairs):
    ds=[]; ys=[]
    for a,b in pairs:
        ds.append(math.dist(center(a),center(b))); ys.append(math.degrees(abs(_wrap(pose_yaw(a)-pose_yaw(b)))))
    return {"matched_timestamp_count":len(pairs),"center_rmse_m":math.sqrt(sum(x*x for x in ds)/len(ds)) if ds else None,"center_p95_m":p95(ds),"center_max_m":max(ds) if ds else None,"yaw_rmse_deg":math.sqrt(sum(x*x for x in ys)/len(ys)) if ys else None,"yaw_p95_deg":p95(ys),"yaw_max_deg":max(ys) if ys else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pai-root",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--time-alignment-jsonl",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["clip_id"] in PILOT_CLIPS]
    ego_rows=[]; obstacle_rows=[]; summaries=[]; ego_contract=[]; obstacle_contract=[]; offsets=[]
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(Path(a.time_alignment_jsonl),cid,int(item.get("physicalai_t0_us",5100000))); offsets.append(rec["offset_us"])
        raw=pose_rows(read_member(a.pai_root,"labels/egomotion.offline",".egomotion.offline.parquet",cid)); local=load_nurec_poses(clip); meta=json.loads((clip/"rig_trajectories.json").read_text(encoding="utf-8")); w=meta.get("world_to_scene",meta.get("world_to_nre")); matrix=w["matrix"] if isinstance(w,dict) else w
        invw=inv(matrix); pairs_a=[]; pairs_b=[]
        for ts,p in raw:
            target=interpolate_pose(local,ts+rec["offset_us"]); pairs_a.append((_mm(matrix,p),target)); pairs_b.append((_mm(invw,p),target))
        ma=metrics(pairs_a); mb=metrics(pairs_b); direction="world_to_scene" if ma["center_rmse_m"] < mb["center_rmse_m"] else "inverse(world_to_scene)"
        ego_contract.append({"clip_id":cid,"world_to_nre_found":True,"world_to_nre_matrix":matrix,"source":"rig_trajectories.json:world_to_nre.matrix","direction_candidate":"NCore world -> local NuRec scene","public_semantics":NCORE_FRAME_CONVERSION,"A_forward_metrics":ma,"B_inverse_metrics":mb,"selected_direction":direction})
        summaries.append({"clip_id":cid,"A_forward":ma,"B_inverse":mb})
        # Obstacle promotion is intentionally gated by ego binding.
        obstacle_contract.append({"clip_id":cid,"status":"BLOCKED_BY_EGO_BINDING" if direction not in {"world_to_scene"} else "CANDIDATE_ONLY","matched_observation_count":0,"formula":"world_to_nre @ T_rig_world_local @ T_object_rig"})
    forward=[x["A_forward_metrics"] for x in ego_contract]; inverse=[x["B_inverse_metrics"] for x in ego_contract]; verified=all(x["selected_direction"]=="world_to_scene" and x["A_forward_metrics"]["center_rmse_m"] is not None and x["A_forward_metrics"]["center_rmse_m"]<2.0 for x in ego_contract)
    _csv(out/"ego_world_to_nre_crosscheck.csv",[{"clip_id":x["clip_id"],"variant":v,**x[f"{v}_metrics"]} for x in ego_contract for v in ("A_forward","B_inverse")]); _csv(out/"obstacle_world_to_nre_crosscheck.csv",obstacle_contract)
    _dump(out/"world_to_nre_contract.json",{"status":"VERIFIED" if verified else "PARTIALLY_VERIFIED","found":True,"source":"rig_trajectories.json:world_to_nre.matrix","direction":"NCore world -> local NuRec scene","matrix_per_clip":[{"clip_id":x["clip_id"],"matrix":x["world_to_nre_matrix"]} for x in ego_contract],"public_semantics":NCORE_FRAME_CONVERSION,"no_fitted_transform":True,"offset_rederived":False})
    _dump(out/"ego_world_to_nre_summary.json",{"clips":summaries,"binding_status":"VERIFIED" if verified else "PARTIALLY_VERIFIED","max_forward_center_rmse_m":max(x["center_rmse_m"] for x in forward),"max_forward_center_p95_m":max(x["center_p95_m"] for x in forward),"max_forward_yaw_rmse_deg":max(x["yaw_rmse_deg"] for x in forward),"max_forward_yaw_p95_deg":max(x["yaw_p95_deg"] for x in forward)})
    _dump(out/"obstacle_world_to_nre_summary.json",{"status":"READY_TO_RUN" if verified else "BLOCKED_BY_EGO_BINDING","matched_observation_count":0,"reason":"obstacle stage is gated on verified ego world-to-scene binding"})
    _dump(out/"world_to_nre_binding_final_summary.json",{"world_to_nre_found":True,"world_to_nre_direction":"NCore world -> local NuRec scene","world_to_nre_binding_status":"VERIFIED" if verified else "PARTIALLY_VERIFIED","ego_matched_timestamp_count":sum(x["matched_timestamp_count"] for x in forward),"max_ego_center_rmse_m":max(x["center_rmse_m"] for x in forward),"max_ego_center_p95_m":max(x["center_p95_m"] for x in forward),"max_ego_yaw_rmse_deg":max(x["yaw_rmse_deg"] for x in forward),"max_ego_yaw_p95_deg":max(x["yaw_p95_deg"] for x in forward),"final_obstacle_transform_formula":"world_to_nre @ T_rig_world_local(t) @ T_object_rig","matched_observation_count":0,"obstacle_transform_status":"BLOCKED_BY_EGO_BINDING","normalized_obstacle_status":"BLOCKED_BY_EGO_BINDING","obstacle_geometry_block_ready_to_close":False,"per_clip_offset_rederived":False,"cf_proxy_label_set_ready":True,"ttc_proxy_label_set_ready":True,"cf_data_ready":False,"ttc_data_ready":False,"physical_world_obstacle_completeness":"NOT_CLAIMED","remaining_blockers":["EGO_WORLD_TO_NRE_BINDING_NOT_VERIFIED" if not verified else "OBSTACLE_STAGE_NOT_IMPLEMENTED"],"recommended_next_step":"do not promote obstacle transform until ego binding passes provenance and numerical acceptance"})

if __name__=="__main__": main()
