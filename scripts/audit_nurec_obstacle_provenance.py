"""Forensic audit of NuRec obstacle frame and completeness provenance."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path
try:
    from scripts.prepare_nurec_obstacles import load_sequence_tracks, load_obstacle_rows, build_queries, load_time_record, _dump, _csv, pose_yaw, _pose
except ModuleNotFoundError:
    from prepare_nurec_obstacles import load_sequence_tracks, load_obstacle_rows, build_queries, load_time_record, _dump, _csv, pose_yaw, _pose

PUBLIC_NCORE="https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py"
PUBLIC_INSTANT="https://github.com/NVIDIA/instant-nurec/blob/main/instant_nurec/datasets/utils.py"
PUBLIC_API="https://nvidia.github.io/nurec/ncore/reference/apis/data.v3.html"

def _local_meta(clip: Path):
    out={}
    for name in ("metadata.yaml","parsed_config.yaml","data_info.json","datasource_summary.json"):
        p=clip/name; out[name]={"exists":p.is_file()}
        if p.is_file():
            try:
                if p.suffix==".yaml":
                    import yaml; value=yaml.safe_load(p.read_text(encoding="utf-8"))
                else: value=json.loads(p.read_text(encoding="utf-8"))
                out[name]["top_level_keys"]=list(value) if isinstance(value,dict) else None
                if name=="metadata.yaml": out[name]["version_string"]=value.get("version_string"); out[name]["scene_id"]=value.get("scene_id")
                if name=="parsed_config.yaml": out[name]["sequence_tracks_config"]=value.get("checkpoint",{}).get("artifact",{}).get("sequence_tracks")
                if name=="data_info.json": out[name]["pose_range"]=value.get("pose-range"); out[name]["sensor_frame_metadata_present"]=bool(value.get("shards"))
            except Exception as e: out[name]["read_error"]=str(e)
    return out

def _public_trace():
    return {"public_ncore_cuboid_frame_chain":{"repository":"NVIDIA/ncore","revision":"main (retrieved 2026-09-19)","file":"tools/data_converter/pai/converter.py","function":"PaiConverter._convert_clip / _load_cuboid_track_observations","code_path":["obstacle row -> CuboidTrackObservation(track_id,class_id,timestamp_us,reference_frame_id,reference_frame_timestamp_us,bbox3) at lines 2791-2845","poses_writer stores source_frame_id=rig,target_frame_id=world at lines 2666-2676"]},"public_instant_nurec_track_transform_chain":{"repository":"NVIDIA/instant-nurec","revision":"main (retrieved 2026-09-19)","file":"instant_nurec/datasets/utils.py","function":"consolidate_cuboid_tracks","code_path":["get_T_reference_world calls sequence_loader.pose_graph.evaluate_poses(reference_frame_id, world, reference_frame_timestamp_us) at lines 699-705","returns T_world_world_base @ T_reference_world at lines 707","bbox pose is bbox_pose(transform_bbox(observation.bbox3, T_reference_world)) at lines 744-748 and 812-818","function documentation says consolidated poses are relative to world frame at lines 651-674"]},"public_contract":{"url":PUBLIC_API,"fields":["reference_frame_id","reference_frame_timestamp_us","bbox3"]}}

def _crosscheck(seq, obs):
    om={(r["track_id"],r["timestamp_us"]):r for r in obs}; pairs=[]
    for s in seq:
        o=om.get((s["track_id"],s["timestamp_us"]))
        if not o: continue
        center=math.dist(s["center"],o["center"]); yaw=abs((pose_yaw(_pose(s["center"],s["quaternion"]))-pose_yaw(_pose(o["center"],o["quaternion"]))))
        dim=max(abs(a-b) for a,b in zip(s["dimensions"],o["dimensions"])); pairs.append({"track_id":s["track_id"],"timestamp_us":s["timestamp_us"],"center_residual_m":center,"yaw_residual_rad":yaw,"dimension_abs_diff":dim,"track_identity_status":"PROVENANCE_LINKED_TRACK_ID_CANDIDATE"})
    vals=lambda key:[r[key] for r in pairs]
    def p95(v): return sorted(v)[min(len(v)-1,max(0,math.ceil(.95*len(v))-1))] if v else None
    return pairs,{"pair_count":len(pairs),"center_rmse_m":math.sqrt(sum(v*v for v in vals("center_residual_m"))/len(pairs)) if pairs else None,"center_median_m":statistics.median(vals("center_residual_m")) if pairs else None,"center_p95_m":p95(vals("center_residual_m")),"center_max_m":max(vals("center_residual_m")) if pairs else None,"yaw_rmse_deg":math.degrees(math.sqrt(sum(v*v for v in vals("yaw_residual_rad"))/len(pairs))) if pairs else None,"yaw_p95_deg":math.degrees(p95(vals("yaw_residual_rad"))) if pairs else None,"dimension_max_abs":max(vals("dimension_abs_diff")) if pairs else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--time-alignment-jsonl",required=True); p.add_argument("--output-root",required=True); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); time_path=Path(a.time_alignment_jsonl); manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]; summaries=[]; cross_rows=[]
    trace=_public_trace(); _dump(root/"public_obstacle_provenance_trace.json",trace)
    for item in manifest:
        cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); seq,seq_errors,schema=load_sequence_tracks(clip/"sequence_tracks.json"); obs,obs_errors=load_obstacle_rows(clip/"clipgt"/"obstacle.parquet"); rec=load_time_record(time_path,cid,int(item.get("physicalai_t0_us",5_100_000))); pairs,metrics=_crosscheck(seq,obs); out=root/cid; out.mkdir(parents=True,exist_ok=True)
        local=_local_meta(clip); _dump(out/"local_sequence_tracks_lineage.json",{"schema":schema,"serialization_status":"SERIALIZATION_LINEAGE_PARTIAL","local_metadata":local,"field_mapping":{"tracks_id":"tracks_data.tracks_id","tracks_poses":"tracks_data.tracks_poses","tracks_timestamps_us":"tracks_data.tracks_timestamps_us","tracks_label_class":"tracks_data.tracks_label_class","cuboids_dims":"cuboidtracks_data.cuboids_dims"}})
        frame="STRONGLY_SUPPORTED_NOT_VERIFIED"; same="STRONGLY_SUPPORTED_NOT_VERIFIED"; transform="PARTIALLY_VERIFIED"; normalized="PARTIALLY_VERIFIED"
        _dump(out/"sequence_tracks_frame_contract.json",{"sequence_tracks_pose_frame":frame,"sequence_tracks_and_rig_same_frame":same,"object_pose_frame_static_scene":"STRONGLY_SUPPORTED_NOT_VERIFIED","provenance_confidence":"PUBLIC_CHAIN_SUPPORT_LOCAL_SERIALIZATION_NOT_EXACTLY_PINNED","transform_allowed":False})
        _dump(out/"obstacle_provenance_graph.json",{"status":"PARTIALLY_VERIFIED","nodes":["PAI obstacle row","NCore CuboidTrackObservation","pose_graph reference frame","NuRec world/base","sequence_tracks.json"],"edges":["PAI row fields are packaged by public NCore converter","Instant-NuRec evaluates reference frame to world and applies T_world_world_base","local serialization lineage remains partial"]})
        _dump(out/"pai_obstacle_schema.json",{"status":"NOT_AVAILABLE_IN_LOCAL_HF_CACHE","source_search":"exact official obstacle parquet not present in cached 5-clip chunks","expected_fields":["timestamp","center","size","orientation","track_id","class","reference_frame_id","reference_frame_timestamp_us"]})
        _dump(out/"pai_to_ncore_contract.json",{"status":"PUBLIC_CONTRACT_VERIFIED","source":PUBLIC_NCORE,"orientation_order":"qx,qy,qz,qw","dimensions":"passed to BBox3 unchanged","reference_frame":"copied into CuboidTrackObservation","timestamp":"copied as timestamp_us"})
        _dump(out/"ncore_to_nurec_track_contract.json",{"status":"PUBLIC_TRANSFORM_CHAIN_VERIFIED_LOCAL_SERIALIZATION_PARTIAL","source":PUBLIC_INSTANT,"output_frame":"world after T_world_world_base","track_fields":["dimension","label_class","poses","timestamps_us"]})
        _dump(out/"obstacle_dimension_semantics.json",{"status":"LENGTH_WIDTH_HEIGHT_PUBLIC_DOC_ONLY","local_output_names":"extent_x,extent_y,extent_z","source":PUBLIC_INSTANT,"note":"local NuRec extraction is not pinned to exact public revision"}); _dump(out/"obstacle_orientation_semantics.json",{"quaternion_order":"qx,qy,qz,qw","verified":True,"source":PUBLIC_NCORE})
        _dump(out/"observation_completeness_sources.json",{"camera_lidar_frame_ranges_present":True,"obstacle_annotation_complete_claim_found":False,"explicit_empty_annotations_found":False,"pose_timestamps_accepted_as_attestation":False,"status":"UNRESOLVED"}); _dump(out/"empty_scene_attestation_audit.json",{"status":"UNRESOLVED_EMPTY_ATTESTATION","reason":"object-only rows and sensor/pose ranges do not prove complete cuboid annotation coverage"})
        cf,ttc=build_queries(rec["nurec_t0_us"]); obs_ts=sorted({r["timestamp_us"] for r in seq}); temporal_cf=sum(min((abs(t-q) for t in obs_ts),default=10**18)<=50_000 for q in cf)/len(cf); temporal_ttc=sum(min((abs(t-q) for t in obs_ts),default=10**18)<=100_000 for q in ttc)/len(ttc)
        _csv(out/"cf_observation_completeness.csv",[{"timestamp_us":q,"temporal_object_evidence":min((abs(t-q) for t in obs_ts),default=10**18)<=50_000,"frame_completeness_status":"UNRESOLVED"} for q in cf]); _csv(out/"ttc_observation_completeness.csv",[{"timestamp_us":q,"temporal_object_evidence":min((abs(t-q) for t in obs_ts),default=10**18)<=100_000,"frame_completeness_status":"UNRESOLVED"} for q in ttc]); _csv(out/"obstacle_crosscheck.csv",pairs)
        summary={"clip_id":cid,"sequence_tracks_pose_frame":frame,"sequence_tracks_and_rig_same_frame":same,"sequence_tracks_pose_frame_verified":False,"object_pose_frame_static_scene":"STRONGLY_SUPPORTED_NOT_VERIFIED","serialization_lineage":"SERIALIZATION_LINEAGE_PARTIAL","obstacle_transform_status":transform,"normalized_obstacle_status":normalized,"per_clip_offset_rederived":False,"temporal_object_evidence_status":"PARTIAL_OR_COMPLETE_OBJECT_ROWS_ONLY","empty_scene_attestation_status":"UNRESOLVED_EMPTY_ATTESTATION","complete_observation_timeline_status":"UNRESOLVED","cf_temporal_object_evidence_rate":temporal_cf,"cf_complete_observation_rate":None,"ttc_temporal_object_evidence_rate":temporal_ttc,"ttc_complete_observation_rate":None,"cf_data_ready":False,"ttc_data_ready":False,"crosscheck":metrics,"local_pipeline_version":local.get("metadata.yaml",{}).get("version_string"),"pipeline_compatibility_status":"LIKELY_COMPATIBLE"}; summaries.append(summary); cross_rows.append({"clip_id":cid,**metrics})
    _dump(root/"obstacle_provenance_summary.json",{"clip_count":len(summaries),"clips":summaries,"sequence_tracks_pose_frame":"STRONGLY_SUPPORTED_NOT_VERIFIED","obstacle_transform_status":"PARTIALLY_VERIFIED","normalized_obstacle_status":"PARTIALLY_VERIFIED","empty_scene_attestation_status":"UNRESOLVED_EMPTY_ATTESTATION","complete_observation_timeline_status":"UNRESOLVED","cf_data_ready":False,"ttc_data_ready":False}); _csv(root/"five_clip_obstacle_crosscheck.csv",cross_rows)

if __name__=="__main__": main()
