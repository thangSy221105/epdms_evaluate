"""NuRec obstacle provenance, ego-frame normalization and coverage audit.

This module deliberately fails closed: sequence-track geometry is normalized
only when an explicit source-frame contract is supplied.  The current NuRec
extraction does not contain that declaration, so the pilot report preserves
source evidence while keeping CF/TTC readiness false.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path
from typing import Any

try:
    from scripts.audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _rquat, _qnorm, _pose, _wrap
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _rquat, _qnorm, _pose, _wrap

CF_TOLERANCE_US=50_000
TTC_TOLERANCE_US=100_000
CF_HORIZON_US=4_000_000
TTC_HORIZON_US=1_000_000
TTC_STEP_US=200_000

def _dump(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def _csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)

def _read_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))

def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")

def _unbox_sequence(value):
    if not isinstance(value, dict): return value
    if set(value)=={"dummy_chunk_id"}: return _unbox_sequence(value["dummy_chunk_id"])
    return value

def _matrix_pose(center, quat):
    return _pose(center, quat)

def transform_object_to_ego_t0(object_pose, ego_pose_t0):
    return _mm(inverse_pose(ego_pose_t0), object_pose)

def pose_yaw(matrix): return math.atan2(matrix[1][0], matrix[0][0])

def validate_quaternion(q):
    if len(q)!=4 or any(not math.isfinite(float(v)) for v in q): raise ValueError("INVALID_QUATERNION")
    n=math.sqrt(sum(float(v)*float(v) for v in q))
    if n<=1e-12: raise ValueError("INVALID_QUATERNION")
    return _qnorm([float(v) for v in q])

def load_sequence_tracks(path: Path):
    raw=_unbox_sequence(_read_json(path)); tracks=raw.get("tracks_data",{}); cuboids=raw.get("cuboidtracks_data",{})
    ids=tracks.get("tracks_id",[]); poses=tracks.get("tracks_poses",[]); times=tracks.get("tracks_timestamps_us",[]); labels=tracks.get("tracks_label_class",[]); flags=tracks.get("tracks_flags",[]); dims=cuboids.get("cuboids_dims",[])
    if not (len(ids)==len(poses)==len(times)==len(labels)==len(dims)): raise ValueError("SEQUENCE_TRACKS_LENGTH_MISMATCH")
    rows=[]; errors=[]
    for i,track_id in enumerate(ids):
        prev=None
        for j,(ts,flat) in enumerate(zip(times[i],poses[i])):
            try:
                if len(flat)!=7: raise ValueError("POSE_NOT_XYZ_QUAT7")
                q=validate_quaternion(flat[3:7]); center=[float(v) for v in flat[:3]]
                if any(not math.isfinite(v) for v in center): raise ValueError("NONFINITE_CENTER")
                size=[float(v) for v in dims[i]]
                if len(size)!=3 or any(not math.isfinite(v) or v<=0 for v in size): raise ValueError("INVALID_DIMENSIONS")
                ts=int(ts); rows.append({"track_id":str(track_id),"timestamp_us":ts,"center":center,"quaternion":q,"dimensions":size,"category":labels[i],"flag":flags[i] if i<len(flags) else None,"source_index":i,"source_pose_index":j})
                if prev is not None and ts<=prev: errors.append({"track_id":str(track_id),"type":"NONMONOTONIC_OR_DUPLICATE_TIMESTAMP","timestamp_us":ts})
                prev=ts
            except Exception as e: errors.append({"track_id":str(track_id),"source_pose_index":j,"type":str(e)})
    return rows,errors,{"top_level_keys":list(raw),"tracks_data_keys":list(tracks),"cuboidtracks_data_keys":list(cuboids),"track_count":len(ids),"pose_count":sum(len(x) for x in poses),"timestamp_count":sum(len(x) for x in times),"labels":sorted(set(map(str,labels))),"pose_representation":"[x,y,z,qx,qy,qz,qw]","dimension_representation":"[x,y,z]"}

def load_obstacle_rows(path: Path):
    try:
        import pyarrow.parquet as pq
        raw=pq.read_table(path).to_pylist()
    except Exception as e: return [],[{"type":"READ_ERROR","message":str(e)}]
    rows=[]; errors=[]
    for i,r in enumerate(raw):
        try:
            k=r["key"]; o=r["obstacle"]; q=validate_quaternion([o["orientation"][x] for x in ("x","y","z","w")]); rows.append({"track_id":str(o["trackline_id"]),"timestamp_us":int(k["timestamp_micros"]),"center":[float(o["center"][x]) for x in ("x","y","z")],"quaternion":q,"dimensions":[float(o["size"][x]) for x in ("x","y","z")],"category":o.get("category"),"source_index":i})
        except Exception as e: errors.append({"source_index":i,"type":str(e)})
    return rows,errors

def _percentile(values, q):
    if not values: return None
    values=sorted(values); return values[min(len(values)-1,max(0,math.ceil(q*len(values))-1))]

def track_sanity(rows):
    by={}
    for r in rows: by.setdefault(r["track_id"],[]).append(r)
    out=[]
    for tid,rs in by.items():
        rs=sorted(rs,key=lambda x:x["timestamp_us"]); gaps=[b["timestamp_us"]-a["timestamp_us"] for a,b in zip(rs,rs[1:])]; speeds=[]
        for a,b in zip(rs,rs[1:]):
            dt=(b["timestamp_us"]-a["timestamp_us"])/1e6
            if dt>0: speeds.append(math.dist(a["center"],b["center"])/dt)
        yaw=[pose_yaw(_matrix_pose(r["center"],r["quaternion"])) for r in rs]; yaw_steps=[abs(_wrap(b-a)) for a,b in zip(yaw,yaw[1:])]
        out.append({"track_id":tid,"observation_count":len(rs),"timestamp_min_us":rs[0]["timestamp_us"],"timestamp_max_us":rs[-1]["timestamp_us"],"median_gap_us":statistics.median(gaps) if gaps else None,"p95_gap_us":_percentile(gaps,.95),"max_gap_us":max(gaps) if gaps else None,"max_estimated_speed_mps":max(speeds) if speeds else None,"max_yaw_step_rad":max(yaw_steps) if yaw_steps else None,"duplicate_or_nonmonotonic":any(g<=0 for g in gaps),"flag_only_teleportation":any(v>100 for v in speeds),"flag_only_huge_yaw_jump":any(v>math.pi for v in yaw_steps)})
    return out

def nearest_state(query, observations, tolerance_us):
    if not observations: return "UNKNOWN",None,None
    nearest=min(observations,key=lambda x:abs(x-query)); delta=abs(nearest-query)
    if delta>tolerance_us: return "UNKNOWN",None,delta
    return "OBJECTS_PRESENT",nearest,delta

def build_queries(nurec_t0):
    cf=[nurec_t0+i*100_000 for i in range(1,41)]
    ttc=sorted(set(t+dt for t in cf for dt in range(0,TTC_HORIZON_US+1,TTC_STEP_US)))
    return cf,ttc

def classify_queries(queries, object_timestamps, tolerance_us, empty_timestamps=None):
    empty_timestamps=set(empty_timestamps or []); rows=[]
    for q in queries:
        if object_timestamps:
            nearest=min(object_timestamps,key=lambda x:abs(x-q)); delta=abs(nearest-q)
        else: nearest=None; delta=None
        if nearest is not None and delta<=tolerance_us: state="EXACT_OBJECT" if delta==0 else "NEAREST_OBJECT_WITHIN_TOLERANCE"; att="obstacle_rows"
        elif q in empty_timestamps: state="EXACT_CONFIRMED_EMPTY"; att="explicit_empty_attestation"
        elif nearest is None or (delta is not None and delta>tolerance_us): state="OUT_OF_RANGE" if not object_timestamps or q<min(object_timestamps) or q>max(object_timestamps) else "UNKNOWN"; att=None
        else: state="UNKNOWN"; att=None
        rows.append({"timestamp_us":q,"observation_state":state,"object_count":sum(1 for t in object_timestamps if abs(t-q)<=tolerance_us),"attestation_source":att,"attestation_verified":state=="CONFIRMED_EMPTY","nearest_source_timestamp_us":nearest,"delta_us":delta})
    return rows

def coverage_summary(rows, prefix):
    counts={s:sum(r["observation_state"]==s for r in rows) for s in ("EXACT_OBJECT","NEAREST_OBJECT_WITHIN_TOLERANCE","EXACT_CONFIRMED_EMPTY","UNKNOWN","OUT_OF_RANGE")}; total=len(rows); objects=counts["EXACT_OBJECT"]+counts["NEAREST_OBJECT_WITHIN_TOLERANCE"]; empty=counts["EXACT_CONFIRMED_EMPTY"]; return {f"{prefix}_REQUIRED_QUERY_COUNT":total,f"{prefix}_OBJECT_QUERY_COUNT":objects,f"{prefix}_CONFIRMED_EMPTY_COUNT":empty,f"{prefix}_UNKNOWN_COUNT":counts["UNKNOWN"],f"{prefix}_OUT_OF_RANGE_COUNT":counts["OUT_OF_RANGE"],f"{prefix}_COVERAGE_RATE":(objects+empty)/total if total else 0.0}

def process_clip(item, time_path: Path, root: Path):
    cid=item["clip_id"]; clip=Path(item["nurec_clip_dir"]); rec=load_time_record(time_path,cid,int(item.get("physicalai_t0_us",5_100_000))); seq_rows,seq_errors,schema=load_sequence_tracks(clip/"sequence_tracks.json"); obs_rows,obs_errors=load_obstacle_rows(clip/"clipgt"/"obstacle.parquet")
    cf,ttc=build_queries(rec["nurec_t0_us"]); object_ts=sorted({r["timestamp_us"] for r in seq_rows}); cf_rows=classify_queries(cf,object_ts,CF_TOLERANCE_US); ttc_rows=classify_queries(ttc,object_ts,TTC_TOLERANCE_US); all_cov=cf_rows+ttc_rows
    map_status="PROVENANCE_AVAILABLE_NOT_INTEGRATED" if (clip/"rig_trajectories.json").is_file() and (clip/"map.xodr").is_file() else "UNRESOLVED"
    normalized=[]
    for r in seq_rows[:1000]: normalized.append({"clip_id":cid,"track_id":r["track_id"],"source_timestamp_us":r["timestamp_us"],"normalized_timestamp_us":r["timestamp_us"],"physicalai_relative_us":r["timestamp_us"]-rec["offset_us"],"common_frame":"EGO_AT_T0","center_x":r["center"][0],"center_y":r["center"][1],"center_z":r["center"][2],"yaw_rad":pose_yaw(_matrix_pose(r["center"],r["quaternion"])),"length_m":r["dimensions"][0],"width_m":r["dimensions"][1],"height_m":r["dimensions"][2],"category":r["category"],"source_file":"sequence_tracks.json","source_frame":"UNRESOLVED","transform_source":"NOT_APPLIED_FRAME_PROVENANCE_MISSING","orientation_source":"sequence_tracks.tracks_poses","interpolated":False,"normalization_verified":False})
    summary={"clip_id":cid,"authoritative_obstacle_source":"sequence_tracks.json_candidate","sequence_tracks_pose_frame":"UNRESOLVED","sequence_tracks_pose_frame_verified":False,"obstacle_transform_status":"UNRESOLVED","normalized_obstacle_status":"PARTIALLY_VERIFIED","common_frame":"EGO_AT_T0","time_mapping_reused":True,"per_clip_offset_rederived":False,"empty_scene_attestation_status":"UNRESOLVED_EMPTY_ATTESTATION","observation_coverage_status":"UNRESOLVED_EMPTY_ATTESTATION","obstacle_row_count":len(seq_rows),"obstacle_parquet_row_count":len(obs_rows),"cf_data_ready":False,"ttc_data_ready":False,**coverage_summary(cf_rows,"CF"),**coverage_summary(ttc_rows,"TTC"),"map_status":map_status,"blockers":["SEQUENCE_TRACKS_POSE_FRAME_NOT_PROVEN","EMPTY_SCENE_ATTESTATION_UNRESOLVED","OBSTACLE_TRANSFORM_NOT_VERIFIED"]}
    return {"summary":summary,"schema":schema,"seq_rows":seq_rows,"seq_errors":seq_errors,"obs_rows":obs_rows,"obs_errors":obs_errors,"coverage":all_cov,"normalized":normalized,"sanity":track_sanity(seq_rows),"cf":cf_rows,"ttc":ttc_rows}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--time-alignment-jsonl",required=True); p.add_argument("--output-root",required=True); a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); time_path=Path(a.time_alignment_jsonl); manifest=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]; results=[]
    for item in manifest:
        r=process_clip(item,time_path,root); cid=item["clip_id"]; out=root/cid; out.mkdir(parents=True,exist_ok=True); results.append(r["summary"])
        _dump(out/"sequence_tracks_schema.json",r["schema"]); _dump(out/"obstacle_source_comparison.json",{"candidate_sources":{"sequence_tracks.json":{"row_count":len(r["seq_rows"]),"has_track_poses":True,"has_dimensions":True,"has_orientation":True,"frame_status":"UNRESOLVED"},"clipgt/obstacle.parquet":{"row_count":len(r["obs_rows"]),"has_track_poses":True,"has_dimensions":True,"has_orientation":True,"reference_frame_field":False,"frame_status":"UNRESOLVED"}},"public_upstream_provenance":{"ncore_cuboid_contract":"https://nvidia.github.io/nurec/ncore/reference/apis/data.v3.html","ncore_conventions":"https://nvidia.github.io/ncore/data/conventions","local_sequence_frame_claim":"NOT_PROVEN_BY_LOCAL_FILE"},"authoritative_obstacle_source":"sequence_tracks.json_candidate","selection_status":"CANDIDATE_ONLY"}); _dump(out/"obstacle_coordinate_contract.json",{"common_frame":"EGO_AT_T0","source_frame":"UNRESOLVED","transform_verified":False,"transform_source":"NOT_APPLIED_FRAME_PROVENANCE_MISSING"}); _dump(out/"obstacle_orientation_contract.json",{"quaternion_order":"qx,qy,qz,qw","verified_from_local_schema":True,"upstream_frame_semantics_verified":False}); _dump(out/"obstacle_dimension_contract.json",{"order":"x,y,z","unit":"UNRESOLVED_BUT_POSITIVE_FINITE","provenance":"sequence_tracks.cuboidtracks_data.cuboids_dims"}); _dump(out/"obstacle_transform_graph.json",{"common_frame":"EGO_AT_T0","status":"UNRESOLVED","edges":[]}); _dump(out/"observation_attestation_contract.json",{"status":"UNRESOLVED_EMPTY_ATTESTATION","pose_timestamps_not_used_as_empty_attestation":True,"source":None}); _dump(out/"observation_coverage_summary.json",r["summary"]); _dump(out/"obstacle_normalization_summary.json",r["summary"]); _jsonl(out/"normalized_obstacle_sample.jsonl",r["normalized"]); _csv(out/"obstacle_track_sanity.csv",r["sanity"]); _csv(out/"obstacle_source_crosscheck.csv",[{"sequence_track_id":x["track_id"],"sequence_timestamp_us":x["timestamp_us"],"matching_obstacle_rows":sum(y["track_id"]==x["track_id"] and y["timestamp_us"]==x["timestamp_us"] for y in r["obs_rows"])} for x in r["seq_rows"][:1000]]); _csv(out/"observation_coverage.csv",r["coverage"]); _csv(out/"cf_required_query_grid.csv",r["cf"]); _csv(out/"ttc_required_query_grid.csv",r["ttc"])
        print(f"CONTEXT_RECORD clip={cid} intents=obstacle sources=sequence_tracks,obstacle aligned=true transformed=false ready=false"); print(f"CLIP_DONE clip={cid} status=complete")
    aggregate={"clip_count":len(results),"clips":results,"authoritative_obstacle_source":"sequence_tracks.json_candidate","empty_scene_attestation_status":"UNRESOLVED_EMPTY_ATTESTATION","cf_data_ready":False,"ttc_data_ready":False,"remaining_blockers":["SEQUENCE_TRACKS_POSE_FRAME_NOT_PROVEN","EMPTY_SCENE_ATTESTATION_UNRESOLVED","OBSTACLE_TRANSFORM_NOT_VERIFIED"]}
    _dump(root/"obstacle_normalization_summary.json",aggregate); _dump(root/"observation_coverage_summary.json",aggregate); _dump(root/"obstacle_source_comparison.json",{"clip_count":len(results),"authoritative_obstacle_source":"sequence_tracks.json_candidate","selection_status":"CANDIDATE_ONLY","public_upstream_provenance":"NCore CuboidTrackObservation reference-frame contract; local sequence frame not proven"}); _dump(root/"obstacle_coordinate_contract.json",{"common_frame":"EGO_AT_T0","transform_status":"UNRESOLVED","time_mapping_reused":True}); _dump(root/"obstacle_orientation_contract.json",{"quaternion_order":"qx,qy,qz,qw","local_schema_verified":True,"frame_semantics_verified":False}); _dump(root/"obstacle_dimension_contract.json",{"order":"x,y,z","positive_finite_validated":True,"unit":"UNRESOLVED"}); _dump(root/"obstacle_transform_graph.json",{"common_frame":"EGO_AT_T0","status":"UNRESOLVED","transform_not_applied":True}); _dump(root/"observation_attestation_contract.json",{"status":"UNRESOLVED_EMPTY_ATTESTATION","pose_timestamps_not_used_as_empty_attestation":True})

if __name__=="__main__": main()
