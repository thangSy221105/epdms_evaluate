"""Production NuRec obstacle normalization and observation-grid helpers.

The accepted geometry contract is that ``sequence_tracks.json`` already stores
cuboids in ``NCORE_LOCAL_WORLD``.  This module therefore performs schema
normalization only: it does not apply a transform, fit a correction, infer an
empty frame, or rederive a time offset.  Observation completeness remains a
separate strict audit concern.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.epdms.observation_contract import build_ttc_projection_timestamps
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.epdms.observation_contract import build_ttc_projection_timestamps

try:
    from scripts.audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _rquat, _qnorm, _pose, _wrap
except ModuleNotFoundError:
    from audit_nurec_coordinate_alignment import interpolate_pose, load_nurec_poses, load_time_record, inverse_pose, _mm, _rquat, _qnorm, _pose, _wrap

CF_TOLERANCE_US=50_000
TTC_TOLERANCE_US=100_000
CF_HORIZON_US=4_000_000
TTC_HORIZON_US=1_000_000
TTC_STEP_US=200_000

NCORE_LOCAL_WORLD = "NCORE_LOCAL_WORLD"
OBSTACLE_REFERENCE_POINT = "cuboid_center"
GEOMETRY_BLOCK_STATUS = "CLOSED"
OBSTACLE_TRANSFORM_STATUS = "VERIFIED"
NORMALIZED_OBSTACLE_STATUS = "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES"
OBSTACLE_ROW_QUALITY_STATUS = "VERIFIED_WITH_RETAINED_LOCALIZED_ANOMALIES"

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


def load_geometry_anomaly_keys(audit_dir: Path | None, offsets: dict[str, int] | None = None):
    """Load retained center-anomaly identities from a prior forensic report.

    The report is diagnostic input only.  Rows are never removed or corrected;
    the key is used solely to carry ``RETAINED_LOCALIZED_ANOMALY`` provenance
    into normalized production records.
    """
    if audit_dir is None:
        return set(), {"source": None, "status": "NOT_PROVIDED"}
    path = Path(audit_dir)
    if path.is_dir():
        path = path / "center_outlier_full_provenance.csv"
    if not path.is_file():
        return set(), {"source": str(path), "status": "FILE_NOT_FOUND"}
    import csv as _csv_module

    offsets = offsets or {}
    keys = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in _csv_module.DictReader(handle):
            clip_id = str(row.get("clip_id") or "")
            track_id = str(row.get("track_id") or "")
            mapped = row.get("mapped_nurec_timestamp_us")
            if mapped not in (None, ""):
                timestamp = int(float(mapped))
            elif row.get("raw_timestamp_us") not in (None, ""):
                timestamp = int(float(row["raw_timestamp_us"])) + int(offsets.get(clip_id, 0))
            else:
                continue
            if clip_id and track_id:
                keys.add((clip_id, track_id, timestamp))
    return keys, {"source": str(path), "status": "READ", "retained_anomaly_count": len(keys)}


def _quaternion_to_yaw(quaternion):
    return pose_yaw(_matrix_pose([0.0, 0.0, 0.0], quaternion))


def normalize_sequence_track_row(clip_id, row, anomaly_keys=None):
    """Convert one verified sequence-track row to the canonical flat schema."""
    anomaly_keys = anomaly_keys or set()
    key = (str(clip_id), str(row["track_id"]), int(row["timestamp_us"]))
    quality = "RETAINED_LOCALIZED_ANOMALY" if key in anomaly_keys else "VERIFIED"
    quaternion = [float(value) for value in row["quaternion"]]
    dimensions = [float(value) for value in row["dimensions"]]
    return {
        "clip_id": str(clip_id),
        "trackline_id": str(row["track_id"]),
        "timestamp_us": int(row["timestamp_us"]),
        "center_x": float(row["center"][0]),
        "center_y": float(row["center"][1]),
        "center_z": float(row["center"][2]),
        "yaw_rad": float(_quaternion_to_yaw(quaternion)),
        "length_m": dimensions[0],
        "width_m": dimensions[1],
        "height_m": dimensions[2],
        "category": row.get("category"),
        "track_flag": row.get("flag"),
        "source_file": "sequence_tracks.json",
        "source_pose_index": int(row["source_pose_index"]),
        "source_track_index": int(row["source_index"]),
        "source_pose_field": "tracks_data.tracks_poses",
        "source_timestamp_field": "tracks_data.tracks_timestamps_us",
        "source_dimension_field": "cuboidtracks_data.cuboids_dims",
        "quaternion_x": quaternion[0],
        "quaternion_y": quaternion[1],
        "quaternion_z": quaternion[2],
        "quaternion_w": quaternion[3],
        "coordinate_frame": NCORE_LOCAL_WORLD,
        "reference_point": OBSTACLE_REFERENCE_POINT,
        "normalization_verified": True,
        "geometry_quality_status": quality,
        "transform_applied": False,
        "per_clip_offset_rederived": False,
    }


def normalize_sequence_tracks(clip_id, rows, anomaly_keys=None):
    """Normalize all sequence-track rows without truncation or geometry edits."""
    return [normalize_sequence_track_row(clip_id, row, anomaly_keys) for row in rows]


def normalized_to_context_obstacle(row):
    """Build the nested obstacle shape consumed by the scorer normalizer."""
    return {
        "timestamp_micros": int(row["timestamp_us"]),
        "trackline_id": row["trackline_id"],
        "category": row["category"],
        "center": {"x": row["center_x"], "y": row["center_y"], "z": row["center_z"]},
        "size": {"x": row["length_m"], "y": row["width_m"], "z": row["height_m"]},
        "orientation": {
            "x": row["quaternion_x"], "y": row["quaternion_y"],
            "z": row["quaternion_z"], "w": row["quaternion_w"],
        },
        "coordinate_frame": row["coordinate_frame"],
        "reference_point": row["reference_point"],
        "source_file": row["source_file"],
        "source_pose_index": row["source_pose_index"],
        "normalization_verified": row["normalization_verified"],
        "geometry_quality_status": row["geometry_quality_status"],
    }


def build_obstacle_context(clip_id, t0_us, normalized_rows, confirmed_empty_timestamps=None):
    """Create a separate context artifact without asserting prediction alignment."""
    obstacle = {
        "all_obstacles": [normalized_to_context_obstacle(row) for row in normalized_rows],
        "coordinate_frame": NCORE_LOCAL_WORLD,
        "reference_point": OBSTACLE_REFERENCE_POINT,
        "normalization_verified": True,
        "source_file": "sequence_tracks.json",
        "geometry_contract_status": GEOMETRY_BLOCK_STATUS,
        "obstacle_transform_status": OBSTACLE_TRANSFORM_STATUS,
        "geometry_quality_status": OBSTACLE_ROW_QUALITY_STATUS,
    }
    if confirmed_empty_timestamps:
        obstacle["confirmed_empty_timestamps_us"] = sorted(int(value) for value in confirmed_empty_timestamps)
    return {
        "clip_id": str(clip_id),
        "t0_us": int(t0_us),
        "obstacle_frame": NCORE_LOCAL_WORLD,
        "obstacle_reference_point": OBSTACLE_REFERENCE_POINT,
        "semantic_context": {"obstacle": obstacle},
        "coordinate_alignment_verified": False,
        "coordinate_alignment_status": "NOT_ASSERTED_BY_OBSTACLE_NORMALIZATION",
    }


def build_scorer_query_grid(nurec_t0, future_poses=40, frequency_hz=10.0, ttc_horizon_s=1.0, include_t0=True):
    """Reproduce score_record's CF grid and import the production TTC helper."""
    start = 0 if include_t0 else 1
    cf = [int(nurec_t0) + int(round(index * 1_000_000.0 / frequency_hz)) for index in range(start, future_poses + 1)]
    ttc = build_ttc_projection_timestamps(np.asarray(cf, dtype=np.int64), float(ttc_horizon_s))
    return cf, [int(value) for value in ttc.tolist()], {
        "cf_includes_t0": bool(include_t0),
        "cf_future_pose_count": int(future_poses),
        "cf_frequency_hz": float(frequency_hz),
        "cf_horizon_s": float(future_poses / frequency_hz),
        "ttc_horizon_s": float(ttc_horizon_s),
        "ttc_step_s": 0.2,
        "ttc_query_builder": "tools.epdms.observation_contract.build_ttc_projection_timestamps",
        "verified": True,
    }


def _nearest_timestamp(query, timestamps):
    if not timestamps:
        return None, None
    nearest = min(timestamps, key=lambda value: abs(int(value) - int(query)))
    return int(nearest), abs(int(nearest) - int(query))


def classify_scorer_queries(queries, object_rows, label_set_timestamps=None, tolerance_us=50_000, empty_timestamps=None):
    """Classify only proven object rows or explicit empty attestations."""
    by_timestamp = {}
    for row in object_rows:
        by_timestamp.setdefault(int(row["timestamp_us"]), []).append(row)
    object_timestamps = sorted(by_timestamp)
    label_set_timestamps = sorted(set(int(value) for value in (label_set_timestamps or object_timestamps)))
    empty_timestamps = set(int(value) for value in (empty_timestamps or set()))
    output = []
    for query in queries:
        query = int(query)
        obstacle_ts, obstacle_delta = _nearest_timestamp(query, object_timestamps)
        label_ts, label_delta = _nearest_timestamp(query, label_set_timestamps)
        if obstacle_ts is not None and obstacle_delta <= tolerance_us:
            state = "OBJECTS_PRESENT"
            object_count = len(by_timestamp[obstacle_ts])
            source = "sequence_tracks.json.object_rows"
            empty_verified = False
            object_verified = True
        elif query in empty_timestamps:
            state = "CONFIRMED_EMPTY"
            object_count = 0
            source = "explicit_empty_attestation"
            empty_verified = True
            object_verified = False
        else:
            state = "MISSING"
            object_count = 0
            source = "label_set_timestamp_without_empty_semantics" if label_ts is not None and label_delta <= tolerance_us else None
            empty_verified = False
            object_verified = False
        output.append({
            "query_timestamp_us": query,
            "nearest_obstacle_timestamp_us": obstacle_ts,
            "nearest_label_set_timestamp_us": label_ts,
            "obstacle_delta_us": obstacle_delta,
            "label_set_delta_us": label_delta,
            "object_count": object_count,
            "observation_state": state,
            "attestation_source": source,
            "attestation_verified": empty_verified,
            "object_presence_verified": object_verified,
        })
    return output


def strict_coverage_summary(rows, prefix):
    total = len(rows)
    objects = sum(row["observation_state"] == "OBJECTS_PRESENT" for row in rows)
    empty = sum(row["observation_state"] == "CONFIRMED_EMPTY" for row in rows)
    missing = sum(row["observation_state"] == "MISSING" for row in rows)
    return {
        f"{prefix}_REQUIRED_QUERY_COUNT": total,
        f"{prefix}_OBJECT_PRESENT_COUNT": objects,
        f"{prefix}_CONFIRMED_EMPTY_COUNT": empty,
        f"{prefix}_MISSING_COUNT": missing,
        f"{prefix}_COMPLETE_OBSERVATION_RATE": (objects + empty) / total if total else 0.0,
        f"{prefix}_TEMPORAL_OBJECT_EVIDENCE_RATE": objects / total if total else 0.0,
        f"{prefix}_DATA_READY": bool(total and missing == 0),
    }

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
    """Backward-compatible future-only grid using the production TTC helper."""
    cf, ttc, _ = build_scorer_query_grid(nurec_t0, include_t0=False)
    return cf, ttc

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

def process_clip(item, time_path: Path, anomaly_keys=None):
    clip_id = item["clip_id"]
    clip = Path(item["nurec_clip_dir"])
    record = load_time_record(time_path, clip_id, int(item.get("physicalai_t0_us", 5_100_000)))
    sequence_rows, sequence_errors, schema = load_sequence_tracks(clip / "sequence_tracks.json")
    obstacle_rows, obstacle_errors = load_obstacle_rows(clip / "clipgt" / "obstacle.parquet")
    normalized = normalize_sequence_tracks(clip_id, sequence_rows, anomaly_keys)
    cf, ttc, grid = build_scorer_query_grid(record["nurec_t0_us"])
    cf_rows = classify_scorer_queries(cf, normalized, tolerance_us=CF_TOLERANCE_US)
    ttc_rows = classify_scorer_queries(ttc, normalized, tolerance_us=TTC_TOLERANCE_US)
    cf_summary = strict_coverage_summary(cf_rows, "CF")
    ttc_summary = strict_coverage_summary(ttc_rows, "TTC")
    anomaly_count = sum(row["geometry_quality_status"] == "RETAINED_LOCALIZED_ANOMALY" for row in normalized)
    summary = {
        "clip_id": clip_id,
        "authoritative_obstacle_source": "sequence_tracks.json",
        "sequence_tracks_pose_frame": NCORE_LOCAL_WORLD,
        "sequence_tracks_pose_frame_verified": True,
        "obstacle_transform_status": OBSTACLE_TRANSFORM_STATUS,
        "normalized_obstacle_status": NORMALIZED_OBSTACLE_STATUS,
        "obstacle_row_quality_status": OBSTACLE_ROW_QUALITY_STATUS,
        "obstacle_geometry_block": GEOMETRY_BLOCK_STATUS,
        "coordinate_frame": NCORE_LOCAL_WORLD,
        "reference_point": OBSTACLE_REFERENCE_POINT,
        "time_mapping_source": record.get("source"),
        "time_mapping_verified": bool(record.get("verified", False)),
        "time_mapping_reused": True,
        "per_clip_offset_rederived": False,
        "empty_observation_attestation_status": "NOT_AVAILABLE",
        "label_set_empty_semantics_status": "UNRESOLVED",
        "complete_observation_timeline_status": "NOT_FOUND",
        "obstacle_row_count": len(normalized),
        "obstacle_parquet_row_count": len(obstacle_rows),
        "normalized_retained_anomaly_count": anomaly_count,
        "cf_data_ready": cf_summary["CF_DATA_READY"],
        "ttc_data_ready": ttc_summary["TTC_DATA_READY"],
        **cf_summary,
        **ttc_summary,
        "query_grid": grid,
        "sequence_errors": len(sequence_errors),
        "obstacle_errors": len(obstacle_errors),
        "blockers": [] if cf_summary["CF_DATA_READY"] and ttc_summary["TTC_DATA_READY"] else ["EXPLICIT_EMPTY_OBSERVATION_ATTESTATION_UNAVAILABLE"],
    }
    return {
        "summary": summary,
        "schema": schema,
        "seq_rows": sequence_rows,
        "seq_errors": sequence_errors,
        "obs_rows": obstacle_rows,
        "obs_errors": obstacle_errors,
        "normalized": normalized,
        "context": build_obstacle_context(clip_id, record["nurec_t0_us"], normalized),
        "sanity": track_sanity(sequence_rows),
        "cf": cf_rows,
        "ttc": ttc_rows,
        "cf_grid": cf,
        "ttc_grid": ttc,
        "query_grid": grid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-alignment-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry-audit-dir", default=None)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    time_path = Path(args.time_alignment_jsonl)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    offsets = {}
    for item in manifest:
        record = load_time_record(time_path, item["clip_id"], int(item.get("physicalai_t0_us", 5_100_000)))
        offsets[item["clip_id"]] = int(record["offset_us"])
    anomaly_keys, anomaly_evidence = load_geometry_anomaly_keys(args.geometry_audit_dir, offsets)
    results, all_normalized, all_context = [], [], []
    for item in manifest:
        result = process_clip(item, time_path, anomaly_keys)
        clip_id = item["clip_id"]
        output = root / clip_id
        output.mkdir(parents=True, exist_ok=True)
        results.append(result["summary"])
        all_normalized.extend(result["normalized"])
        all_context.append(result["context"])
        _jsonl(output / "normalized_obstacle.jsonl", result["normalized"])
        _dump(output / "normalized_context.json", result["context"])
        _dump(output / "normalized_obstacle_contract.json", {
            "clip_id": clip_id,
            "authoritative_obstacle_source": "sequence_tracks.json",
            "sequence_tracks_pose_frame": NCORE_LOCAL_WORLD,
            "sequence_tracks_pose_frame_verified": True,
            "obstacle_transform_status": OBSTACLE_TRANSFORM_STATUS,
            "normalized_obstacle_status": NORMALIZED_OBSTACLE_STATUS,
            "obstacle_row_quality_status": OBSTACLE_ROW_QUALITY_STATUS,
            "obstacle_geometry_block": GEOMETRY_BLOCK_STATUS,
            "retained_anomaly_count": result["summary"]["normalized_retained_anomaly_count"],
            "no_transform_applied": True,
            "per_clip_offset_rederived": False,
        })
        _dump(output / "sequence_tracks_schema.json", result["schema"])
        _dump(output / "observation_coverage_summary.json", result["summary"])
        _csv(output / "obstacle_track_sanity.csv", result["sanity"])
        _csv(output / "cf_observation_queries.csv", [{"clip_id": clip_id, **row} for row in result["cf"]])
        _csv(output / "ttc_observation_queries.csv", [{"clip_id": clip_id, **row} for row in result["ttc"]])
        print(f"CONTEXT_RECORD clip={clip_id} source=sequence_tracks.json frame={NCORE_LOCAL_WORLD} normalization_verified=true")
        print(f"CLIP_DONE clip={clip_id} cf_ready={result['summary']['CF_DATA_READY']} ttc_ready={result['summary']['TTC_DATA_READY']}")
    aggregate = {
        "pilot_clip_count": len(results),
        "clips": results,
        "authoritative_obstacle_source": "sequence_tracks.json",
        "sequence_tracks_pose_frame": NCORE_LOCAL_WORLD,
        "sequence_tracks_pose_frame_verified": True,
        "obstacle_transform_status": OBSTACLE_TRANSFORM_STATUS,
        "normalized_obstacle_status": NORMALIZED_OBSTACLE_STATUS,
        "obstacle_row_quality_status": OBSTACLE_ROW_QUALITY_STATUS,
        "obstacle_geometry_block": GEOMETRY_BLOCK_STATUS,
        "retained_anomaly_count": sum(result["normalized_retained_anomaly_count"] for result in results),
        "geometry_anomaly_evidence": anomaly_evidence,
        "per_clip_offset_rederived": False,
        "cf_data_ready_pilot": bool(results) and all(result["CF_DATA_READY"] for result in results),
        "ttc_data_ready_pilot": bool(results) and all(result["TTC_DATA_READY"] for result in results),
        "cf_data_ready_full_300": "NOT_EVALUATED",
        "ttc_data_ready_full_300": "NOT_EVALUATED",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
        "remaining_blockers": [] if results and all(result["CF_DATA_READY"] and result["TTC_DATA_READY"] for result in results) else ["EXPLICIT_EMPTY_OBSERVATION_ATTESTATION_UNAVAILABLE"],
    }
    _jsonl(root / "normalized_obstacles.jsonl", all_normalized)
    _jsonl(root / "normalized_context.jsonl", all_context)
    _dump(root / "normalized_obstacle_contract.json", aggregate)
    _dump(root / "obstacle_normalization_summary.json", aggregate)
    _dump(root / "observation_coverage_summary.json", aggregate)
    _dump(root / "geometry_anomaly_evidence.json", anomaly_evidence)

if __name__=="__main__": main()
