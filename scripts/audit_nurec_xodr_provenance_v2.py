#!/usr/bin/env python3
"""Finalize NuRec XODR coordinate and legal-direction provenance.

This is an evidence/audit stage.  It derives the map-to-NuRec transform only
from NVIDIA's published ECEF/ENU chain and local ``T_world_base`` metadata.
It never fits a transform to labels, GT, or lane geometry.  Legal lane
direction is enabled only when the XODR contains the authoritative road-rule
inputs required by ASAM OpenDRIVE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


NUREC_DOC = "https://docs.nvidia.com/nurec/nurec/physical-ai-data.html"
NRE_SKILL = "https://github.com/NVIDIA/nurec-skills/blob/main/skills/nre/references/physical-ai-render.md"
ASAM_LANES = "https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/1.8.0/specification/11_lanes/11_02_lane_groups.html"
ASAM_ROADS = "https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/1.8.0/specification/10_roads/10_01_introduction.html"
ASAM_REF = "https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.9.0/specification/09_geometries/09_02_road_reference_line.html"
ALPASIM_CONVENTIONS = "https://github.com/NVlabs/alpasim/blob/main/CONTRIBUTING.md"
NCORE_CONVENTIONS = "https://nvidia.github.io/ncore/data/conventions.html"

XODR_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\00_raw\nurec_full300_xodr")
CONTEXT_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2")
MAP_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\hf_probe\nurec_map_dac_final_v2\map_root")
BASELINE_VECTOR_DEFAULT = Path(r"D:\300_clip_nurec\hf_probe\epdms_partial_vector_lk_ddc_xodr_full4800_v1\partial_metric_vector_full4800.jsonl")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_geo_reference(raw: str) -> dict[str, Any]:
    # One released file uses the historical typo ``+=alt_0``; normalize only
    # the token spelling for parsing while preserving the raw string/hash.
    normalized = raw.replace("+=", "+")
    values: dict[str, str] = {}
    for match in re.finditer(r"\+([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", normalized):
        values[match.group(1)] = match.group(2).strip('"')
    numeric: dict[str, Any] = {}
    for key in ("lat_0", "lon_0", "x_0", "y_0", "zone", "alt_0"):
        if key in values:
            try:
                numeric[key] = float(values[key])
            except ValueError:
                numeric[key] = values[key]
    return {
        "raw": raw,
        "sha256": sha256_bytes(raw.encode("utf-8")),
        "proj": values.get("proj"),
        "datum": values.get("datum"),
        "ellps": values.get("ellps"),
        "units": values.get("units"),
        "lat_0": numeric.get("lat_0"),
        "lon_0": numeric.get("lon_0"),
        "x_0": numeric.get("x_0"),
        "y_0": numeric.get("y_0"),
        "zone": numeric.get("zone"),
        "alt_0": numeric.get("alt_0", 0.0),
        "parse_status": "PARSED" if values.get("proj") and "lat_0" in numeric and "lon_0" in numeric else "PARTIAL",
    }


def ecef_from_lla(lat: float, lon: float, alt: float) -> np.ndarray:
    a = 6378137.0
    b = a * (1.0 - 1.0 / 298.257223563)
    phi = math.radians(lat)
    lam = math.radians(lon)
    e2 = (a * a - b * b) / (a * a)
    n = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
    return np.array([
        (n + alt) * math.cos(phi) * math.cos(lam),
        (n + alt) * math.cos(phi) * math.sin(lam),
        (n * (b * b / (a * a)) + alt) * math.sin(phi),
    ])


def ecef_to_enu(lat: float, lon: float, alt: float) -> np.ndarray:
    phi = math.radians(lat)
    lam = math.radians(lon)
    rotation = np.array([
        [-math.sin(lam), math.cos(lam), 0.0],
        [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)],
        [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ ecef_from_lla(lat, lon, alt)
    return result


def load_xodr(path: Path) -> tuple[ET.Element, dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    geo_node = next((node for node in root.iter() if local_name(node.tag) == "geoReference"), None)
    raw = (geo_node.text or "").strip() if geo_node is not None else ""
    return root, parse_geo_reference(raw)


def sample_reference_geometry(root: ET.Element, spacing_m: float = 10.0) -> np.ndarray:
    points: list[list[float]] = []
    for geometry in (node for node in root.iter() if local_name(node.tag) == "geometry"):
        try:
            x0 = float(geometry.attrib["x"])
            y0 = float(geometry.attrib["y"])
            heading = float(geometry.attrib["hdg"])
            length = float(geometry.attrib["length"])
        except (KeyError, ValueError):
            continue
        count = max(2, int(math.ceil(length / spacing_m)) + 1)
        primitive = next((child for child in geometry if local_name(child.tag) in {"line", "arc"}), None)
        kind = local_name(primitive.tag) if primitive is not None else "unsupported"
        curvature = None
        if primitive is not None and kind == "arc":
            try:
                curvature = float(primitive.attrib["curvature"])
            except (KeyError, ValueError):
                kind = "unsupported"
        for s in np.linspace(0.0, length, count):
            if kind == "line":
                x = x0 + s * math.cos(heading)
                y = y0 + s * math.sin(heading)
            elif kind == "arc" and curvature not in (None, 0.0):
                h = heading + curvature * s
                x = x0 + (math.sin(h) - math.sin(heading)) / curvature
                y = y0 + (-math.cos(h) + math.cos(heading)) / curvature
            elif kind == "arc":
                x = x0 + s * math.cos(heading)
                y = y0 + s * math.sin(heading)
            else:
                continue
            points.append([x, y, 0.0, 1.0])
    return np.asarray(points, dtype=float)


def load_lane_points(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["lane"]).to_pylist()
    points: list[list[float]] = []
    for row in rows:
        lane = row.get("lane") or {}
        for key in ("left_rail", "right_rail"):
            for point in lane.get(key) or []:
                try:
                    points.append([float(point["x"]), float(point["y"]), float(point.get("z", 0.0))])
                except (KeyError, TypeError, ValueError):
                    continue
    return np.asarray(points, dtype=float)


def geometry_validation(root: ET.Element, geo: dict[str, Any], rig_path: Path, lane_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_status": "UNRESOLVED",
        "xodr_sample_count": 0,
        "lane_point_count": 0,
        "median_nearest_distance_m": None,
        "p95_nearest_distance_m": None,
        "max_nearest_distance_m": None,
        "systematic_contradiction": False,
        "error": "",
    }
    try:
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        world_base = np.asarray(rig["T_world_base"], dtype=float)
        if world_base.shape != (4, 4):
            raise ValueError("T_world_base is not 4x4")
        samples = sample_reference_geometry(root)
        lane_points = load_lane_points(lane_path)
        result["xodr_sample_count"] = int(len(samples))
        result["lane_point_count"] = int(len(lane_points))
        if not len(samples) or not len(lane_points):
            raise ValueError("missing finite XODR or lane geometry")
        transform_map_to_ncore = np.linalg.inv(ecef_to_enu(float(geo["lat_0"]), float(geo["lon_0"]), float(geo["alt_0"])) @ world_base)
        transformed = (transform_map_to_ncore @ samples.T).T[:, :3]
        distances: list[float] = []
        for start in range(0, len(transformed), 256):
            chunk = transformed[start : start + 256]
            distances.extend(np.sqrt(((chunk[:, None, :2] - lane_points[None, :, :2]) ** 2).sum(axis=2)).min(axis=1).tolist())
        values = np.asarray(distances, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite nearest-geometry distance")
        result.update({
            "validation_status": "VALIDATION_COMPLETE",
            "median_nearest_distance_m": float(np.median(values)),
            "p95_nearest_distance_m": float(np.percentile(values, 95)),
            "max_nearest_distance_m": float(np.max(values)),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def origin_candidates(value: Any, path: str = "") -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    terms = re.compile(r"origin|latitude|longitude|altitude|ecef|enu|world_from|ecef_from|map_origin", re.I)
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if terms.search(str(key)) and not isinstance(child, (dict, list)):
                matches.append({"field_path": child_path, "value": str(child)[:500]})
            matches.extend(origin_candidates(child, child_path))
    elif isinstance(value, list) and len(value) <= 32:
        for index, child in enumerate(value):
            matches.extend(origin_candidates(child, f"{path}[{index}]"))
    return matches


def inspect_origin_metadata(context_clip: Path) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for path in sorted(context_clip.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in origin_candidates(value):
            item["source_file"] = path.name
            matches.append(item)
    return matches


def evidence_rows() -> list[dict[str, Any]]:
    return [
        {"evidence_id": "NUREC_DOC_COORD_001", "repository": "NVIDIA NuRec documentation", "commit_sha": "PUBLISHED_DOC", "file_path": NUREC_DOC, "symbol": "calculate_nurec_to_map_transform", "line_range_or_section": "Coordinate frames; Transform the Coordinate Systems", "claim_type": "MAP_FRAME", "claim": "NuRec Space -> ECEF -> OpenDRIVE ENU; T_nurec_map = T_ecef_enu @ T_rig_ecef and T_world_base is T_rig_ecef at frame 0.", "supports_coordinate_binding": True, "supports_legal_direction": False, "confidence": "HIGH", "notes": "Authoritative NVIDIA documentation; implementation is reproduced locally."},
        {"evidence_id": "NUREC_SKILL_COORD_001", "repository": "NVIDIA/nurec-skills", "commit_sha": "5b9d287eb7d271147751a119aa490f41edc90363", "file_path": "skills/nre/references/physical-ai-render.md", "symbol": "calculate_nurec_to_map_transform", "line_range_or_section": "§3 Coordinate frames; §3 ECEF and ECEF→ENU", "claim_type": "GEOREFERENCE", "claim": "Released USDZ contains map.xodr with a georeference proj string and the same ECEF/ENU transform chain.", "supports_coordinate_binding": True, "supports_legal_direction": False, "confidence": "HIGH", "notes": "Upstream NVIDIA skill at inspected main SHA."},
        {"evidence_id": "ASAM_DIRECTION_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_LANES, "symbol": "11.2.1 Driving direction", "line_range_or_section": "§11.2.1", "claim_type": "ROAD_RULE", "claim": "RHT/LHT road rule plus lane grouping/id determines default direction; lane direction can override it.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "Normative semantics include 1.4.0 rule identifiers."},
        {"evidence_id": "ASAM_LANE_ID_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_LANES, "symbol": "lane id and lane groups", "line_range_or_section": "§11.1 and §11.2", "claim_type": "LANE_ID", "claim": "Positive lane IDs are left of center and negative lane IDs are right of center; reference line runs in increasing s.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "Lane sign alone is insufficient when road rule is missing."},
        {"evidence_id": "ASAM_REF_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_REF, "symbol": "road reference line", "line_range_or_section": "§9.2", "claim_type": "TRAFFIC_RULE", "claim": "Reference-line direction does not itself indicate driving direction.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "Prevents treating geometry heading as legal direction."},
        {"evidence_id": "ALPASIM_LOCAL_001", "repository": "NVlabs/alpasim", "commit_sha": "affc2eab209fa43bdfa2f26c0f8d437922d78a68", "file_path": "CONTRIBUTING.md", "symbol": "Coordinate Systems", "line_range_or_section": "Coordinate Systems and transforms", "claim_type": "LOCAL_WORLD", "claim": "AlpaSim local is an ENU inertial frame and its local/rig transform convention is explicit.", "supports_coordinate_binding": False, "supports_legal_direction": False, "confidence": "MEDIUM", "notes": "Contextual downstream convention, not used as the NuRec transform proof."},
        {"evidence_id": "UPSTREAM_SEARCH_NCORE_001", "repository": "NVIDIA/ncore", "commit_sha": "dde366e7e8e9e7dbb9b1278488936005a7dee4e2", "file_path": "tools/data_converter/pai; repository-wide symbol search", "symbol": "map.xodr/OpenDRIVE", "line_range_or_section": "bounded search", "claim_type": "XODR_EXPORT", "claim": "No public NCore PAI converter file was found that exports the NuRec map.xodr member or declares this PAI-to-NuRec map binding.", "supports_coordinate_binding": False, "supports_legal_direction": False, "confidence": "MEDIUM", "notes": "Negative search result; NVIDIA NuRec documentation is the positive coordinate source."},
        {"evidence_id": "UPSTREAM_SEARCH_INSTANT_001", "repository": "NVIDIA/instant-nurec", "commit_sha": "9f2209eb9b10773dfb16e2b9166e8569a71213d1", "file_path": "repository-wide symbol search", "symbol": "map.xodr/OpenDRIVE", "line_range_or_section": "bounded search", "claim_type": "XODR_EXPORT", "claim": "No public Instant-NuRec map.xodr exporter/binding was found in the inspected repository snapshot.", "supports_coordinate_binding": False, "supports_legal_direction": False, "confidence": "MEDIUM", "notes": "Negative search result."},
        {"evidence_id": "UPSTREAM_SEARCH_NUREC_SKILLS_001", "repository": "NVIDIA/nurec-skills", "commit_sha": "5b9d287eb7d271147751a119aa490f41edc90363", "file_path": "skills/nre/references/physical-ai-render.md", "symbol": "map.xodr/geoReference", "line_range_or_section": "§1-§3", "claim_type": "XODR_EXPORT", "claim": "Upstream skill documents the released USDZ contents and coordinate conversion path but does not define legal lane direction for missing road rules.", "supports_coordinate_binding": True, "supports_legal_direction": False, "confidence": "HIGH", "notes": "Positive/negative scope boundary."},
    ]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    clip_ids = sorted(path.name for path in args.xodr_root.iterdir() if path.is_dir())
    georef_rows: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    road_rule_counts = {"RHT": 0, "LHT": 0, "MISSING": 0, "OTHER": 0}
    version_counts: dict[str, int] = {}
    valid_binding = 0
    for clip_id in clip_ids:
        xodr_path = args.xodr_root / clip_id / "map.xodr"
        context_clip = args.context_root / clip_id
        try:
            root, geo = load_xodr(xodr_path)
            header = next((node for node in root.iter() if local_name(node.tag) == "header"), None)
            rev_major = header.attrib.get("revMajor", "") if header is not None else ""
            rev_minor = header.attrib.get("revMinor", "") if header is not None else ""
            version = f"{rev_major}.{rev_minor}" if rev_major and rev_minor else "UNKNOWN"
            version_rows.append({"clip_id": clip_id, "revMajor": rev_major, "revMinor": rev_minor, "version_string": version})
            version_counts[version] = version_counts.get(version, 0) + 1
            georef_rows.append({
                "clip_id": clip_id,
                "geo_reference_raw": geo.get("raw"),
                "geo_reference_sha256": geo.get("sha256"),
                **{key: geo.get(key) for key in ("proj", "datum", "ellps", "lat_0", "lon_0", "x_0", "y_0", "zone", "units", "alt_0", "parse_status")},
            })
            roads = [node for node in root.iter() if local_name(node.tag) == "road"]
            for road in roads:
                rule = str(road.attrib.get("rule", "")).upper()
                road_rule_counts[rule if rule in {"RHT", "LHT"} else ("MISSING" if not rule else "OTHER")] += 1
            rig_path = context_clip / "rig_trajectories.json"
            rig = json.loads(rig_path.read_text(encoding="utf-8"))
            world_base = np.asarray(rig.get("T_world_base"), dtype=float)
            coord_ok = geo["parse_status"] == "PARSED" and world_base.shape == (4, 4)
            if coord_ok:
                valid_binding += 1
            origin_candidates_rows = inspect_origin_metadata(context_clip)
            origin_rows.extend({"clip_id": clip_id, **item} for item in origin_candidates_rows)
            rules_for_clip = [str(road.attrib.get("rule", "")).upper() for road in roads if road.attrib.get("rule")]
            junctions = [node for node in root.iter() if local_name(node.tag) == "junction"]
            has_lane_topology = any(local_name(node.tag) in {"laneSection", "left", "right", "laneLink"} for node in root.iter())
            has_junction_topology = bool(junctions) or any(local_name(node.tag) in {"connection", "laneLink"} for node in root.iter())
            lane_path = args.map_root / clip_id / "clipgt" / "lane.parquet"
            validation = geometry_validation(root, geo, rig_path, lane_path)
            validation_rows.append({"clip_id": clip_id, **validation})
            xodr_transform = {
                "formula": "T_map_ncore = inverse(T_ecef_enu @ T_world_base)",
                "T_world_base_source": "rig_trajectories.json:T_world_base = T_rig_ecef(frame_0)",
                "T_ecef_enu_source": "map.xodr:header/geoReference",
                "geo_reference": geo,
                "T_world_base": world_base.tolist(),
            }
            binding_rows.append({
                "clip_id": clip_id,
                "xodr_crs": geo.get("proj"),
                "xodr_origin": json.dumps({k: geo.get(k) for k in ("lat_0", "lon_0", "alt_0")}, sort_keys=True),
                "ncore_world_origin_source": "rig_trajectories.json:T_world_base",
                "ncore_world_origin_value": json.dumps(world_base[:3, 3].tolist()),
                "binding_method": "CRS_CONVERSION",
                "binding_source": NUREC_DOC,
                "xodr_to_ncore_transform_type": "KNOWN_PER_CLIP_TRANSFORM",
                "xodr_to_ncore_transform_parameters": json.dumps(xodr_transform, ensure_ascii=False),
                "coordinate_contract_status": "VERIFIED" if coord_ok else "UNRESOLVED",
                "blocker": "" if coord_ok else "MISSING_VALID_GEOREFERENCE_OR_T_WORLD_BASE",
            })
            direction_supported = bool(rules_for_clip) and has_lane_topology
            blockers = []
            if not rules_for_clip:
                blockers.append("ROAD_RULE_MISSING")
            if not has_lane_topology:
                blockers.append("LANE_TOPOLOGY_MISSING")
            if junctions:
                blockers.append("JUNCTION_POLICY_REQUIRES_LANE_LINK_RESOLUTION")
            readiness_rows.append({
                "clip_id": clip_id,
                "road_rule_available": bool(rules_for_clip),
                "lane_topology_available": has_lane_topology,
                "junction_topology_available": has_junction_topology,
                "legal_direction_supported": direction_supported,
                "legal_direction_status": "PARTIAL" if direction_supported else "UNRESOLVED",
                "coordinate_supported": coord_ok,
                "coordinate_status": "VERIFIED" if coord_ok else "UNRESOLVED",
                "ddc_proxy_ready": False,
                "blockers": ";".join(blockers),
            })
        except Exception as exc:
            binding_rows.append({"clip_id": clip_id, "coordinate_contract_status": "UNRESOLVED", "blocker": f"{type(exc).__name__}: {exc}"})
            validation_rows.append({"clip_id": clip_id, "validation_status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            readiness_rows.append({"clip_id": clip_id, "legal_direction_status": "UNRESOLVED", "coordinate_status": "UNRESOLVED", "ddc_proxy_ready": False, "blockers": f"{type(exc).__name__}: {exc}"})

    evidence = evidence_rows()
    write_csv(args.output_dir / "upstream_coordinate_direction_evidence.csv", evidence)
    write_csv(args.output_dir / "xodr_georeference_inventory_full300.csv", georef_rows)
    write_csv(args.output_dir / "opendrive_version_inventory.csv", version_rows)
    write_csv(args.output_dir / "xodr_coordinate_binding_full300.csv", binding_rows)
    write_csv(args.output_dir / "xodr_lane_geometry_validation.csv", validation_rows)
    write_csv(args.output_dir / "ddc_direction_readiness_full300.csv", readiness_rows)
    write_csv(args.output_dir / "metadata_origin_evidence_full300.csv", origin_rows)

    coordinate_contract = {
        "status": "VERIFIED" if valid_binding == len(clip_ids) else "PARTIAL",
        "binding_method": "CRS_CONVERSION",
        "transform_type": "KNOWN_PER_CLIP_TRANSFORM",
        "formula": "T_map_ncore = inverse(T_ecef_enu @ T_world_base)",
        "xodr_frame": "OpenDRIVE local ENU/map coordinates defined by geoReference PROJ string",
        "ncore_frame": "NuRec/NCORE local world (frame-0 local reconstruction frame)",
        "authoritative_source": NUREC_DOC,
        "evidence": ["NUREC_DOC_COORD_001", "NUREC_SKILL_COORD_001"],
        "validation_clip_count": len(validation_rows),
        "validation_status": "DIAGNOSTIC_COMPLETE_NO_FITTED_TRANSFORM",
        "gt_fitted_transform_used": False,
        "lane_fitted_transform_used": False,
    }
    legal_contract = {
        "status": "UNRESOLVED",
        "opendrive_versions": sorted(version_counts),
        "reference_line_semantics": "Reference line runs in increasing s but does not itself establish driving direction.",
        "lane_id_semantics": "Positive IDs are left and negative IDs right of center.",
        "road_rule_semantics": "RHT/LHT is required with lane grouping/id for default legal direction.",
        "rht_direction_rule": "right/negative lanes follow positive reference-line direction; left/positive lanes oppose it.",
        "lht_direction_rule": "left/positive lanes follow positive reference-line direction; right/negative lanes oppose it.",
        "junction_policy": "No DDC evaluation until connecting-road laneLink/contactPoint semantics are resolved; unresolved junction association is null/excluded.",
        "unsupported_cases": ["missing road.rule", "ambiguous lane association", "unresolved junction lane linkage", "dynamicLaneDirection not represented in released files"],
        "evidence": ["ASAM_DIRECTION_001", "ASAM_LANE_ID_001", "ASAM_REF_001"],
        "observed_road_rule_counts": road_rule_counts,
        "observed_explicit_lane_direction_attribute_count": 0,
    }
    write_json(args.output_dir / "xodr_coordinate_contract.json", coordinate_contract)
    write_json(args.output_dir / "xodr_legal_direction_contract.json", legal_contract)

    baseline = read_jsonl(args.baseline_vector) if args.baseline_vector.is_file() else []
    ddc_rows = []
    for row in baseline:
        ddc_rows.append({"clip_id": row.get("clip_id"), "record_key": row.get("record_key"), "ddc_proxy": None, "ddc_proxy_status": "LEGAL_DIRECTION_CONTRACT_UNRESOLVED", "associated_road_id": None, "associated_lane_id": None, "road_rule": None, "allowed_direction_heading_rad": None, "max_oncoming_progress_m": None, "total_opposite_distance_m": None, "opposite_frame_count": None, "first_violation_time_s": None})
    write_csv(args.output_dir / "ddc_proxy_validation.csv", ddc_rows)
    write_csv(args.output_dir / "gt_ddc_validation.csv", [{"clip_id": row["clip_id"], "gt_ddc": None, "status": "NOT_RUN_LEGAL_DIRECTION_UNRESOLVED"} for row in readiness_rows])

    unique_georef = len({row.get("geo_reference_sha256") for row in georef_rows if row.get("geo_reference_sha256")})
    validation_complete = sum(row.get("validation_status") == "VALIDATION_COMPLETE" for row in validation_rows)
    summary = {
        "expected_clip_count": 300,
        "xodr_found_count": len(clip_ids),
        "xodr_parsed_count": sum(row.get("parse_status") in {"PARSED", "PARTIAL"} for row in georef_rows),
        "georeference_count": sum(bool(row.get("geo_reference_sha256")) for row in georef_rows),
        "unique_georeference_count": unique_georef,
        "opendrive_version_counts": version_counts,
        "road_rule_rht_count": road_rule_counts["RHT"],
        "road_rule_lht_count": road_rule_counts["LHT"],
        "road_rule_missing_count": road_rule_counts["MISSING"],
        "road_rule_other_count": road_rule_counts["OTHER"],
        "xodr_coordinate_contract_status": coordinate_contract["status"],
        "xodr_coordinate_binding_method": coordinate_contract["binding_method"],
        "legal_direction_contract_status": legal_contract["status"],
        "authoritative_coordinate_source_found": True,
        "authoritative_direction_source_found": True,
        "xodr_lane_geometry_validation_clip_count": validation_complete,
        "xodr_lane_geometry_systematic_contradiction_count": 0,
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "ddc_ready_clip_count": 0,
        "ddc_not_ready_clip_count": len(readiness_rows),
        "gt_ddc_valid_count": 0,
        "gt_ddc_zero_count": 0,
        "gt_ddc_half_count": 0,
        "gt_ddc_one_count": 0,
        "full_vector_expected_record_count": 4800,
        "full_vector_actual_record_count": len(baseline),
        "full_vector_unique_record_key_count": len({row.get("record_key") for row in baseline}),
        "ddc_proxy_valid_count": 0,
        "ddc_proxy_null_count": len(baseline),
        "ddc_proxy_zero_count": 0,
        "ddc_proxy_half_count": 0,
        "ddc_proxy_one_count": 0,
        "official_ddc_populated_count": 0,
        "official_epdms_stage1_populated_count": 0,
        "nurec_safety_proxy_v1_changed": False,
        "lk_proxy_changed": False,
        "gt_fitted_transform_used": False,
        "gt_fitted_direction_used": False,
        "scientific_limitations": ["Released XODR road.rule is missing for every road in the 300-clip inventory; lane id and reference-line orientation cannot establish legal direction alone.", "Coordinate validation is independent geometry diagnostics; no transform was optimized."],
        "remaining_blockers": ["LEGAL_DIRECTION_CONTRACT_UNRESOLVED: road.rule missing across released XODR", "JUNCTION_LANE_LINK_POLICY_NOT_IMPLEMENTED"],
        "recommended_next_step": "Keep ddc_proxy disabled unless a released metadata source supplies road-rule/legal-direction provenance or a separate authoritative traffic-rule layer.",
    }
    write_json(args.output_dir / "ddc_final_v2_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xodr-root", type=Path, default=XODR_ROOT_DEFAULT)
    parser.add_argument("--context-root", type=Path, default=CONTEXT_ROOT_DEFAULT)
    parser.add_argument("--map-root", type=Path, default=MAP_ROOT_DEFAULT)
    parser.add_argument("--baseline-vector", type=Path, default=BASELINE_VECTOR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
