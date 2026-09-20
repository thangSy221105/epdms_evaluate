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
ASAM_ROADS = "https://simulation.pages.asam.net/opendrive-group/opendrive-antora-gen/ASAM_OpenDRIVE_Specification/v1.9.0/specification/10_roads/10_01_introduction.html"
ASAM_REF = "https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.9.0/specification/09_geometries/09_02_road_reference_line.html"
ALPASIM_CONVENTIONS = "https://github.com/NVlabs/alpasim/blob/main/CONTRIBUTING.md"
NCORE_CONVENTIONS = "https://nvidia.github.io/ncore/data/conventions.html"

XODR_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\00_raw\nurec_full300_xodr")
CONTEXT_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2")
MAP_ROOT_DEFAULT = Path(r"D:\300_clip_nurec\hf_probe\nurec_map_dac_final_v2\map_root")
BASELINE_VECTOR_DEFAULT = Path(r"D:\300_clip_nurec\hf_probe\epdms_partial_vector_lk_ddc_xodr_full4800_v1\partial_metric_vector_full4800.jsonl")

# The released inventory is OpenDRIVE 1.4.  In that version the normative
# road semantics say that an omitted ``road/@rule`` means RHT.  Keep this
# table explicit so an unknown future revision fails closed rather than
# silently inheriting a newer-version assumption.
DEFAULT_RHT_VERSIONS = {"1.4"}
SUPPORTED_DIRECTION_VERSIONS = {"1.4"}


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
    invalid_numeric_fields: list[str] = []
    for key in ("lat_0", "lon_0", "x_0", "y_0", "zone", "alt_0"):
        if key in values:
            try:
                parsed = float(values[key])
                if not math.isfinite(parsed):
                    raise ValueError("non-finite numeric value")
                numeric[key] = parsed
            except ValueError:
                invalid_numeric_fields.append(key)
                numeric[key] = None
    has_required_numeric = (
        "lat_0" in numeric and numeric.get("lat_0") is not None
        and "lon_0" in numeric and numeric.get("lon_0") is not None
    )
    # The current NVIDIA chain uses the geodetic origin to construct the
    # ECEF->ENU frame.  A missing altitude is the documented zero-origin
    # default; a present but invalid value is not silently accepted.
    alt_valid = "alt_0" not in invalid_numeric_fields
    supported_projection = values.get("proj") in {"tmerc", "utm", "longlat", "aeqd", "stere", "lcc"}
    if "alt_0" not in numeric and alt_valid:
        numeric["alt_0"] = 0.0
    parse_status = "PARSED"
    if not values.get("proj") or not has_required_numeric or not alt_valid:
        parse_status = "PARTIAL"
    if not supported_projection:
        parse_status = "UNSUPPORTED_PROJECTION" if values.get("proj") else "PARTIAL"
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
        "alt_0": numeric.get("alt_0"),
        "invalid_numeric_fields": invalid_numeric_fields,
        "supported_projection": supported_projection,
        "parse_status": parse_status,
    }


def effective_road_rule(version: str, road_rule: str | None) -> tuple[str | None, str]:
    """Return the version-scoped effective traffic rule and its provenance."""
    normalized = (road_rule or "").strip().upper()
    if normalized in {"RHT", "LHT"}:
        return normalized, "ROAD_ATTRIBUTE"
    if not normalized and version in DEFAULT_RHT_VERSIONS:
        return "RHT", "ASAM_DEFAULT"
    return None, "UNRESOLVED_UNSUPPORTED_VERSION_OR_VALUE"


def lane_direction_metadata(root: ET.Element, version: str) -> dict[str, Any]:
    direction_values: list[str] = []
    dynamic_values: list[str] = []
    for node in root.iter():
        if local_name(node.tag) == "lane" and "direction" in node.attrib:
            direction_values.append(str(node.attrib["direction"]))
        if "dynamicLaneDirection" in node.attrib:
            dynamic_values.append(str(node.attrib["dynamicLaneDirection"]))
    # direction/dynamicLaneDirection are newer than this released 1.4
    # inventory.  Record their absence without citing newer semantics as
    # evidence for the old file version.
    feature_supported = version not in {"1.4"}
    return {
        "lane_direction_attribute_count": len(direction_values),
        "lane_direction_values": sorted(set(direction_values)),
        "dynamic_lane_direction_attribute_count": len(dynamic_values),
        "dynamic_lane_direction_values": sorted(set(dynamic_values)),
        "lane_direction_override_supported_for_version": feature_supported,
        "lane_direction_override_status": (
            "NOT_PRESENT_IN_RELEASED_FILES" if not direction_values and not dynamic_values
            else "PRESENT_BUT_VERSION_SEMANTICS_REQUIRE_REVIEW"
        ),
    }


def direction_contract_for_root(root: ET.Element, version: str) -> dict[str, Any]:
    roads = [node for node in root.iter() if local_name(node.tag) == "road"]
    lane_meta = lane_direction_metadata(root, version)
    attribute_counts = {"RHT": 0, "LHT": 0, "MISSING": 0, "OTHER": 0}
    effective_counts = {"RHT": 0, "LHT": 0, "UNRESOLVED": 0}
    regular_unresolved = 0
    junction_road_count = 0
    for road in roads:
        raw_rule = str(road.attrib.get("rule", "")).strip().upper()
        attribute_key = raw_rule if raw_rule in {"RHT", "LHT"} else ("MISSING" if not raw_rule else "OTHER")
        attribute_counts[attribute_key] += 1
        effective, _source = effective_road_rule(version, raw_rule)
        if effective is None:
            effective_counts["UNRESOLVED"] += 1
        else:
            effective_counts[effective] += 1
        is_junction_road = road.attrib.get("junction", "-1") not in {"", "-1"}
        junction_road_count += int(is_junction_road)
        if not is_junction_road and effective is None:
            regular_unresolved += 1
    junctions = [node for node in root.iter() if local_name(node.tag) == "junction"]
    has_lane_topology = any(local_name(node.tag) in {"laneSection", "left", "right", "lane"} for node in root.iter())
    has_junction_topology = bool(junctions) or any(local_name(node.tag) in {"connection", "laneLink"} for node in root.iter())
    overrides_present = bool(lane_meta["lane_direction_attribute_count"] or lane_meta["dynamic_lane_direction_attribute_count"])
    if version not in SUPPORTED_DIRECTION_VERSIONS:
        status = "UNRESOLVED"
    elif overrides_present and not lane_meta["lane_direction_override_supported_for_version"]:
        status = "UNRESOLVED"
    elif not has_lane_topology or regular_unresolved:
        status = "UNRESOLVED"
    elif junctions or junction_road_count:
        status = "PARTIAL"
    else:
        status = "VERIFIED"
    blockers: list[str] = []
    if version not in SUPPORTED_DIRECTION_VERSIONS:
        blockers.append("UNSUPPORTED_OPENDRIVE_VERSION")
    if overrides_present and not lane_meta["lane_direction_override_supported_for_version"]:
        blockers.append("LANE_DIRECTION_OVERRIDE_NOT_SUPPORTED_BY_FILE_VERSION")
    if not has_lane_topology:
        blockers.append("LANE_TOPOLOGY_MISSING")
    if regular_unresolved:
        blockers.append("REGULAR_ROAD_EFFECTIVE_RULE_UNRESOLVED")
    if junctions or junction_road_count:
        blockers.append("JUNCTION_EXCLUDED_LANE_LINK_POLICY_UNRESOLVED")
    return {
        "road_rule_attribute_counts": attribute_counts,
        "effective_road_rule_counts": effective_counts,
        "road_rule_attribute_available": attribute_counts["RHT"] + attribute_counts["LHT"] > 0,
        "effective_road_rule_available": effective_counts["UNRESOLVED"] == 0,
        "effective_road_rule": "RHT" if effective_counts["RHT"] and not effective_counts["LHT"] else ("LHT" if effective_counts["LHT"] and not effective_counts["RHT"] else ("MIXED" if effective_counts["RHT"] and effective_counts["LHT"] else None)),
        "effective_road_rule_source": "ASAM_DEFAULT_OR_ROAD_ATTRIBUTE" if effective_counts["UNRESOLVED"] == 0 else "UNRESOLVED",
        "lane_topology_available": has_lane_topology,
        "junction_topology_available": has_junction_topology,
        "junction_road_count": junction_road_count,
        "regular_road_unresolved_count": regular_unresolved,
        "legal_direction_supported": status in {"VERIFIED", "PARTIAL"} and not regular_unresolved,
        "legal_direction_status": status,
        "blockers": blockers,
        **lane_meta,
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


def map_to_ncore_transform(t_ecef_enu: np.ndarray, world_base: np.ndarray) -> np.ndarray:
    """Reproduce NVIDIA's map->NuRec chain without any fitted correction."""
    t_ecef_enu = np.asarray(t_ecef_enu, dtype=float)
    world_base = np.asarray(world_base, dtype=float)
    if t_ecef_enu.shape != (4, 4) or world_base.shape != (4, 4):
        raise ValueError("transform inputs must be 4x4")
    if not np.isfinite(t_ecef_enu).all() or not np.isfinite(world_base).all():
        raise ValueError("transform inputs must be finite")
    return np.linalg.inv(t_ecef_enu @ world_base)


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
        "source_transform_available": False,
        "transform_implemented": False,
        "geometry_validated": False,
        "xodr_sample_count": 0,
        "lane_point_count": 0,
        "median_nearest_distance_m": None,
        "p95_nearest_distance_m": None,
        "max_nearest_distance_m": None,
        "systematic_contradiction": None,
        "error": "",
    }
    try:
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        world_base = np.asarray(rig["T_world_base"], dtype=float)
        if world_base.shape != (4, 4):
            raise ValueError("T_world_base is not 4x4")
        if not np.isfinite(world_base).all():
            raise ValueError("T_world_base contains non-finite values")
        if geo.get("parse_status") != "PARSED" or not geo.get("supported_projection"):
            raise ValueError("geoReference is not valid for the documented transform chain")
        result["source_transform_available"] = True
        samples = sample_reference_geometry(root)
        lane_points = load_lane_points(lane_path)
        result["xodr_sample_count"] = int(len(samples))
        result["lane_point_count"] = int(len(lane_points))
        if not len(samples) or not len(lane_points):
            raise ValueError("missing finite XODR or lane geometry")
        if not np.isfinite(samples).all() or not np.isfinite(lane_points).all():
            raise ValueError("non-finite XODR or lane geometry")
        transform_map_to_ncore = map_to_ncore_transform(
            ecef_to_enu(float(geo["lat_0"]), float(geo["lon_0"]), float(geo["alt_0"])),
            world_base,
        )
        if not np.isfinite(transform_map_to_ncore).all():
            raise ValueError("non-finite map-to-NuRec transform")
        result["transform_implemented"] = True
        transformed = (transform_map_to_ncore @ samples.T).T[:, :3]
        distances: list[float] = []
        for start in range(0, len(transformed), 256):
            chunk = transformed[start : start + 256]
            distances.extend(np.sqrt(((chunk[:, None, :2] - lane_points[None, :, :2]) ** 2).sum(axis=2)).min(axis=1).tolist())
        values = np.asarray(distances, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite nearest-geometry distance")
        result.update({
            # There is deliberately no post-hoc distance threshold here.  A
            # nearest-lane diagnostic is not a semantic proof that the two
            # producers serialize the same geometry.  The measured candidate
            # is consequently rejected by this audit gate, not silently
            # promoted to ACCEPT.
            "validation_status": "REJECT",
            "validation_decision_reason": "NO_PREDECLARED_GEOMETRY_ACCEPTANCE_RULE",
            "median_nearest_distance_m": float(np.median(values)),
            "p95_nearest_distance_m": float(np.percentile(values, 95)),
            "max_nearest_distance_m": float(np.max(values)),
            # No contradiction is *proven* by this diagnostic alone.  Keep
            # this null rather than claiming that large or small distances
            # establish semantic equivalence without a declared rule.
            "systematic_contradiction": None,
            "geometry_validated": False,
        })
    except Exception as exc:
        result["validation_status"] = "FAILED"
        result["systematic_contradiction"] = True
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
        {"evidence_id": "ASAM_ROAD_RULE_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_ROADS, "symbol": "10.1 Introduction to roads / road/@rule", "line_range_or_section": "§10.1; rule annotated as asam.net:xodr:1.4.0", "claim_type": "ROAD_RULE_DEFAULT", "claim": "The road rule is RHT or LHT; when road/@rule is missing, RHT is assumed.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "Applied only to the observed OpenDRIVE 1.4 inventory; unknown versions fail closed."},
        {"evidence_id": "ASAM_DIRECTION_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_LANES, "symbol": "11.2.1 Driving direction", "line_range_or_section": "§11.2.1", "claim_type": "ROAD_RULE", "claim": "RHT/LHT plus lane grouping/id determines default direction; lane direction can override it.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "The released files are 1.4; the road section evidence records that omitted rule defaults to RHT. Lane-direction semantics from newer versions are not applied to 1.4."},
        {"evidence_id": "ASAM_LANE_ID_001", "repository": "ASAM OpenDRIVE", "commit_sha": "PUBLISHED_SPEC", "file_path": ASAM_LANES, "symbol": "lane id and lane groups", "line_range_or_section": "§11.1 and §11.2", "claim_type": "LANE_ID", "claim": "Positive lane IDs are left of center and negative lane IDs are right of center; reference line runs in increasing s.", "supports_coordinate_binding": False, "supports_legal_direction": True, "confidence": "HIGH", "notes": "The lane sign is evaluated together with the effective version-scoped road rule; it is not used as an independent heading heuristic."},
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
    road_rule_attribute_counts = {"RHT": 0, "LHT": 0, "MISSING": 0, "OTHER": 0}
    effective_road_rule_counts = {"RHT": 0, "LHT": 0, "UNRESOLVED": 0}
    version_counts: dict[str, int] = {}
    source_transform_available_count = 0
    transform_implemented_count = 0
    geometry_validated_count = 0
    validation_failed_count = 0
    systematic_contradiction_count = 0
    legal_status_counts = {"VERIFIED": 0, "PARTIAL": 0, "UNRESOLVED": 0}
    observed_lane_direction_attribute_count = 0
    observed_dynamic_lane_direction_count = 0
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
            direction = direction_contract_for_root(root, version)
            for key, count in direction["road_rule_attribute_counts"].items():
                road_rule_attribute_counts[key] += count
            for key, count in direction["effective_road_rule_counts"].items():
                effective_road_rule_counts[key] += count
            legal_status_counts[direction["legal_direction_status"]] += 1
            observed_lane_direction_attribute_count += direction["lane_direction_attribute_count"]
            observed_dynamic_lane_direction_count += direction["dynamic_lane_direction_attribute_count"]
            rig_path = context_clip / "rig_trajectories.json"
            rig = json.loads(rig_path.read_text(encoding="utf-8"))
            world_base = np.asarray(rig.get("T_world_base"), dtype=float)
            source_transform_available = (
                geo.get("parse_status") == "PARSED"
                and bool(geo.get("supported_projection"))
                and world_base.shape == (4, 4)
                and np.isfinite(world_base).all()
            )
            if source_transform_available:
                source_transform_available_count += 1
            origin_candidates_rows = inspect_origin_metadata(context_clip)
            origin_rows.extend({"clip_id": clip_id, **item} for item in origin_candidates_rows)
            lane_path = args.map_root / clip_id / "clipgt" / "lane.parquet"
            validation = geometry_validation(root, geo, rig_path, lane_path)
            validation_rows.append({"clip_id": clip_id, **validation})
            if validation.get("transform_implemented"):
                transform_implemented_count += 1
            if validation.get("geometry_validated"):
                geometry_validated_count += 1
            if validation.get("validation_status") == "FAILED":
                validation_failed_count += 1
            if validation.get("systematic_contradiction") is True:
                systematic_contradiction_count += 1
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
                "source_transform_available": source_transform_available,
                "transform_implemented": bool(validation.get("transform_implemented")),
                "geometry_validated": bool(validation.get("geometry_validated")),
                "systematic_contradiction": validation.get("systematic_contradiction"),
                "coordinate_contract_status": "VERIFIED" if source_transform_available and validation.get("transform_implemented") and validation.get("geometry_validated") and validation.get("systematic_contradiction") is False else "PARTIAL" if source_transform_available and validation.get("transform_implemented") else "UNRESOLVED",
                "blocker": "" if source_transform_available and validation.get("transform_implemented") and validation.get("geometry_validated") and validation.get("systematic_contradiction") is False else "GEOMETRY_VALIDATION_NOT_ACCEPTED_OR_INCOMPLETE" if source_transform_available else "MISSING_VALID_GEOREFERENCE_OR_T_WORLD_BASE",
            })
            coordinate_status = "VERIFIED" if source_transform_available and validation.get("transform_implemented") and validation.get("geometry_validated") and validation.get("systematic_contradiction") is False else "PARTIAL" if source_transform_available and validation.get("transform_implemented") else "UNRESOLVED"
            blockers = list(direction["blockers"])
            if coordinate_status != "VERIFIED":
                blockers.append("XODR_COORDINATE_GEOMETRY_VALIDATION_NOT_ACCEPTED")
            ddc_ready = coordinate_status == "VERIFIED" and direction["legal_direction_status"] == "VERIFIED" and not blockers
            readiness_rows.append({
                "clip_id": clip_id,
                "road_rule_attribute_available": direction["road_rule_attribute_available"],
                "effective_road_rule_available": direction["effective_road_rule_available"],
                "effective_road_rule": direction["effective_road_rule"],
                "effective_road_rule_source": direction["effective_road_rule_source"],
                "lane_topology_available": direction["lane_topology_available"],
                "junction_topology_available": direction["junction_topology_available"],
                "legal_direction_supported": direction["legal_direction_supported"],
                "legal_direction_status": direction["legal_direction_status"],
                "coordinate_supported": coordinate_status == "VERIFIED",
                "coordinate_status": coordinate_status,
                "ddc_proxy_ready": ddc_ready,
                "blockers": ";".join(blockers),
                "lane_direction_attribute_count": direction["lane_direction_attribute_count"],
                "dynamic_lane_direction_attribute_count": direction["dynamic_lane_direction_attribute_count"],
                "lane_direction_override_status": direction["lane_direction_override_status"],
            })
        except Exception as exc:
            binding_rows.append({"clip_id": clip_id, "coordinate_contract_status": "UNRESOLVED", "blocker": f"{type(exc).__name__}: {exc}"})
            validation_rows.append({"clip_id": clip_id, "validation_status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            legal_status_counts["UNRESOLVED"] += 1
            readiness_rows.append({"clip_id": clip_id, "legal_direction_status": "UNRESOLVED", "coordinate_status": "UNRESOLVED", "ddc_proxy_ready": False, "blockers": f"{type(exc).__name__}: {exc}"})

    evidence = evidence_rows()
    write_csv(args.output_dir / "upstream_coordinate_direction_evidence.csv", evidence)
    write_csv(args.output_dir / "xodr_georeference_inventory_full300.csv", georef_rows)
    write_csv(args.output_dir / "opendrive_version_inventory.csv", version_rows)
    write_csv(args.output_dir / "xodr_coordinate_binding_full300.csv", binding_rows)
    write_csv(args.output_dir / "xodr_lane_geometry_validation.csv", validation_rows)
    write_csv(args.output_dir / "ddc_direction_readiness_full300.csv", readiness_rows)
    write_csv(args.output_dir / "metadata_origin_evidence_full300.csv", origin_rows)

    validation_attempted_count = len(validation_rows)
    validation_complete_count = sum(row.get("validation_status") in {"ACCEPT", "REJECT"} for row in validation_rows)
    validation_failed_count = sum(row.get("validation_status") == "FAILED" for row in validation_rows)
    coordinate_verified_count = sum(
        row.get("source_transform_available") is True
        and row.get("transform_implemented") is True
        and row.get("geometry_validated") is True
        and row.get("systematic_contradiction") is False
        for row in validation_rows
    )
    coordinate_status = (
        "VERIFIED"
        if coordinate_verified_count == len(clip_ids) and len(clip_ids) > 0
        else "PARTIAL"
        if source_transform_available_count == len(clip_ids) and transform_implemented_count == len(clip_ids)
        else "UNRESOLVED"
    )
    coordinate_contract = {
        "status": coordinate_status,
        "binding_method": "CRS_CONVERSION",
        "transform_type": "KNOWN_PER_CLIP_TRANSFORM",
        "formula": "T_map_ncore = inverse(T_ecef_enu @ T_world_base)",
        "xodr_frame": "OpenDRIVE local ENU/map coordinates defined by geoReference PROJ string",
        "ncore_frame": "NuRec/NCORE local world (frame-0 local reconstruction frame)",
        "authoritative_source": NUREC_DOC,
        "evidence": ["NUREC_DOC_COORD_001", "NUREC_SKILL_COORD_001"],
        "source_transform_available": source_transform_available_count == len(clip_ids),
        "transform_implemented": transform_implemented_count == len(clip_ids),
        "geometry_validated": coordinate_verified_count == len(clip_ids) and len(clip_ids) > 0,
        "systematic_contradiction": True if systematic_contradiction_count > 0 else None,
        "systematic_contradiction_status": "PROVEN" if systematic_contradiction_count > 0 else "NOT_PROVEN_BY_DIAGNOSTIC",
        "validation_attempted_count": validation_attempted_count,
        "validation_complete_count": validation_complete_count,
        "validation_failed_count": validation_failed_count,
        "systematic_contradiction_count": systematic_contradiction_count,
        "validation_status": "ACCEPT" if coordinate_status == "VERIFIED" else "REJECT",
        "validation_decision_reason": None if coordinate_status == "VERIFIED" else "NO_PREDECLARED_GEOMETRY_ACCEPTANCE_RULE",
        "geometry_validation_note": "Nearest-lane distances are measured diagnostics only; without a pre-declared semantic acceptance threshold they cannot independently promote the coordinate contract.",
        "gt_fitted_transform_used": False,
        "lane_fitted_transform_used": False,
    }
    legal_direction_status = (
        "UNRESOLVED"
        if legal_status_counts["UNRESOLVED"]
        else "PARTIAL"
        if legal_status_counts["PARTIAL"]
        else "VERIFIED"
    )
    legal_contract = {
        "status": legal_direction_status,
        "opendrive_versions": sorted(version_counts),
        "reference_line_semantics": "Reference line runs in increasing s but does not itself establish driving direction.",
        "lane_id_semantics": "Positive IDs are left and negative IDs right of center.",
        "road_rule_semantics": "For OpenDRIVE 1.4, RHT/LHT controls default direction and an omitted road/@rule defaults to RHT.",
        "rht_direction_rule": "right/negative lanes follow positive reference-line direction; left/positive lanes oppose it.",
        "lht_direction_rule": "left/positive lanes follow positive reference-line direction; right/negative lanes oppose it.",
        "junction_policy": "No DDC evaluation until connecting-road laneLink/contactPoint semantics are resolved; unresolved junction association is null/excluded.",
        "missing_rule_policy": "ASAM_DEFAULT_RHT_FOR_SUPPORTED_VERSION_1.4",
        "unsupported_cases": ["unsupported OpenDRIVE version", "ambiguous lane association", "unresolved junction lane linkage", "lane direction override if present but unsupported by file version"],
        "evidence": ["ASAM_ROAD_RULE_001", "ASAM_DIRECTION_001", "ASAM_LANE_ID_001", "ASAM_REF_001"],
        "observed_road_rule_attribute_counts": road_rule_attribute_counts,
        "observed_effective_road_rule_counts": effective_road_rule_counts,
        "observed_explicit_lane_direction_attribute_count": observed_lane_direction_attribute_count,
        "observed_dynamic_lane_direction_attribute_count": observed_dynamic_lane_direction_count,
        "lane_direction_override_policy": "NOT_PRESENT_IN_RELEASED_1.4_FILES; NEWER_VERSION_FEATURES_NOT_USED",
        "junction_status": "JUNCTION_EXCLUDED" if legal_direction_status == "PARTIAL" else "NOT_PRESENT_OR_RESOLVED",
    }
    write_json(args.output_dir / "xodr_coordinate_contract.json", coordinate_contract)
    write_json(args.output_dir / "xodr_legal_direction_contract.json", legal_contract)

    baseline = read_jsonl(args.baseline_vector) if args.baseline_vector.is_file() else []
    ddc_rows = []
    for row in baseline:
        ddc_rows.append({"clip_id": row.get("clip_id"), "record_key": row.get("record_key"), "ddc_proxy": None, "ddc_proxy_status": f"LEGAL_DIRECTION_CONTRACT_{legal_direction_status}", "associated_road_id": None, "associated_lane_id": None, "road_rule": None, "allowed_direction_heading_rad": None, "max_oncoming_progress_m": None, "total_opposite_distance_m": None, "opposite_frame_count": None, "first_violation_time_s": None})
    write_csv(args.output_dir / "ddc_proxy_validation.csv", ddc_rows)
    write_csv(args.output_dir / "gt_ddc_validation.csv", [{"clip_id": row["clip_id"], "gt_ddc": None, "status": "NOT_RUN_LEGAL_DIRECTION_UNRESOLVED"} for row in readiness_rows])

    unique_georef = len({row.get("geo_reference_sha256") for row in georef_rows if row.get("geo_reference_sha256")})
    expected_vector_count = 4800
    actual_vector_count = len(baseline)
    unique_record_keys = len({row.get("record_key") for row in baseline})
    official_ddc_populated_count = sum(row.get("ddc") is not None for row in baseline)
    missing_components_ddc_count = sum("ddc" in (row.get("missing_components") or []) for row in baseline)
    official_stage1_populated_count = sum(row.get("official_epdms_stage1") is not None for row in baseline)
    proxy_ddc_populated_count = sum(row.get("ddc_proxy") is not None for row in baseline)
    ddc_ready_count = sum(row.get("ddc_proxy_ready") is True for row in readiness_rows)
    remaining_blockers: list[str] = []
    if coordinate_contract["status"] != "VERIFIED":
        remaining_blockers.append("XODR_COORDINATE_CONTRACT_NOT_VERIFIED")
    if legal_contract["status"] != "VERIFIED":
        remaining_blockers.append("LEGAL_DIRECTION_CONTRACT_NOT_VERIFIED")
    if legal_contract["status"] == "PARTIAL":
        remaining_blockers.append("JUNCTION_LANE_LINK_POLICY_NOT_IMPLEMENTED")
    if legal_contract["status"] == "UNRESOLVED":
        remaining_blockers.append("EFFECTIVE_ROAD_RULE_OR_VERSION_UNRESOLVED")
    if actual_vector_count != expected_vector_count or unique_record_keys != expected_vector_count:
        remaining_blockers.append("BASELINE_VECTOR_IDENTITY_INVALID")
    summary = {
        "expected_clip_count": len(clip_ids),
        "xodr_found_count": len(clip_ids),
        "xodr_parsed_count": sum(row.get("parse_status") in {"PARSED", "PARTIAL"} for row in georef_rows),
        "georeference_count": sum(bool(row.get("geo_reference_sha256")) for row in georef_rows),
        "unique_georeference_count": unique_georef,
        "opendrive_version_counts": version_counts,
        "road_rule_attribute_rht_count": road_rule_attribute_counts["RHT"],
        "road_rule_attribute_lht_count": road_rule_attribute_counts["LHT"],
        "road_rule_attribute_missing_count": road_rule_attribute_counts["MISSING"],
        "road_rule_attribute_other_count": road_rule_attribute_counts["OTHER"],
        "effective_rht_count": effective_road_rule_counts["RHT"],
        "effective_lht_count": effective_road_rule_counts["LHT"],
        "effective_rule_unresolved_count": effective_road_rule_counts["UNRESOLVED"],
        "xodr_coordinate_contract_status": coordinate_contract["status"],
        "xodr_coordinate_binding_method": coordinate_contract["binding_method"],
        "legal_direction_contract_status": legal_contract["status"],
        "authoritative_coordinate_source_found": True,
        "authoritative_direction_source_found": True,
        "coordinate_validation_attempted_count": validation_attempted_count,
        "coordinate_validation_complete_count": validation_complete_count,
        "coordinate_validation_failed_count": validation_failed_count,
        "coordinate_systematic_contradiction_count": systematic_contradiction_count,
        "xodr_lane_geometry_validation_clip_count": validation_complete_count,
        "xodr_lane_geometry_systematic_contradiction_count": systematic_contradiction_count,
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "ddc_ready_clip_count": ddc_ready_count,
        "ddc_not_ready_clip_count": len(readiness_rows) - ddc_ready_count,
        "gt_ddc_valid_count": sum(row.get("gt_ddc") is not None for row in baseline),
        "gt_ddc_zero_count": sum(row.get("gt_ddc") == 0 for row in baseline),
        "gt_ddc_half_count": sum(row.get("gt_ddc") == 0.5 for row in baseline),
        "gt_ddc_one_count": sum(row.get("gt_ddc") == 1 for row in baseline),
        "full_vector_expected_record_count": expected_vector_count,
        "full_vector_actual_record_count": actual_vector_count,
        "full_vector_unique_record_key_count": unique_record_keys,
        "full_vector_identity_status": "VERIFIED" if actual_vector_count == expected_vector_count and unique_record_keys == expected_vector_count else "INVALID",
        "ddc_proxy_valid_count": proxy_ddc_populated_count,
        "ddc_proxy_null_count": sum(row.get("ddc_proxy") is None for row in baseline),
        "ddc_proxy_zero_count": sum(row.get("ddc_proxy") == 0 for row in baseline),
        "ddc_proxy_half_count": sum(row.get("ddc_proxy") == 0.5 for row in baseline),
        "ddc_proxy_one_count": sum(row.get("ddc_proxy") == 1 for row in baseline),
        "official_ddc_populated_count": official_ddc_populated_count,
        "missing_components_ddc_count": missing_components_ddc_count,
        "official_epdms_stage1_populated_count": official_stage1_populated_count,
        "lk_mutation_status": "NOT_MUTATED_BY_THIS_AUDIT_SCRIPT",
        "nurec_safety_proxy_v1_mutation_status": "NOT_MUTATED_BY_THIS_AUDIT_SCRIPT",
        "gt_fitted_transform_used": False,
        "lane_fitted_transform_used": False,
        "heuristic_direction_used": False,
        "scientific_limitations": ["OpenDRIVE 1.4 missing road/@rule is interpreted as ASAM default RHT for regular roads; junction lane-link/contact-point semantics remain excluded.", "Nearest-lane distances are measured diagnostics. No distance threshold or fitted transform was introduced, so they do not by themselves prove semantic coordinate equivalence."],
        "remaining_blockers": remaining_blockers,
        "recommended_next_step": "Keep ddc_proxy disabled until the coordinate geometry validation has a pre-declared acceptance contract and junction legal-direction semantics are resolved.",
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
