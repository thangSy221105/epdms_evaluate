#!/usr/bin/env python3
"""Inventory and selectively recover OpenDRIVE members from NuRec USDZ files.

This module deliberately stops at evidence collection.  It does not infer a
NuRec coordinate transform, lane legality, or a DDC score from a missing or
ambiguous OpenDRIVE contract.  USDZ archives are inspected through the
existing HTTP Range/ZIP-central-directory implementation and only an exact
``map.xodr`` member is fetched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.acquire_full300_data as acquire


REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
RELEASE = "26.04_release"
XODR_BASENAME = "map.xodr"
EXPECTED_CLIP_COUNT = 300


def _auth_headers_fallback() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _patch_auth() -> None:
    try:
        acquire.auth_headers()
    except ModuleNotFoundError:
        acquire.auth_headers = _auth_headers_fallback


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _archive_url(clip_id: str) -> tuple[str, str]:
    archive = f"sample_set/{RELEASE}/{clip_id}/{clip_id}.usdz"
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{archive}"
    return archive, url


def _find_xodr(members: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    exact = members.get(XODR_BASENAME)
    if exact is not None:
        return XODR_BASENAME, exact
    candidates = [
        (name, entry)
        for name, entry in members.items()
        if Path(name).name.lower() == XODR_BASENAME
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _scan_one(clip_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    archive, url = _archive_url(clip_id)
    row: dict[str, Any] = {
        "clip_id": clip_id,
        "archive": archive,
        "archive_url": url,
        "scan_status": "UNKNOWN",
        "archive_size_bytes": None,
        "member_count": None,
        "xodr_member_found": False,
        "xodr_member_path": "",
        "xodr_compressed_bytes": None,
        "xodr_uncompressed_bytes": None,
        "error": "",
    }
    download = {
        "clip_id": clip_id,
        "archive": archive,
        "member": XODR_BASENAME,
        "status": "NOT_SCANNED",
        "path": "",
        "downloaded_bytes": 0,
        "integrity_status": "NOT_FETCHED",
        "error": "",
    }
    try:
        archive_size, _ = acquire.discover_usdz_size(url)
        members = acquire.remote_central_directory(url, archive_size)
        found = _find_xodr(members)
        row.update({
            "scan_status": "SCANNED",
            "archive_size_bytes": archive_size,
            "member_count": len(members),
            "xodr_member_found": found is not None,
            "xodr_member_path": found[0] if found else "",
            "xodr_compressed_bytes": int(found[1]["compressed_size"]) if found else None,
            "xodr_uncompressed_bytes": int(found[1]["uncompressed_size"]) if found else None,
        })
        download.update({
            "member": found[0] if found else XODR_BASENAME,
            "status": "PRESENT_IN_CENTRAL_DIRECTORY" if found else "MEMBER_NOT_FOUND",
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        row.update({"scan_status": "SCAN_FAILED", "error": error})
        download.update({"status": "SCAN_FAILED", "error": error})
    return row, download


def _fetch_xodr(scan: dict[str, Any], destination_root: Path) -> dict[str, Any]:
    result = dict(scan)
    clip_id = str(scan["clip_id"])
    destination = destination_root / clip_id / XODR_BASENAME
    result.update({"path": str(destination), "downloaded_bytes": 0, "integrity_status": "NOT_FETCHED"})
    if scan.get("scan_status") != "SCANNED" or not scan.get("xodr_member_found"):
        result["status"] = "MEMBER_NOT_FOUND" if scan.get("scan_status") == "SCANNED" else "SCAN_FAILED"
        return result
    if destination.is_file():
        result.update({"status": "ALREADY_PRESENT", "integrity_status": "READABLE", "downloaded_bytes": destination.stat().st_size})
        return result
    try:
        archive_size, _ = acquire.discover_usdz_size(str(scan["archive_url"]))
        members = acquire.remote_central_directory(str(scan["archive_url"]), archive_size)
        member_name = str(scan["xodr_member_path"])
        member = members.get(member_name)
        if member is None:
            found = _find_xodr(members)
            member = found[1] if found else None
        if member is None:
            result.update({"status": "MEMBER_DISAPPEARED", "integrity_status": "FAILED"})
            return result
        raw = acquire.fetch_member(str(scan["archive_url"]), archive_size, member)
        if not raw.lstrip().startswith(b"<"):
            result.update({"status": "NOT_XML", "integrity_status": "INVALID", "downloaded_bytes": len(raw)})
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".xodr.part")
        partial.write_bytes(raw)
        partial.replace(destination)
        result.update({"status": "DOWNLOADED", "integrity_status": "VALID_XML_CANDIDATE", "downloaded_bytes": len(raw)})
    except Exception as exc:
        result.update({"status": "DOWNLOAD_FAILED", "integrity_status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
    return result


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xodr(path: Path, clip_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "clip_id": clip_id,
        "path": str(path),
        "parse_status": "MISSING",
        "road_count": 0,
        "junction_count": 0,
        "geometry_count": 0,
        "geometry_types": "",
        "lane_direction_attr_count": 0,
        "lane_direction_values": "",
        "road_rule_values": "",
        "lane_id_positive_count": 0,
        "lane_id_negative_count": 0,
        "geo_reference_present": False,
        "geo_reference_sha256": "",
        "parse_error": "",
    }
    if not path.is_file():
        return row
    try:
        import hashlib

        raw = path.read_bytes()
        root = ET.fromstring(raw)
        roads = [element for element in root.iter() if _local_name(element.tag) == "road"]
        junctions = [element for element in root.iter() if _local_name(element.tag) == "junction"]
        geometries = [element for element in root.iter() if _local_name(element.tag) in {"line", "arc", "spiral", "poly3", "paramPoly3"}]
        types = sorted({_local_name(element.tag) for element in geometries})
        directions = [str(element.attrib["direction"]) for element in root.iter() if _local_name(element.tag) == "lane" and "direction" in element.attrib]
        rules = [str(element.attrib["rule"]) for element in root.iter() if _local_name(element.tag) == "road" and "rule" in element.attrib]
        lane_ids = []
        for element in root.iter():
            if _local_name(element.tag) == "lane" and "id" in element.attrib:
                try:
                    lane_ids.append(int(element.attrib["id"]))
                except ValueError:
                    pass
        geo = next((element for element in root.iter() if _local_name(element.tag) == "geoReference"), None)
        row.update({
            "parse_status": "PARSED",
            "road_count": len(roads),
            "junction_count": len(junctions),
            "geometry_count": len(geometries),
            "geometry_types": ";".join(types),
            "lane_direction_attr_count": len(directions),
            "lane_direction_values": ";".join(sorted(set(directions))),
            "road_rule_values": ";".join(sorted(set(rules))),
            "lane_id_positive_count": sum(value > 0 for value in lane_ids),
            "lane_id_negative_count": sum(value < 0 for value in lane_ids),
            "geo_reference_present": geo is not None,
            "geo_reference_sha256": hashlib.sha256((geo.text or "").encode("utf-8")).hexdigest() if geo is not None else "",
        })
    except Exception as exc:
        row.update({"parse_status": "PARSE_FAILED", "parse_error": f"{type(exc).__name__}: {exc}"})
    return row


def _direction_evidence(schema: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in schema if row.get("parse_status") == "PARSED"]
    explicit = sum(int(row.get("lane_direction_attr_count") or 0) for row in parsed)
    geo = sum(1 for row in parsed if row.get("geo_reference_present"))
    return {
        "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
        "authority": "ASAM_OPENDRIVE_LANE_DIRECTION_SEMANTICS",
        "parsed_clip_count": len(parsed),
        "explicit_lane_direction_attribute_count": explicit,
        "geo_reference_clip_count": geo,
        "legal_direction_status": "EVIDENCE_PRESENT_BUT_NOT_YET_BOUND_TO_NUREC" if explicit or parsed else "UNRESOLVED",
        "xodr_to_nurec_coordinate_status": "UNRESOLVED",
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "fail_closed_reason": "XODR_DIRECTION_AND_COORDINATE_CONTRACT_NOT_VERIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xodr-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    _patch_auth()
    clip_ids = sorted({str(row.get("clip_id", "")).strip() for row in _read_jsonl(args.time_mapping) if row.get("clip_id")})
    if len(clip_ids) != EXPECTED_CLIP_COUNT:
        raise SystemExit(f"expected {EXPECTED_CLIP_COUNT} clip IDs, got {len(clip_ids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.xodr_root.mkdir(parents=True, exist_ok=True)
    scan_rows: list[dict[str, Any]] = []
    download_rows: list[dict[str, Any]] = []
    # Keep the default single-worker mode predictable and gentle for HF.
    if args.workers <= 1:
        for index, clip_id in enumerate(clip_ids, 1):
            row, download = _scan_one(clip_id)
            scan_rows.append(row)
            download_rows.append(download)
            if index % 25 == 0 or index == len(clip_ids):
                print(f"scanned {index}/{len(clip_ids)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_scan_one, clip_id): clip_id for clip_id in clip_ids}
            for future in as_completed(futures):
                row, download = future.result()
                scan_rows.append(row)
                download_rows.append(download)
    scan_rows.sort(key=lambda row: row["clip_id"])
    download_rows.sort(key=lambda row: row["clip_id"])
    downloaded: list[dict[str, Any]] = []
    for scan in scan_rows:
        if scan.get("xodr_member_found"):
            downloaded.append(_fetch_xodr(scan, args.xodr_root))
        else:
            downloaded.append({**scan, "status": "MEMBER_NOT_FOUND", "path": "", "downloaded_bytes": 0, "integrity_status": "NOT_FETCHED"})
    schema_rows = [_parse_xodr(args.xodr_root / row["clip_id"] / XODR_BASENAME, row["clip_id"]) for row in scan_rows]
    evidence = _direction_evidence(schema_rows)
    summary = {
        "clip_count": len(clip_ids),
        "scanned_count": sum(row.get("scan_status") == "SCANNED" for row in scan_rows),
        "scan_failed_count": sum(row.get("scan_status") == "SCAN_FAILED" for row in scan_rows),
        "xodr_member_found_count": sum(bool(row.get("xodr_member_found")) for row in scan_rows),
        "xodr_downloaded_count": sum(row.get("status") in {"DOWNLOADED", "ALREADY_PRESENT"} for row in downloaded),
        "xodr_parsed_count": sum(row.get("parse_status") == "PARSED" for row in schema_rows),
        "direction_evidence": evidence,
    }
    _write_csv(args.output_dir / "xodr_member_inventory_full300.csv", scan_rows)
    _write_csv(args.output_dir / "xodr_download_inventory.csv", downloaded)
    _write_csv(args.output_dir / "xodr_schema_inventory.csv", schema_rows)
    _write_csv(args.output_dir / "xodr_direction_semantics_evidence.csv", [{**evidence, "clip_id": "DATASET"}])
    _write_json(args.output_dir / "xodr_direction_contract.json", evidence)
    _write_json(args.output_dir / "xodr_inventory_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
