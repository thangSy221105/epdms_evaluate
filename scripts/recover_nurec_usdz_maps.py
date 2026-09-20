#!/usr/bin/env python3
"""Recover official NuRec ClipGT map members from USDZ ZIP containers.

The public NuRec tree exposes the released clip as a USDZ container.  This
script inspects only ZIP central directories and fetches the exact ClipGT map
members with HTTP Range requests.  It never downloads a complete USDZ and it
keeps the recovered staging tree separate from the existing context tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.acquire_full300_data as acquire


REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
RELEASE = "26.04_release"
MAP_MEMBERS = {
    "lane": "clipgt/lane.parquet",
    "intersection_area": "clipgt/intersection_area.parquet",
    "drivable_space": "clipgt/drivable_space.parquet",
}
COPY_MEMBERS = ("lane", "intersection_area", "drivable_space")


def _auth_headers_fallback() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _patch_auth() -> None:
    """Use the installed hub helper when available, otherwise cached login."""

    try:
        acquire.auth_headers()
    except ModuleNotFoundError:
        acquire.auth_headers = _auth_headers_fallback


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parquet_metadata(raw: bytes) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(BytesIO(raw))
        schema = metadata.schema_arrow
        return {
            "integrity_status": "VALID",
            "row_count": int(metadata.metadata.num_rows),
            "column_names": list(schema.names),
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
        }
    except Exception as exc:  # pragma: no cover - exercised by corrupt fixtures
        return {
            "integrity_status": "INVALID",
            "row_count": None,
            "column_names": [],
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
            "integrity_error": f"{type(exc).__name__}: {exc}",
        }


def _local_map_status(root: Path, clip_id: str) -> dict[str, Any]:
    clip_root = root / clip_id
    result: dict[str, Any] = {}
    for component, relative in MAP_MEMBERS.items():
        path = clip_root / relative
        result[f"{component}_local_exists"] = path.is_file()
        result[f"{component}_local_path"] = str(path) if path.is_file() else ""
    return result


def _prepare_isolated_map_root(source_root: Path, destination_root: Path, clip_ids: list[str]) -> None:
    """Copy only transform metadata and existing maps into the new staging root."""

    for clip_id in clip_ids:
        source_clip = source_root / clip_id
        destination_clip = destination_root / clip_id
        rig = source_clip / "rig_trajectories.json"
        if rig.is_file():
            destination_clip.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rig, destination_clip / rig.name)
        for component, relative in MAP_MEMBERS.items():
            source = source_clip / relative
            if source.is_file():
                destination = destination_clip / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


def _scan_one(clip_id: str, existing_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive = f"sample_set/{RELEASE}/{clip_id}/{clip_id}.usdz"
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{archive}"
    row: dict[str, Any] = {
        "clip_id": clip_id,
        "archive": archive,
        "archive_url": url,
        "scan_status": "UNKNOWN",
        "archive_size_bytes": None,
        "member_count": None,
    }
    downloads: list[dict[str, Any]] = []
    local = _local_map_status(existing_root, clip_id)
    row.update(local)
    try:
        archive_size, _headers = acquire.discover_usdz_size(url)
        members = acquire.remote_central_directory(url, archive_size)
        row.update({"scan_status": "SCANNED", "archive_size_bytes": archive_size, "member_count": len(members)})
        for component, member_path in MAP_MEMBERS.items():
            found = members.get(member_path)
            if found is None:
                found_pair = acquire.find_member(members, Path(member_path).name)
                found = found_pair[1] if found_pair else None
                actual_path = found_pair[0] if found_pair else ""
            else:
                actual_path = member_path
            row[f"{component}_member_exists"] = found is not None
            row[f"{component}_member_path"] = actual_path
            row[f"{component}_member_uncompressed_bytes"] = int(found["uncompressed_size"]) if found else None
            row[f"{component}_member_compressed_bytes"] = int(found["compressed_size"]) if found else None
            downloads.append({
                "clip_id": clip_id,
                "component": component,
                "archive": archive,
                "member": actual_path or member_path,
                "member_exists": found is not None,
                "archive_size_bytes": archive_size,
                "status": "PRESENT_IN_CENTRAL_DIRECTORY" if found else "MEMBER_NOT_FOUND",
                "downloaded_bytes": 0,
                "path": "",
                "integrity_status": "NOT_FETCHED",
                "error": "",
            })
    except Exception as exc:
        row.update({"scan_status": "SCAN_FAILED", "scan_error": f"{type(exc).__name__}: {exc}"})
        for component, member_path in MAP_MEMBERS.items():
            row.setdefault(f"{component}_member_exists", False)
            row.setdefault(f"{component}_member_path", member_path)
            row.setdefault(f"{component}_member_uncompressed_bytes", None)
            row.setdefault(f"{component}_member_compressed_bytes", None)
            downloads.append({
                "clip_id": clip_id,
                "component": component,
                "archive": archive,
                "member": member_path,
                "member_exists": False,
                "archive_size_bytes": None,
                "status": "SCAN_FAILED",
                "downloaded_bytes": 0,
                "path": "",
                "integrity_status": "NOT_FETCHED",
                "error": row["scan_error"],
            })
    return row, downloads


def _fetch_one(download: dict[str, Any], map_root: Path, scan_row: dict[str, Any]) -> dict[str, Any]:
    component = download["component"]
    clip_id = download["clip_id"]
    destination = map_root / clip_id / MAP_MEMBERS[component]
    download = dict(download)
    download["path"] = str(destination)
    if destination.is_file():
        try:
            metadata = _parquet_metadata(destination.read_bytes())
            download.update({"status": "ALREADY_PRESENT", **metadata})
            return download
        except OSError as exc:
            download.update({"status": "READ_FAILED", "error": f"{type(exc).__name__}: {exc}"})
            return download
    if not download.get("member_exists") or scan_row.get("scan_status") != "SCANNED":
        download["status"] = "MEMBER_NOT_FOUND" if scan_row.get("scan_status") == "SCANNED" else "SCAN_FAILED"
        return download
    try:
        url = scan_row["archive_url"]
        member_name = download["member"]
        # The central-directory lookup is repeated here to keep worker state
        # self-contained and avoid passing large member maps between threads.
        archive_size, _ = acquire.discover_usdz_size(url)
        members = acquire.remote_central_directory(url, archive_size)
        member = members.get(member_name)
        if member is None:
            pair = acquire.find_member(members, Path(member_name).name)
            member = pair[1] if pair else None
        if member is None:
            download["status"] = "MEMBER_DISAPPEARED"
            return download
        raw = acquire.fetch_member(url, archive_size, member)
        metadata = _parquet_metadata(raw)
        if metadata["integrity_status"] != "VALID":
            download.update({"status": "INTEGRITY_FAILED", "downloaded_bytes": len(raw), **metadata})
            return download
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(raw)
        partial.replace(destination)
        download.update({"status": "DOWNLOADED", "downloaded_bytes": len(raw), **metadata})
    except Exception as exc:
        download.update({"status": "DOWNLOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"})
    return download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-ids-file", type=Path, default=None)
    parser.add_argument("--time-mapping", type=Path, default=None, help="JSONL manifest from which the exact 300 clip IDs are read.")
    parser.add_argument("--source-map-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    _patch_auth()
    if args.clip_ids_file is not None:
        clip_ids = sorted({line.strip() for line in args.clip_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()})
    elif args.time_mapping is not None:
        clip_ids = sorted({str(row.get("clip_id", "")).strip() for row in _read_jsonl(args.time_mapping) if row.get("clip_id")})
    else:
        raise SystemExit("provide --clip-ids-file or --time-mapping")
    if len(clip_ids) != 300:
        raise SystemExit(f"expected 300 clip IDs, got {len(clip_ids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_isolated_map_root(args.source_map_root, args.map_root, clip_ids)

    scan_rows: list[dict[str, Any]] = []
    planned_downloads: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_scan_one, clip_id, args.map_root): clip_id for clip_id in clip_ids}
        for index, future in enumerate(as_completed(futures), 1):
            row, downloads = future.result()
            scan_rows.append(row)
            planned_downloads.extend(downloads)
            if index % 25 == 0 or index == len(futures):
                print(f"USDZ_SCAN_PROGRESS {index}/{len(futures)}", flush=True)
    scan_rows.sort(key=lambda row: row["clip_id"])
    planned_downloads.sort(key=lambda row: (row["clip_id"], row["component"]))
    _write_csv(args.output_dir / "usdz_map_member_inventory_full300.csv", scan_rows)

    scan_by_clip = {row["clip_id"]: row for row in scan_rows}
    fetched: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_fetch_one, item, args.map_root, scan_by_clip[item["clip_id"]]): item
            for item in planned_downloads
            if item["component"] in COPY_MEMBERS
        }
        for index, future in enumerate(as_completed(futures), 1):
            fetched.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"USDZ_MAP_FETCH_PROGRESS {index}/{len(futures)}", flush=True)
    fetched.sort(key=lambda row: (row["clip_id"], row["component"]))
    _write_csv(args.output_dir / "map_download_inventory.csv", fetched)

    component_counts: dict[str, dict[str, int]] = {}
    for component in MAP_MEMBERS:
        component_counts[component] = {}
        for row in fetched:
            if row["component"] == component:
                status = str(row.get("status", "UNKNOWN"))
                component_counts[component][status] = component_counts[component].get(status, 0) + 1
    summary = {
        "repository": REPO_ID,
        "release": RELEASE,
        "clip_count": len(clip_ids),
        "central_directory_scanned_count": sum(row.get("scan_status") == "SCANNED" for row in scan_rows),
        "central_directory_failed_count": sum(row.get("scan_status") != "SCANNED" for row in scan_rows),
        "member_presence_counts": {
            component: sum(bool(row.get(f"{component}_member_exists")) for row in scan_rows)
            for component in MAP_MEMBERS
        },
        "download_status_counts": component_counts,
        "http_range_only": True,
        "full_usdz_downloaded": False,
        "source_map_root": str(args.source_map_root),
        "isolated_map_root": str(args.map_root),
    }
    _write_json(args.output_dir / "usdz_map_recovery_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
