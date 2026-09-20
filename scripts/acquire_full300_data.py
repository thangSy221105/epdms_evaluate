"""Acquire minimal PAI/NuRec members for the full-300 readiness pipeline.

The official PAI repository stores egomotion.offline as chunked ZIP archives
and the official NuRec repository stores sequence_tracks.json inside USDZ ZIP
containers. This script uses HTTP Range requests to inspect ZIP central
directories and fetch only the required members. It never downloads a full
NuRec USDZ unless a caller explicitly replaces this component-level workflow.

All joins are exact UUID joins. Existing valid local files are reused.
Downloaded data is written outside git under D:\\300_clip_nurec by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CLIP_COUNT = 300
PAI_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
NUREC_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
NUREC_RELEASE = "26.04_release"
UUID_RE = re.compile(
    r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
REQUIRED_EGO_COLUMNS = {"timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_url(repo_id: str, path: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id, path, repo_type="dataset")


def auth_headers() -> dict[str, str]:
    from huggingface_hub import get_token

    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def range_request(url: str, start: int, length: int, retries: int = 4) -> tuple[bytes, dict[str, str]]:
    if length <= 0:
        return b"", {}
    headers = {"Range": f"bytes={start}-{start + length - 1}", **auth_headers()}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
                if response.status not in (200, 206):
                    raise RuntimeError(f"HTTP_STATUS_{response.status}")
                if response.status == 200 and len(payload) < length:
                    raise RuntimeError("HTTP_RANGE_RESPONSE_TRUNCATED")
                return payload, {key.lower(): value for key, value in response.headers.items()}
        except (OSError, urllib.error.HTTPError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"RANGE_REQUEST_FAILED:{url}:{start}:{length}:{last_error}") from last_error


def local_central_directory(path: Path) -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename: {
                "filename": info.filename,
                "compress_type": info.compress_type,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "local_header_offset": info.header_offset,
                "crc32": info.CRC,
            }
            for info in archive.infolist()
        }


def parse_central_directory(data: bytes) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cursor = 0
    header_size = struct.calcsize("<4s6H3L5H2L")
    while cursor + 4 <= len(data):
        if data[cursor : cursor + 4] != b"PK\x01\x02":
            break
        if cursor + header_size > len(data):
            raise ValueError("ZIP_CENTRAL_DIRECTORY_TRUNCATED")
        unpacked = struct.unpack_from("<4s6H3L5H2L", data, cursor)
        (
            _signature,
            _version_made,
            _version_needed,
            flags,
            compress_type,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            _disk_number,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = unpacked
        start = cursor + header_size
        end = start + name_length
        if end + extra_length + comment_length > len(data):
            raise ValueError("ZIP_CENTRAL_DIRECTORY_ENTRY_TRUNCATED")
        name_bytes = data[start:end]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        filename = name_bytes.decode(encoding, errors="strict")
        extra = data[end : end + extra_length]
        zip64_values: list[int] = []
        extra_cursor = 0
        while extra_cursor + 4 <= len(extra):
            extra_id, extra_size = struct.unpack_from("<HH", extra, extra_cursor)
            extra_start = extra_cursor + 4
            extra_end = extra_start + extra_size
            if extra_end > len(extra):
                raise ValueError("ZIP_EXTRA_FIELD_TRUNCATED")
            if extra_id == 0x0001:
                zip64_values.extend(
                    struct.unpack_from("<" + "Q" * ((extra_size) // 8), extra, extra_start)
                    if extra_size >= 8
                    else ()
                )
            extra_cursor = extra_end
        zip64_index = 0
        if uncompressed_size == 0xFFFFFFFF:
            if zip64_index >= len(zip64_values):
                raise ValueError("ZIP64_UNCOMPRESSED_SIZE_MISSING")
            uncompressed_size = zip64_values[zip64_index]
            zip64_index += 1
        if compressed_size == 0xFFFFFFFF:
            if zip64_index >= len(zip64_values):
                raise ValueError("ZIP64_COMPRESSED_SIZE_MISSING")
            compressed_size = zip64_values[zip64_index]
            zip64_index += 1
        if local_header_offset == 0xFFFFFFFF:
            if zip64_index >= len(zip64_values):
                raise ValueError("ZIP64_LOCAL_OFFSET_MISSING")
            local_header_offset = zip64_values[zip64_index]
            zip64_index += 1
        result[filename] = {
            "filename": filename,
            "compress_type": int(compress_type),
            "compressed_size": int(compressed_size),
            "uncompressed_size": int(uncompressed_size),
            "local_header_offset": int(local_header_offset),
            "crc32": int(crc32),
        }
        cursor = end + extra_length + comment_length
    if not result:
        raise ValueError("ZIP_CENTRAL_DIRECTORY_EMPTY_OR_UNREADABLE")
    return result


def remote_central_directory(url: str, size: int) -> dict[str, dict[str, Any]]:
    tail_size = min(size, 128 * 1024)
    tail, _headers = range_request(url, size - tail_size, tail_size)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError("ZIP_EOCD_NOT_FOUND")
    _signature, _disk, _cd_disk, _disk_entries, entries, cd_size, cd_offset, _comment_length = struct.unpack_from(
        "<4s4H2LH", tail, eocd
    )
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        zip64 = tail.rfind(b"PK\x06\x06", 0, eocd)
        if zip64 < 0 or zip64 + 56 > len(tail):
            raise ValueError("ZIP64_EOCD_NOT_FOUND")
        (
            _signature64,
            _record_size,
            _version_made64,
            _version_needed64,
            _disk64,
            _cd_disk64,
            _disk_entries64,
            entries64,
            cd_size64,
            cd_offset64,
        ) = struct.unpack_from("<4sQ2H2I4Q", tail, zip64)
        entries, cd_size, cd_offset = entries64, cd_size64, cd_offset64
    tail_start = size - tail_size
    if tail_start <= cd_offset and cd_offset + cd_size <= size:
        central_start = cd_offset - tail_start
        central = tail[central_start : central_start + cd_size]
    else:
        central, _headers = range_request(url, cd_offset, cd_size)
    parsed = parse_central_directory(central)
    if len(parsed) != entries:
        raise ValueError(f"ZIP_ENTRY_COUNT_MISMATCH:{len(parsed)}:{entries}")
    return parsed


def fetch_member(url: str, archive_size: int, member: dict[str, Any]) -> bytes:
    local_offset = int(member["local_header_offset"])
    local_header, _headers = range_request(url, local_offset, 4096)
    if local_header[:4] != b"PK\x03\x04":
        raise ValueError("ZIP_LOCAL_HEADER_INVALID")
    _signature, _version, _flags, method, _mtime, _mdate, _crc, _compressed, _uncompressed, name_len, extra_len = struct.unpack_from(
        "<4s5H3L2H", local_header, 0
    )
    data_start = local_offset + 30 + name_len + extra_len
    compressed_size = int(member["compressed_size"])
    if data_start + compressed_size > archive_size:
        raise ValueError("ZIP_MEMBER_OUT_OF_RANGE")
    compressed, _headers = range_request(url, data_start, compressed_size)
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"ZIP_COMPRESSION_UNSUPPORTED:{method}")
    if len(raw) != int(member["uncompressed_size"]):
        raise ValueError("ZIP_MEMBER_SIZE_MISMATCH")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != int(member["crc32"]):
        raise ValueError("ZIP_MEMBER_CRC_MISMATCH")
    return raw


def local_or_remote_member(
    archive_path: Path | None,
    remote_url: str | None,
    remote_size: int | None,
    member: dict[str, Any],
) -> bytes:
    if archive_path is not None:
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(member["filename"])
    if remote_url is None or remote_size is None:
        raise ValueError("ZIP_SOURCE_UNAVAILABLE")
    return fetch_member(remote_url, remote_size, member)


def validate_ego_bytes(raw: bytes) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(pa.BufferReader(raw))
    except Exception as exc:
        return {"schema_valid": False, "status": f"PARQUET_READ_ERROR:{type(exc).__name__}", "row_count": None}
    columns = set(table.column_names)
    missing = sorted(REQUIRED_EGO_COLUMNS - columns)
    if missing:
        return {"schema_valid": False, "status": f"MISSING_COLUMNS:{','.join(missing)}", "row_count": table.num_rows}
    rows = table.to_pylist()
    try:
        timestamps = [int(row["timestamp"]) for row in rows]
        nonnegative = [value for value in timestamps if value >= 0]
        quaternion_valid = all(
            all(math.isfinite(float(row[key])) for key in ("qx", "qy", "qz", "qw"))
            and math.sqrt(sum(float(row[key]) ** 2 for key in ("qx", "qy", "qz", "qw"))) > 0
            for row in rows
        )
    except (TypeError, ValueError, OverflowError):
        return {"schema_valid": False, "status": "NON_NUMERIC_OR_INVALID_POSE", "row_count": table.num_rows}
    valid = table.num_rows > 1 and len(nonnegative) >= 2 and quaternion_valid
    return {
        "schema_valid": valid,
        "status": "VALID" if valid else "INSUFFICIENT_VALID_POSE_ROWS",
        "row_count": table.num_rows,
        "timestamp_min_us": min(timestamps) if timestamps else None,
        "timestamp_max_us": max(timestamps) if timestamps else None,
        "nonnegative_timestamp_count": len(nonnegative),
        "columns": sorted(columns),
    }


def valid_existing_ego(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = validate_ego_bytes(path.read_bytes())
    if result.get("schema_valid"):
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), **result}
    return None


def discover_manifest(path: Path) -> list[str]:
    rows = read_jsonl(path)
    ids = sorted(str(row.get("clip_id") or "") for row in rows if row.get("clip_id"))
    if len(ids) != EXPECTED_CLIP_COUNT or len(set(ids)) != EXPECTED_CLIP_COUNT:
        raise ValueError(f"EXPECTED_EXACTLY_{EXPECTED_CLIP_COUNT}_UNIQUE_CLIP_IDS")
    return ids


def load_pai_archive_entries(repo_id: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    entries = list(HfApi().list_repo_tree(repo_id, path_in_repo="labels/egomotion.offline", repo_type="dataset", recursive=False))
    result = []
    for entry in entries:
        path = str(getattr(entry, "path", ""))
        if path.endswith(".zip"):
            result.append({"path": path, "size": int(getattr(entry, "size", 0) or 0), "url": hf_url(repo_id, path)})
    if not result:
        raise ValueError("PAI_EGOMOTION_ARCHIVE_LIST_EMPTY")
    return result


def local_archive_by_remote_name(local_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    directory = local_root / "labels" / "egomotion.offline"
    paths = directory.glob("*.zip") if directory.is_dir() else []
    for path in paths:
        match = re.search(r"chunk_(\d+)", path.name)
        if match:
            result[f"labels/egomotion.offline/egomotion.offline.chunk_{int(match.group(1)):04d}.zip"] = path
    return result


def scan_one_archive(entry: dict[str, Any], local_archive: Path | None) -> tuple[str, dict[str, dict[str, Any]], str | None]:
    try:
        members = local_central_directory(local_archive) if local_archive is not None else remote_central_directory(entry["url"], entry["size"])
        return entry["path"], members, None
    except Exception as exc:
        return entry["path"], {}, f"{type(exc).__name__}:{exc}"


def scan_pai_archives(
    entries: list[dict[str, Any]],
    local_by_name: dict[str, Path],
    workers: int,
    target_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    hits: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(scan_one_archive, entry, local_by_name.get(entry["path"])): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            path, members, error = future.result()
            completed += 1
            if error:
                errors.append({"archive": path, "status": "SCAN_FAILED", "error": error})
            for member_name, member in members.items():
                match = UUID_RE.search(Path(member_name).name)
                if match and Path(member_name).name.lower().endswith(".egomotion.offline.parquet") and match.group(1).lower() in target_ids:
                    clip_id = match.group(1).lower()
                    if clip_id not in hits:
                        hits[clip_id] = {
                            "clip_id": clip_id,
                            "archive": path,
                            "member": member_name,
                            "member_size_bytes": int(member["uncompressed_size"]),
                            "compressed_size_bytes": int(member["compressed_size"]),
                            "archive_size_bytes": int(entry["size"]),
                            "archive_url": entry["url"],
                            "archive_local_path": str(local_by_name[path]) if path in local_by_name else None,
                            "member_metadata": member,
                        }
            if completed % 100 == 0 or completed == len(entries):
                print(f"PAI_ARCHIVE_SCAN_PROGRESS {completed}/{len(entries)} hits={len(hits)}", flush=True)
    return hits, errors


def acquire_pai_clip(clip_id: str, source: dict[str, Any], target_root: Path) -> dict[str, Any]:
    destination = target_root / "physicalai_offline" / clip_id / "egomotion.offline.parquet"
    existing = valid_existing_ego(destination)
    if existing:
        return {
            "clip_id": clip_id,
            "source_url_or_repository": PAI_REPO,
            "archive": source["archive"],
            "member": source["member"],
            "size_bytes": existing["size_bytes"],
            "downloaded_bytes": 0,
            "sha256_if_practical": existing["sha256"],
            "schema_valid": True,
            "status": "ALREADY_PRESENT",
            "path": str(destination),
            **{key: value for key, value in existing.items() if key not in {"path", "size_bytes", "sha256", "status"}},
        }
    try:
        member = source["member_metadata"]
        archive_local = Path(source["archive_local_path"]) if source.get("archive_local_path") else None
        raw = local_or_remote_member(archive_local, source.get("archive_url"), int(source["archive_size_bytes"]), member)
        integrity = validate_ego_bytes(raw)
        if not integrity.get("schema_valid"):
            return {
                "clip_id": clip_id,
                "source_url_or_repository": PAI_REPO,
                "archive": source["archive"],
                "member": source["member"],
                "size_bytes": len(raw),
                "downloaded_bytes": int(source["compressed_size_bytes"]),
                "sha256_if_practical": hashlib.sha256(raw).hexdigest(),
                "schema_valid": False,
                "status": f"INTEGRITY_FAILED:{integrity.get('status')}",
                **integrity,
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".parquet.part")
        partial.write_bytes(raw)
        partial.replace(destination)
        return {
            **integrity,
            "clip_id": clip_id,
            "source_url_or_repository": PAI_REPO,
            "archive": source["archive"],
            "member": source["member"],
            "size_bytes": len(raw),
            "downloaded_bytes": int(source["compressed_size_bytes"]),
            "sha256_if_practical": hashlib.sha256(raw).hexdigest(),
            "schema_valid": True,
            "status": "DOWNLOADED",
            "path": str(destination),
        }
    except Exception as exc:
        return {
            "clip_id": clip_id,
            "source_url_or_repository": PAI_REPO,
            "archive": source.get("archive"),
            "member": source.get("member"),
            "size_bytes": None,
            "downloaded_bytes": 0,
            "sha256_if_practical": None,
            "schema_valid": False,
            "status": f"DOWNLOAD_FAILED:{type(exc).__name__}",
            "error": str(exc),
            "path": str(destination),
        }


def discover_usdz_size(url: str) -> tuple[int, dict[str, str]]:
    payload, headers = range_request(url, 0, 1)
    content_range = headers.get("content-range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1)), headers
    content_length = headers.get("content-length")
    if content_length and len(payload) == int(content_length):
        return int(content_length), headers
    raise ValueError("REMOTE_ARCHIVE_SIZE_UNAVAILABLE")


def find_member(members: dict[str, dict[str, Any]], basename: str) -> tuple[str, dict[str, Any]] | None:
    exact = [(name, value) for name, value in members.items() if Path(name).name == basename]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        root = [(name, value) for name, value in exact if "/" not in name.strip("/")]
        return root[0] if len(root) == 1 else None
    return None


def load_existing_sequence_rows(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {"read_status": "READ_ERROR", "schema_status": "ROOT_NOT_OBJECT"}
        while isinstance(value, dict) and set(value) == {"dummy_chunk_id"}:
            value = value["dummy_chunk_id"]
        tracks = value.get("tracks_data") if isinstance(value, dict) else None
        cuboids = value.get("cuboidtracks_data") if isinstance(value, dict) else None
        ids = tracks.get("tracks_id", []) if isinstance(tracks, dict) else []
        poses = tracks.get("tracks_poses", []) if isinstance(tracks, dict) else []
        raw_times = tracks.get("tracks_timestamps_us", []) if isinstance(tracks, dict) else []
        labels = tracks.get("tracks_label_class", []) if isinstance(tracks, dict) else []
        dims = cuboids.get("cuboids_dims", []) if isinstance(cuboids, dict) else []
        timestamps = [int(v) for track in raw_times for v in track if isinstance(v, (int, float))]
        schema_valid = (
            isinstance(tracks, dict)
            and isinstance(cuboids, dict)
            and len(ids) == len(poses) == len(raw_times) == len(labels) == len(dims)
            and len(ids) > 0
        )
        return {
            "read_status": "READ",
            "schema_status": "VALID" if schema_valid else "SCHEMA_UNRESOLVED",
            "row_count": len(timestamps) if timestamps else None,
            "timestamp_min_us": min(timestamps) if timestamps else None,
            "timestamp_max_us": max(timestamps) if timestamps else None,
        }
    except Exception as exc:
        return {"read_status": f"READ_ERROR:{type(exc).__name__}", "schema_status": "UNREADABLE"}


def reconcile_existing_output(output: Path, pai_before: int, sequence_before: int, other_downloaded_bytes: int = 0) -> dict[str, Any]:
    """Refresh inventories after a retry without redownloading valid members."""
    pai = read_csv(output / "pai_egomotion_download_inventory.csv")
    for row in pai:
        row["status"] = "DOWNLOADED" if int(row.get("downloaded_bytes") or 0) > 0 else "ALREADY_PRESENT"
    nurec = read_csv(output / "sequence_tracks_full300_download_inventory.csv")
    for row in nurec:
        path = Path(row["path"])
        if not path.is_file():
            continue
        integrity = load_existing_sequence_rows(path)
        if integrity.get("schema_status") != "VALID":
            row["read_status"] = integrity.get("read_status")
            row["schema_status"] = integrity.get("schema_status")
            continue
        row.update(integrity)
        row["size_bytes"] = path.stat().st_size
        if int(row.get("downloaded_bytes") or 0) == 0:
            row["downloaded_bytes"] = path.stat().st_size
        row["downloaded_this_round"] = "True"
        row["read_status"] = "DOWNLOADED"
        row["error"] = ""
    write_csv(output / "pai_egomotion_download_inventory.csv", pai)
    write_csv(output / "sequence_tracks_full300_download_inventory.csv", nurec)
    write_csv(
        output / "pai_egomotion_integrity.csv",
        [
            {
                "clip_id": row["clip_id"],
                "path": row.get("path"),
                "schema_valid": row.get("schema_valid"),
                "row_count": row.get("row_count"),
                "timestamp_min_us": row.get("timestamp_min_us"),
                "timestamp_max_us": row.get("timestamp_max_us"),
                "nonnegative_timestamp_count": row.get("nonnegative_timestamp_count"),
                "status": row.get("status"),
                "sha256": row.get("sha256_if_practical"),
            }
            for row in pai
        ],
    )
    write_csv(
        output / "sequence_tracks_integrity.csv",
        [
            {
                "clip_id": row["clip_id"],
                "path": row.get("path"),
                "read_status": row.get("read_status"),
                "schema_status": row.get("schema_status"),
                "row_count": row.get("row_count"),
                "timestamp_min_us": row.get("timestamp_min_us"),
                "timestamp_max_us": row.get("timestamp_max_us"),
                "size_bytes": row.get("size_bytes"),
            }
            for row in nurec
        ],
    )
    write_csv(
        output / "download_execution_log.csv",
        [
            {
                "clip_id": row["clip_id"],
                "component": "PAI_EGOMOTION_OFFLINE",
                "status": row.get("status"),
                "path": row.get("path"),
                "downloaded_bytes": row.get("downloaded_bytes", 0),
                "error": row.get("error"),
            }
            for row in pai
        ]
        + [
            {
                "clip_id": row["clip_id"],
                "component": "NUREC_SEQUENCE_TRACKS_MEMBER",
                "status": row.get("read_status"),
                "path": row.get("path"),
                "downloaded_bytes": row.get("downloaded_bytes", 0),
                "error": row.get("error"),
            }
            for row in nurec
        ],
    )
    pai_bytes = sum(int(row.get("downloaded_bytes") or 0) for row in pai)
    nurec_bytes = sum(int(row.get("downloaded_bytes") or 0) for row in nurec)
    summary = {
        "generated_at_utc": utc_now(),
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "pai_egomotion_before": pai_before,
        "pai_egomotion_after": sum(row.get("schema_valid", "").lower() == "true" for row in pai),
        "pai_egomotion_downloaded_count": sum(row.get("status") == "DOWNLOADED" for row in pai),
        "pai_egomotion_download_failed_count": sum(str(row.get("status", "")).startswith(("DOWNLOAD_FAILED", "INTEGRITY_FAILED")) for row in pai),
        "sequence_tracks_before": sequence_before,
        "sequence_tracks_after": sum(row.get("schema_status") == "VALID" for row in nurec),
        "sequence_tracks_downloaded_count": sum(row.get("read_status") == "DOWNLOADED" for row in nurec),
        "sequence_tracks_download_failed_count": sum(str(row.get("read_status", "")).startswith("DOWNLOAD_FAILED") for row in nurec),
        "pai_egomotion_downloaded_bytes": pai_bytes,
        "nurec_sequence_tracks_downloaded_bytes": nurec_bytes,
        "other_downloaded_bytes": other_downloaded_bytes,
        "total_downloaded_bytes": pai_bytes + nurec_bytes + other_downloaded_bytes,
        "full_clip_download_count": 0,
        "component_level_download": True,
        "full_nurec_usdz_downloaded": False,
        "raw_data_committed_to_git": False,
    }
    write_json(output / "full300_acquisition_summary.json", summary)
    return summary


def acquire_nurec_clip(clip_id: str, target_root: Path, repo_id: str) -> dict[str, Any]:
    destination = target_root / "nurec_full300" / clip_id / "sequence_tracks.json"
    if destination.is_file() and load_existing_sequence_rows(destination).get("schema_status") == "VALID":
        return {
            "clip_id": clip_id,
            "path": str(destination),
            "source": repo_id,
            "archive": f"sample_set/{NUREC_RELEASE}/{clip_id}/{clip_id}.usdz",
            "member": "sequence_tracks.json",
            "size_bytes": destination.stat().st_size,
            "downloaded_bytes": 0,
            "downloaded_this_round": False,
            "read_status": "ALREADY_PRESENT",
            "schema_status": "VALID",
        }
    archive = f"sample_set/{NUREC_RELEASE}/{clip_id}/{clip_id}.usdz"
    url = hf_url(repo_id, archive)
    try:
        archive_size, _headers = discover_usdz_size(url)
        members = remote_central_directory(url, archive_size)
        found = find_member(members, "sequence_tracks.json")
        if found is None:
            return {
                "clip_id": clip_id,
                "path": str(destination),
                "source": repo_id,
                "archive": archive,
                "member": None,
                "size_bytes": None,
                "downloaded_bytes": 0,
                "downloaded_this_round": False,
                "read_status": "NOT_FOUND_UPSTREAM",
                "schema_status": "MEMBER_NOT_FOUND",
            }
        member_name, member = found
        raw = fetch_member(url, archive_size, member)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("SEQUENCE_TRACKS_ROOT_NOT_OBJECT")
        while isinstance(value, dict) and set(value) == {"dummy_chunk_id"}:
            value = value["dummy_chunk_id"]
        tracks = value.get("tracks_data") if isinstance(value, dict) else None
        cuboids = value.get("cuboidtracks_data") if isinstance(value, dict) else None
        ids = tracks.get("tracks_id", []) if isinstance(tracks, dict) else []
        poses = tracks.get("tracks_poses", []) if isinstance(tracks, dict) else []
        raw_times = tracks.get("tracks_timestamps_us", []) if isinstance(tracks, dict) else []
        labels = tracks.get("tracks_label_class", []) if isinstance(tracks, dict) else []
        dims = cuboids.get("cuboids_dims", []) if isinstance(cuboids, dict) else []
        if not (isinstance(tracks, dict) and isinstance(cuboids, dict) and len(ids) == len(poses) == len(raw_times) == len(labels) == len(dims) and len(ids) > 0):
            raise ValueError("SEQUENCE_TRACKS_SCHEMA_INVALID")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".json.part")
        partial.write_bytes(raw)
        partial.replace(destination)
        return {
            "clip_id": clip_id,
            "path": str(destination),
            "source": repo_id,
            "archive": archive,
            "member": member_name,
            "size_bytes": len(raw),
            "downloaded_bytes": len(raw),
            "downloaded_this_round": True,
            "read_status": "DOWNLOADED",
            "schema_status": "VALID",
            "archive_size_bytes": archive_size,
            "member_compressed_size_bytes": int(member["compressed_size"]),
        }
    except urllib.error.HTTPError as exc:
        status = "NOT_FOUND_UPSTREAM" if exc.code == 404 else f"DOWNLOAD_FAILED:HTTP_{exc.code}"
        return {
            "clip_id": clip_id,
            "path": str(destination),
            "source": repo_id,
            "archive": archive,
            "member": None,
            "size_bytes": None,
            "downloaded_bytes": 0,
            "downloaded_this_round": False,
            "read_status": status,
            "schema_status": status,
        }
    except Exception as exc:
        return {
            "clip_id": clip_id,
            "path": str(destination),
            "source": repo_id,
            "archive": archive,
            "member": None,
            "size_bytes": None,
            "downloaded_bytes": 0,
            "downloaded_this_round": False,
            "read_status": f"DOWNLOAD_FAILED:{type(exc).__name__}",
            "error": str(exc),
            "schema_status": "UNREADABLE",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/nurec_cf_ttc_full300_manifest.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\full300_data_acquisition_v1"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\300_clip_nurec\00_raw"))
    parser.add_argument("--existing-pai-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\pai_obstacle_offline_v1"))
    parser.add_argument("--pai-repo", default=PAI_REPO)
    parser.add_argument("--nurec-repo", default=NUREC_REPO)
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--pai-before-count", type=int, default=2)
    parser.add_argument("--sequence-before-count", type=int, default=130)
    parser.add_argument("--other-downloaded-bytes", type=int, default=0)
    parser.add_argument("--reconcile-existing", action="store_true")
    parser.add_argument("--skip-pai", action="store_true")
    parser.add_argument("--skip-nurec", action="store_true")
    args = parser.parse_args()

    if args.reconcile_existing:
        print(json.dumps(reconcile_existing_output(args.output_root, args.pai_before_count, args.sequence_before_count, args.other_downloaded_bytes), indent=2, ensure_ascii=False))
        return 0

    clip_ids = discover_manifest(args.manifest)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "download_plan.csv",
        [
            {
                "clip_id": clip_id,
                "pai_component": "labels/egomotion.offline",
                "nurec_component": f"sample_set/{NUREC_RELEASE}/{clip_id}/{clip_id}.usdz::sequence_tracks.json",
                "requested": True,
                "source_pai": args.pai_repo,
                "source_nurec": args.nurec_repo,
            }
            for clip_id in clip_ids
        ],
    )

    pai_sources: dict[str, dict[str, Any]] = {}
    pai_scan_errors: list[dict[str, Any]] = []
    if not args.skip_pai:
        entries = load_pai_archive_entries(args.pai_repo)
        local_by_name = local_archive_by_remote_name(args.existing_pai_root)
        pai_sources, pai_scan_errors = scan_pai_archives(entries, local_by_name, args.scan_workers, set(clip_ids))
        write_json(
            output / "pai_archive_index.json",
            {
                "repository": args.pai_repo,
                "archive_count": len(entries),
                "target_hit_count": len(pai_sources),
                "scan_errors": pai_scan_errors,
                "targets": pai_sources,
            },
        )
    else:
        entries = []

    pai_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
        futures = {
            pool.submit(acquire_pai_clip, clip_id, pai_sources[clip_id], args.raw_root): clip_id
            for clip_id in clip_ids
            if clip_id in pai_sources
        }
        for future in as_completed(futures):
            pai_rows.append(future.result())
    pai_rows.extend(
        {
            "clip_id": clip_id,
            "source_url_or_repository": args.pai_repo,
            "archive": None,
            "member": None,
            "size_bytes": None,
            "downloaded_bytes": 0,
            "sha256_if_practical": None,
            "schema_valid": False,
            "status": "NOT_FOUND_UPSTREAM" if not args.skip_pai else "NOT_ATTEMPTED",
            "path": str(args.raw_root / "physicalai_offline" / clip_id / "egomotion.offline.parquet"),
        }
        for clip_id in clip_ids
        if clip_id not in pai_sources
    )
    pai_rows.sort(key=lambda row: row["clip_id"])
    write_csv(output / "pai_egomotion_download_inventory.csv", pai_rows)
    write_csv(
        output / "pai_egomotion_integrity.csv",
        [
            {
                "clip_id": row["clip_id"],
                "path": row.get("path"),
                "schema_valid": row.get("schema_valid"),
                "row_count": row.get("row_count"),
                "timestamp_min_us": row.get("timestamp_min_us"),
                "timestamp_max_us": row.get("timestamp_max_us"),
                "nonnegative_timestamp_count": row.get("nonnegative_timestamp_count"),
                "status": row.get("status"),
                "sha256": row.get("sha256_if_practical"),
            }
            for row in pai_rows
        ],
    )

    if not args.skip_nurec:
        nurec_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
            futures = {pool.submit(acquire_nurec_clip, clip_id, args.raw_root, args.nurec_repo): clip_id for clip_id in clip_ids}
            for future in as_completed(futures):
                nurec_rows.append(future.result())
    else:
        nurec_rows = [
            {
                "clip_id": clip_id,
                "path": str(args.raw_root / "nurec_full300" / clip_id / "sequence_tracks.json"),
                "source": args.nurec_repo,
                "archive": None,
                "member": None,
                "size_bytes": None,
                "downloaded_bytes": 0,
                "downloaded_this_round": False,
                "read_status": "NOT_ATTEMPTED",
                "schema_status": "NOT_ATTEMPTED",
            }
            for clip_id in clip_ids
        ]
    nurec_rows.sort(key=lambda row: row["clip_id"])
    write_csv(output / "sequence_tracks_full300_download_inventory.csv", nurec_rows)
    write_csv(
        output / "sequence_tracks_integrity.csv",
        [
            {
                "clip_id": row["clip_id"],
                "path": row.get("path"),
                "read_status": row.get("read_status"),
                "schema_status": row.get("schema_status"),
                "row_count": row.get("row_count"),
                "timestamp_min_us": row.get("timestamp_min_us"),
                "timestamp_max_us": row.get("timestamp_max_us"),
                "size_bytes": row.get("size_bytes"),
            }
            for row in nurec_rows
        ],
    )
    write_csv(
        output / "download_execution_log.csv",
        [
            {
                "clip_id": row["clip_id"],
                "component": "PAI_EGOMOTION_OFFLINE",
                "status": row.get("status"),
                "path": row.get("path"),
                "downloaded_bytes": row.get("downloaded_bytes", 0),
                "error": row.get("error"),
            }
            for row in pai_rows
        ]
        + [
            {
                "clip_id": row["clip_id"],
                "component": "NUREC_SEQUENCE_TRACKS_MEMBER",
                "status": row.get("read_status"),
                "path": row.get("path"),
                "downloaded_bytes": row.get("downloaded_bytes", 0),
                "error": row.get("error"),
            }
            for row in nurec_rows
        ],
    )

    pai_downloaded = sum(int(row.get("downloaded_bytes") or 0) for row in pai_rows)
    nurec_downloaded = sum(int(row.get("downloaded_bytes") or 0) for row in nurec_rows)
    pai_valid = sum(bool(row.get("schema_valid")) for row in pai_rows)
    nurec_valid = sum(row.get("schema_status") == "VALID" for row in nurec_rows)
    summary = {
        "generated_at_utc": utc_now(),
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "pai_egomotion_before": args.pai_before_count,
        "pai_egomotion_after": pai_valid,
        "pai_egomotion_downloaded_count": sum(row.get("status") == "DOWNLOADED" for row in pai_rows),
        "pai_egomotion_download_failed_count": sum(str(row.get("status", "")).startswith(("DOWNLOAD_FAILED", "INTEGRITY_FAILED")) for row in pai_rows),
        "sequence_tracks_before": args.sequence_before_count,
        "sequence_tracks_after": nurec_valid,
        "sequence_tracks_downloaded_count": sum(bool(row.get("downloaded_this_round")) and row.get("schema_status") == "VALID" for row in nurec_rows),
        "sequence_tracks_download_failed_count": sum(str(row.get("read_status", "")).startswith("DOWNLOAD_FAILED") for row in nurec_rows),
        "pai_egomotion_downloaded_bytes": pai_downloaded,
        "nurec_sequence_tracks_downloaded_bytes": nurec_downloaded,
        "other_downloaded_bytes": 0,
        "total_downloaded_bytes": pai_downloaded + nurec_downloaded,
        "full_clip_download_count": 0,
        "pai_archive_scan_count": len(entries),
        "pai_archive_scan_error_count": len(pai_scan_errors),
        "pai_target_archive_hit_count": len(pai_sources),
        "raw_root": str(args.raw_root),
        "component_level_download": True,
        "full_nurec_usdz_downloaded": False,
        "raw_data_committed_to_git": False,
    }
    write_json(output / "full300_acquisition_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
