"""Run identity, effective fingerprint, and checkpoint verification for EPDMS."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class RunIdentityError(ValueError):
    """Base error for run identity and checkpoint verification failures."""
    pass


class AmbiguousCheckpointError(RunIdentityError):
    """Raised when conflicting checkpoint files exist without verifiable relation."""
    pass


def compute_file_content_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file by reading chunks of 64KB."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_map_directory_content_sha256(map_dir: Path) -> str:
    """Computes binary content SHA-256 across all parquet map files in directory."""
    if not map_dir.is_dir():
        return ""
    hasher = hashlib.sha256()
    for clip_dir in sorted(map_dir.iterdir()):
        if clip_dir.is_dir():
            clipgt = clip_dir / "clipgt"
            for pq_name in ["lane.parquet", "intersection_area.parquet"]:
                pq_file = clipgt / pq_name
                if pq_file.is_file():
                    hasher.update(f"{clip_dir.name}/{pq_name}:".encode("utf-8"))
                    with pq_file.open("rb") as f:
                        while chunk := f.read(65536):
                            hasher.update(chunk)
    return hasher.hexdigest()


def compute_run_effective_fingerprint(
    config_dict: Dict[str, Any],
    source_hashes: Dict[str, str],
    runtime_overrides: Optional[Dict[str, Any]] = None,
    clip_scope: Optional[List[str]] = None,
) -> str:
    """Computes a deterministic SHA-256 fingerprint for a run configuration."""
    identity: Dict[str, Any] = {
        "metric_profile": config_dict.get("metric_profile"),
        "horizon_s": config_dict.get("horizon_s"),
        "frequency_hz": config_dict.get("frequency_hz"),
        "strict_mode": config_dict.get("strict_mode"),
        "touch_is_collision": config_dict.get("proxy", {}).get("touch_is_collision"),
        "ttc_horizon_s": config_dict.get("proxy", {}).get("ttc_horizon_s"),
        "progress_stationary_threshold_m": config_dict.get("proxy", {}).get("progress_stationary_threshold_m"),
        "vehicle": config_dict.get("vehicle"),
        "source_hashes": source_hashes,
        "clip_scope": sorted(clip_scope) if clip_scope is not None else "ALL",
        "implementation_version": "v2.2.0_r4_contracts",
    }
    if runtime_overrides:
        identity.update(runtime_overrides)

    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_resume_safety_before_recovery(
    score_dir: Path,
    expected_fingerprint: str,
    manifest_filename: str = "run_manifest.json",
    artifact_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Verifies that existing artifacts in score_dir can safely be resumed BEFORE any recovery.
    
    Checks score_jsonl, score_tmp, error_jsonl, error_tmp.
    Raises:
        RunIdentityError: if any non-empty artifact exists but manifest is missing or fingerprint mismatches.
    Returns:
        Previous manifest dict if valid, or None if starting fresh.
    """
    if artifact_names is None:
        artifact_names = [
            "epdms_scores_300.jsonl",
            "epdms_scores_300.jsonl.tmp",
            "epdms_errors_300.jsonl",
            "epdms_errors_300.jsonl.tmp",
        ]

    manifest_path = score_dir / manifest_filename

    # 1. Check if ANY non-empty artifact exists
    existing_artifacts = []
    for name in artifact_names:
        p = score_dir / name
        if p.is_file() and p.stat().st_size > 0:
            existing_artifacts.append(name)

    if not existing_artifacts:
        # Completely fresh start
        return None

    # 2. Artifacts exist -> Manifest MUST exist
    if not manifest_path.is_file():
        raise RunIdentityError(
            f"Resume rejected: Existing artifacts found ({existing_artifacts}) but manifest "
            f"({manifest_filename}) is missing. Cannot verify run identity or fingerprint."
        )

    # 3. Manifest must be valid JSON and have matching fingerprint
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            prev_manifest = json.load(f)
    except Exception as ex:
        raise RunIdentityError(f"Resume rejected: Failed to parse manifest ({manifest_filename}): {ex}")

    prev_fp = prev_manifest.get("effective_fingerprint")
    if not prev_fp or prev_fp != expected_fingerprint:
        raise RunIdentityError(
            f"Resume rejected: Configuration fingerprint mismatch "
            f"(existing={prev_fp}, current={expected_fingerprint}). "
            f"Existing artifacts ({existing_artifacts}) belong to a different run configuration."
        )

    return prev_manifest


def write_manifest_atomic(manifest_path: Path, manifest_data: Dict[str, Any]) -> None:
    """Atomically writes manifest file using a temporary file and os.replace."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".tmp")
    manifest_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, manifest_path)
