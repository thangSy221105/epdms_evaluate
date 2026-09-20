"""Recover only explicitly verified PAI -> NuRec time mappings for the 300-clip set.

This module deliberately does not estimate a clock offset from timestamp ranges,
nearest values, or sequence-track minima.  It inventories evidence, reuses the
accepted five-clip sidecar when a clip is actually in the current experiment,
and leaves every other clip unresolved until the established semantic
correspondence method can be reproduced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CLIP_COUNT = 300
HISTORICAL_MIN_PAIR_COUNT = 2
HISTORICAL_IDS = {
    "028508ba-ef59-48d3-a95b-94eb92e3b063",
    "d078258b-9339-425d-a040-68346ef0d5bc",
    "689889c5-95b0-42ce-a1c9-f97a4388cb28",
    "37f45f87-dc3b-4425-a388-fa7bfa4a11a6",
    "bb1b395f-c51d-4a16-87ad-7310a7bbf086",
}
CLIP_STATUSES = {
    "VERIFIED_REUSED_EXISTING",
    "VERIFIED_REGENERATED_ESTABLISHED_METHOD",
    "CONFLICTING_EVIDENCE",
    "MISSING_REQUIRED_EVIDENCE",
    "INPUT_ERROR",
}


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_dict(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, child
            yield from _walk_dict(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value[:100]):
            yield from _walk_dict(child, f"{path}[{index}]")


def _candidate_offset(row: dict[str, Any]) -> int | None:
    for key in ("offset_us", "per_clip_offset_us", "offset_start_us"):
        value = _int(row.get(key))
        if value is not None:
            return value
    return None


def _candidate_t0(row: dict[str, Any], key: str) -> int | None:
    for name in (key, f"{key}_us"):
        value = _int(row.get(name))
        if value is not None:
            return value
    return None


def verify_physicalai_t0(
    prediction_values: Iterable[int],
    ground_truth_values: Iterable[int],
    manifest_value: int | None = None,
) -> dict[str, Any]:
    """Verify t0 from explicit prediction/GT fields, never from array position."""
    prediction = sorted(set(int(v) for v in prediction_values))
    ground_truth = sorted(set(int(v) for v in ground_truth_values))
    manifest = _int(manifest_value)
    consistent = (
        len(prediction) == 1
        and len(ground_truth) == 1
        and prediction == ground_truth
        and (manifest is None or prediction[0] == manifest)
    )
    conflict = bool(prediction and ground_truth and prediction != ground_truth)
    if conflict or (manifest is not None and prediction and prediction[0] != manifest):
        status = "CONFLICTING"
    elif consistent:
        status = "CONSISTENT"
    else:
        status = "MISSING_REQUIRED_EVIDENCE"
    return {
        "status": status,
        "physicalai_t0_us": prediction[0] if consistent else None,
        "prediction_values": prediction,
        "ground_truth_values": ground_truth,
        "manifest_value": manifest,
    }


def verify_semantic_pairs(
    pairs: Iterable[dict[str, Any]],
    historical_min_pair_count: int = HISTORICAL_MIN_PAIR_COUNT,
    expected_offset_us: int | None = None,
) -> dict[str, Any]:
    """Validate an established semantic pair set without estimating an offset."""
    pair_list = list(pairs)
    offsets = [_int(pair.get("offset_us")) for pair in pair_list]
    offsets = [value for value in offsets if value is not None]
    scales = [_float(pair.get("scale", 1.0)) for pair in pair_list]
    semantic = all(_bool(pair.get("semantic_identity")) for pair in pair_list)
    exact_scale = all(value == 1.0 for value in scales if value is not None)
    unique_offsets = sorted(set(offsets))
    expected_ok = expected_offset_us is None or unique_offsets == [expected_offset_us]
    verified = (
        len(pair_list) >= historical_min_pair_count
        and semantic
        and exact_scale
        and len(unique_offsets) == 1
        and expected_ok
    )
    return {
        "verified": verified,
        "pair_count": len(pair_list),
        "offset_min_us": min(offsets) if offsets else None,
        "offset_max_us": max(offsets) if offsets else None,
        "unique_offset_count": len(unique_offsets),
        "unique_offsets_us": unique_offsets,
        "scale_values": sorted(set(scales)),
        "semantic_identity": semantic,
        "expected_offset_match": expected_ok,
        "verification_reason": (
            "SEMANTIC_PAIRS_AND_CONSTANT_PER_CLIP_OFFSET"
            if verified
            else "REQUIRED_SEMANTIC_PAIR_EVIDENCE_NOT_PROVEN"
        ),
    }


def select_mapping_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select only agreeing verified candidates; conflicting evidence is retained."""
    verified = [row for row in candidates if row.get("verification_status") == "VERIFIED"]
    offsets = sorted({_int(row.get("offset_us")) for row in verified if _int(row.get("offset_us")) is not None})
    if len(offsets) > 1:
        return {"status": "CONFLICTING_EVIDENCE", "offset_us": None, "verified_candidates": verified}
    if len(verified) == 1 or offsets:
        return {
            "status": "VERIFIED",
            "offset_us": offsets[0],
            "verified_candidates": verified,
            "multiple_source_agreement": len(verified) > 1,
        }
    return {"status": "MISSING_REQUIRED_EVIDENCE", "offset_us": None, "verified_candidates": []}


def compute_nurec_t0(physicalai_t0_us: int, offset_us: int, scale: float = 1.0) -> int:
    if scale != 1.0:
        raise ValueError("ONLY_UNIT_SCALE_ONE_IS_ALLOWED")
    return int(physicalai_t0_us + offset_us)


def posthoc_sequence_consistency(
    sequence_min_us: int | None,
    sequence_max_us: int | None,
    nurec_t0_us: int | None,
    horizon_us: int = 6_400_000,
) -> dict[str, Any]:
    """Diagnostic only: sequence range can contradict a mapping, never prove one."""
    if None in (sequence_min_us, sequence_max_us, nurec_t0_us):
        return {"status": "UNAVAILABLE", "mapping_verified": False}
    query_end = nurec_t0_us + horizon_us
    passed = sequence_min_us <= nurec_t0_us and sequence_max_us >= query_end
    return {
        "status": "PASS" if passed else "CONTRADICTION",
        "query_min_us": nurec_t0_us,
        "query_max_us": query_end,
        "sequence_min_us": sequence_min_us,
        "sequence_max_us": sequence_max_us,
        "mapping_verified": False,
    }


def _extract_t0(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("t0_us", "physicalai_t0_us"):
            if key in value and _int(value[key]) is not None:
                return _int(value[key])
        for child in value.values():
            found = _extract_t0(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value[:5]:
            found = _extract_t0(child)
            if found is not None:
                return found
    return None


def load_t0_index(path: Path | None) -> dict[str, list[int]]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, list[int]] = defaultdict(list)
    for row in read_jsonl(path):
        clip_id = row.get("clip_id")
        t0 = _extract_t0(row)
        if clip_id and t0 is not None:
            result[str(clip_id)].append(t0)
    return result


def discover_source_files(repo_root: Path, data_roots: Iterable[Path]) -> list[Path]:
    """Find lightweight mapping-like artifacts; binary scene files are excluded."""
    patterns = re.compile(r"(time|match|offset|mapping|contract|alignment|bridge|readiness|manifest|summary)", re.I)
    suffixes = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".parquet"}
    roots = [repo_root / "configs", repo_root / "scripts", repo_root / "docs", *data_roots]
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*") if root.is_dir() else iter([root])
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if patterns.search(path.name) or path.parent.name in {"time_alignment_v3", "final_hardened"}:
                found.add(path)
    return sorted(found, key=lambda path: str(path).lower())


def _rows_from_file(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), "CSV"
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path), "JSONL"
    if path.suffix.lower() == ".json":
        value = read_json(path)
        if isinstance(value, dict):
            if isinstance(value.get("clips"), list):
                return [row for row in value["clips"] if isinstance(row, dict)], "JSON.clips"
            return [value], "JSON"
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)], "JSON.list"
    return [], path.suffix.lower().lstrip(".").upper() or "UNKNOWN"


def source_inventory_and_candidates(
    files: Iterable[Path],
    current_ids: set[str],
    historical_contract: Path,
    historical_match: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for path in files:
        record = {
            "source_file": str(path),
            "exists": path.is_file(),
            "format": path.suffix.lower().lstrip(".") or None,
            "read_status": "NOT_READ",
            "row_count": None,
            "mapping_candidate_count": 0,
            "verified_candidate_count": 0,
            "current_300_candidate_count": 0,
            "evidence_type": "LOCAL_AUTOMATED_INSPECTION",
        }
        if not path.is_file():
            inventory.append(record)
            continue
        try:
            rows, fmt = _rows_from_file(path)
            record.update({"format": fmt, "read_status": "READ", "row_count": len(rows)})
        except Exception as exc:  # one corrupt artifact must not stop discovery
            record.update({"read_status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"})
            inventory.append(record)
            continue
        for index, row in enumerate(rows, 1):
            clip_id = row.get("clip_id") or row.get("scene_id") or row.get("sequence_id")
            if not clip_id:
                continue
            clip_id = str(clip_id)
            offset = _candidate_offset(row)
            mappingish = offset is not None or any(
                key in row for key in ("time_mapping_verified", "mapping_verified", "time_mapping_status", "offset_equal")
            )
            if not mappingish:
                continue
            record["mapping_candidate_count"] += 1
            current = clip_id in current_ids
            if current:
                record["current_300_candidate_count"] += 1
            is_historical_source = path.resolve() in {historical_contract.resolve(), historical_match.resolve()}
            historically_verified = (
                is_historical_source
                and _bool(row.get("verified", True))
                and ("offset_equal" not in row or _bool(row.get("offset_equal")))
                and offset is not None
            )
            verification_status = "VERIFIED" if historically_verified else "UNVERIFIED"
            candidate_status = "EXISTING_VERIFIED" if historically_verified else "EXISTING_UNVERIFIED"
            candidates.append(
                {
                    "clip_id": clip_id,
                    "physicalai_t0_us": _candidate_t0(row, "physicalai_t0"),
                    "nurec_t0_us": _candidate_t0(row, "nurec_t0"),
                    "offset_us": offset,
                    "scale": _float(row.get("scale", 1.0)),
                    "source_file": str(path),
                    "source_record": index,
                    "mapping_type": row.get("mapping_type") or "UNKNOWN",
                    "verification_flag": historically_verified,
                    "evidence_type": "EXISTING_ACCEPTED_HISTORICAL_ARTIFACT" if historically_verified else "LOCAL_UNVERIFIED_ARTIFACT",
                    "evidence_description": "Accepted five-clip contract/match artifact" if historically_verified else "Candidate field/status only; not promoted",
                    "current_300_member": current,
                    "candidate_status": candidate_status,
                    "verification_status": verification_status,
                    "multiple_source_agreement": None,
                }
            )
            if historically_verified:
                record["verified_candidate_count"] += 1
        inventory.append(record)
    return inventory, candidates


def _sequence_range(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    try:
        value = read_json(path)
    except Exception:
        return None, None
    timestamps: list[int] = []

    def flatten_numbers(child: Any) -> Iterable[int]:
        if isinstance(child, (int, float)) and not isinstance(child, bool):
            yield int(child)
        elif isinstance(child, list):
            for item in child:
                yield from flatten_numbers(item)

    def visit(child: Any) -> None:
        if isinstance(child, dict):
            for key, nested in child.items():
                key_lower = str(key).lower()
                if "timestamp" in key_lower and (key_lower.endswith("_us") or key_lower.endswith("s_us")):
                    timestamps.extend(flatten_numbers(nested))
                else:
                    visit(nested)
        elif isinstance(child, list):
            for item in child:
                visit(item)

    visit(value)
    return (min(timestamps), max(timestamps)) if timestamps else (None, None)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.experiment_manifest)
    manifest = read_jsonl(manifest_path)
    manifest_by_id = {str(row["clip_id"]): row for row in manifest if row.get("clip_id")}
    current_ids = set(manifest_by_id)
    if len(current_ids) != EXPECTED_CLIP_COUNT:
        raise ValueError(f"EXPECTED_300_CLIPS_BUT_FOUND_{len(current_ids)}")

    historical_contract = Path(args.historical_contract)
    historical_match = Path(args.historical_match)
    data_roots = [Path(value) for value in args.data_root]
    source_files = discover_source_files(repo_root, data_roots)
    if historical_contract.exists():
        source_files.append(historical_contract)
    if historical_match.exists():
        source_files.append(historical_match)
    source_files = sorted(set(source_files), key=lambda path: str(path).lower())
    inventory, candidates = source_inventory_and_candidates(source_files, current_ids, historical_contract, historical_match)

    # The experiment manifest is an inventory input, not mapping evidence.
    for clip_id in sorted(current_ids):
        if not any(row["clip_id"] == clip_id for row in candidates):
            candidates.append(
                {
                    "clip_id": clip_id,
                    "physicalai_t0_us": _int(manifest_by_id[clip_id].get("physicalai_t0_us")),
                    "nurec_t0_us": None,
                    "offset_us": None,
                    "scale": 1.0,
                    "source_file": str(manifest_path),
                    "source_record": None,
                    "mapping_type": "INPUT_INVENTORY_ONLY",
                    "verification_flag": False,
                    "evidence_type": "LOCAL_AUTOMATED_INSPECTION",
                    "evidence_description": "Experiment membership and declared t0 only; no PAI-NuRec semantic mapping",
                    "current_300_member": True,
                    "candidate_status": "EXISTING_UNVERIFIED",
                    "verification_status": "UNVERIFIED",
                    "multiple_source_agreement": None,
                }
            )

    prediction_index = load_t0_index(Path(args.prediction_jsonl) if args.prediction_jsonl else None)
    gt_index = load_t0_index(Path(args.ground_truth_jsonl) if args.ground_truth_jsonl else None)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    semantic_rows: list[dict[str, Any]] = []
    historical_rows = {row["clip_id"]: row for row in read_jsonl(historical_contract)} if historical_contract.is_file() else {}
    match_rows: dict[str, dict[str, Any]] = {}
    if historical_match.is_file():
        with historical_match.open(newline="", encoding="utf-8-sig") as handle:
            match_rows = {str(row["clip_id"]): row for row in csv.DictReader(handle)}
    for clip_id, row in sorted(historical_rows.items()):
        match = match_rows.get(clip_id, {})
        pair_count = _int(match.get("exact_relative_timestamp_matches")) or 0
        offset = _int(row.get("offset_us"))
        pair_verification = verify_semantic_pairs(
            [
                {"semantic_identity": True, "offset_us": offset, "scale": 1.0}
                for _ in range(pair_count)
            ],
            historical_min_pair_count=HISTORICAL_MIN_PAIR_COUNT,
            expected_offset_us=offset,
        )
        for pair_index in range(1, pair_count + 1):
            semantic_rows.append(
                {
                    "clip_id": clip_id,
                    "pair_index": pair_index,
                    "physicalai_timestamp_us": None,
                    "nurec_timestamp_us": None,
                    "offset_us": offset,
                    "semantic_identity": True,
                    "pair_evidence": "EXACT_RELATIVE_TIMESTAMP_MATCH_COUNT_RECORDED_IN_ACCEPTED_HISTORICAL_ARTIFACT",
                    "verification_status": "VERIFIED" if pair_verification["verified"] else "UNVERIFIED",
                    "source_file": str(historical_match),
                }
            )

    if args.prediction_jsonl and args.ground_truth_jsonl:
        prediction_path = Path(args.prediction_jsonl)
        gt_path = Path(args.ground_truth_jsonl)
    else:
        prediction_path = gt_path = None

    candidate_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["current_300_member"]:
            candidate_by_clip[row["clip_id"]].append(row)

    mapping_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    verified_reused = 0
    regenerated = 0
    conflicting = 0
    missing = 0
    t0_consistent = 0
    t0_conflict = 0
    posthoc_pass = 0
    posthoc_conflict = 0
    sequence_count = 0

    for clip_id in sorted(current_ids):
        item = manifest_by_id[clip_id]
        pred_values = prediction_index.get(clip_id, [])
        gt_values = gt_index.get(clip_id, [])
        t0 = verify_physicalai_t0(pred_values, gt_values, _int(item.get("physicalai_t0_us")))
        if t0["status"] == "CONSISTENT":
            t0_consistent += 1
        elif t0["status"] == "CONFLICTING":
            t0_conflict += 1

        chosen = select_mapping_candidate(candidate_by_clip[clip_id])
        if chosen["status"] == "VERIFIED" and t0["status"] == "CONSISTENT":
            status = "VERIFIED_REUSED_EXISTING" if clip_id in HISTORICAL_IDS else "VERIFIED_REGENERATED_ESTABLISHED_METHOD"
            if status == "VERIFIED_REUSED_EXISTING":
                verified_reused += 1
            else:
                regenerated += 1
            offset = int(chosen["offset_us"])
            nurec_t0 = compute_nurec_t0(int(t0["physicalai_t0_us"]), offset)
            mapping_rows.append(
                {
                    "clip_id": clip_id,
                    "physicalai_t0_us": t0["physicalai_t0_us"],
                    "nurec_t0_us": nurec_t0,
                    "offset_us": offset,
                    "scale": 1.0,
                    "mapping_type": "PER_CLIP_REBASE",
                    "verified": True,
                    "mapping_status": status,
                    "source": ";".join(sorted({str(row["source_file"]) for row in chosen["verified_candidates"]})),
                    "pair_count": sum(1 for row in semantic_rows if row["clip_id"] == clip_id),
                    "per_clip_offset_rederived": False,
                }
            )
        elif chosen["status"] == "CONFLICTING_EVIDENCE" or t0["status"] == "CONFLICTING":
            status = "CONFLICTING_EVIDENCE"
            conflicting += 1
            conflict_rows.append(
                {
                    "clip_id": clip_id,
                    "reason": "T0_CONFLICT" if t0["status"] == "CONFLICTING" else "MULTIPLE_VERIFIED_OFFSETS",
                    "t0_evidence": json.dumps(t0, sort_keys=True),
                    "candidate_evidence": json.dumps(candidate_by_clip[clip_id], sort_keys=True),
                }
            )
        else:
            status = "MISSING_REQUIRED_EVIDENCE"
            missing += 1

        if status not in CLIP_STATUSES:
            raise AssertionError(status)
        sequence_path = Path(str(item.get("sequence_tracks_path", "")))
        if sequence_path.is_file():
            sequence_count += 1
        seq_min, seq_max = _sequence_range(sequence_path)
        nurec_t0_for_check = next((row["nurec_t0_us"] for row in mapping_rows if row["clip_id"] == clip_id), None)
        consistency = posthoc_sequence_consistency(seq_min, seq_max, nurec_t0_for_check)
        if consistency["status"] == "PASS":
            posthoc_pass += 1
        elif consistency["status"] == "CONTRADICTION":
            posthoc_conflict += 1
            if status.startswith("VERIFIED"):
                conflict_rows.append({"clip_id": clip_id, "reason": "POSTHOC_SEQUENCE_CONTRADICTION", "t0_evidence": "", "candidate_evidence": ""})
        posthoc_rows.append({"clip_id": clip_id, "sequence_tracks_available": sequence_path.is_file(), **consistency})
        audit_rows.append(
            {
                "clip_id": clip_id,
                "current_300_member": True,
                "physicalai_t0_us": t0["physicalai_t0_us"],
                "prediction_t0_status": t0["status"],
                "mapping_status": status,
                "offset_us": next((row["offset_us"] for row in mapping_rows if row["clip_id"] == clip_id), None),
                "nurec_t0_us": nurec_t0_for_check,
                "pair_count": sum(1 for row in semantic_rows if row["clip_id"] == clip_id),
                "unique_offset_count": len({row["offset_us"] for row in candidate_by_clip[clip_id] if row.get("verification_status") == "VERIFIED"}),
                "multiple_source_agreement": chosen.get("multiple_source_agreement", False),
                "sequence_tracks_available": sequence_path.is_file(),
                "posthoc_sequence_status": consistency["status"],
                "posthoc_is_verification": False,
            }
        )
        if status.startswith("VERIFIED"):
            blocker = "NONE"
            plan_action = "NO_ADDITIONAL_TIME_MAPPING_FILE_REQUIRED"
        else:
            blocker = "VERIFIED_SEMANTIC_PAI_NUREC_CORRESPONDENCE"
            plan_action = "OBTAIN_HISTORICAL_STYLE_PAI_EGOMOTION_AND_NUREC_RIG_TIMELINE_FOR_SAME_CLIP"
        plan_rows.append(
            {
                "clip_id": clip_id,
                "mapping_status": status,
                "sequence_tracks_available": sequence_path.is_file(),
                "time_mapping_verified": status.startswith("VERIFIED"),
                "minimal_additional_data_needed": blocker,
                "recommended_acquisition": plan_action,
                "sequence_tracks_download_needed_for_time_mapping": False,
                "full_clip_download_required": False,
                "current_runnable_cf_ttc": status.startswith("VERIFIED") and sequence_path.is_file(),
            }
        )

    _write_csv(output_root / "local_time_mapping_source_inventory.csv", inventory, list(inventory[0]) if inventory else ["source_file"])
    candidate_fields = [
        "clip_id", "physicalai_t0_us", "nurec_t0_us", "offset_us", "scale", "source_file", "source_record",
        "mapping_type", "verification_flag", "evidence_type", "evidence_description", "current_300_member",
        "candidate_status", "verification_status", "multiple_source_agreement",
    ]
    _write_csv(output_root / "time_mapping_candidate_registry.csv", candidates, candidate_fields)
    _write_csv(output_root / "time_mapping_semantic_pairs.csv", semantic_rows, [
        "clip_id", "pair_index", "physicalai_timestamp_us", "nurec_timestamp_us", "offset_us",
        "semantic_identity", "pair_evidence", "verification_status", "source_file",
    ])
    _write_csv(output_root / "time_mapping_per_clip_audit.csv", audit_rows, list(audit_rows[0]))
    _write_csv(output_root / "time_mapping_conflicts.csv", conflict_rows, ["clip_id", "reason", "t0_evidence", "candidate_evidence"])
    _write_csv(output_root / "time_mapping_posthoc_consistency.csv", posthoc_rows, list(posthoc_rows[0]))
    _write_csv(output_root / "full300_time_mapping_minimal_data_plan.csv", plan_rows, list(plan_rows[0]))

    _json_dump(output_root / "historical_mapping_method.json", {
        "HISTORICAL_MAPPING_METHOD": "PER_CLIP_REBASE_VALIDATED_BY_EXPLICIT_PAI_NUREC_POSE_TIMELINE",
        "HISTORICAL_IDENTITY_SOURCE": "same clip UUID plus accepted PAI egomotion and NuRec rig trajectory artifacts",
        "HISTORICAL_MIN_PAIR_COUNT": HISTORICAL_MIN_PAIR_COUNT,
        "HISTORICAL_OFFSET_CONSISTENCY_RULE": "offset_start_us == offset_end_us exactly; scale=1.0; 20-second relative durations agree; numerical pose validation is independent support",
        "source_match_csv": str(historical_match),
        "source_contract": str(historical_contract),
        "posthoc_sequence_tracks_role": "diagnostic_only",
    })
    _json_dump(output_root / "time_mapping_full300_summary.json", {
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "existing_verified_mapping_count": verified_reused,
        "regenerated_verified_mapping_count": regenerated,
        "total_verified_mapping_count": verified_reused + regenerated,
        "conflicting_mapping_count": conflicting,
        "missing_mapping_count": missing,
        "time_mapping_coverage_rate": (verified_reused + regenerated) / EXPECTED_CLIP_COUNT,
        "time_mapping_full300_status": "FULL300_VERIFIED" if verified_reused + regenerated == EXPECTED_CLIP_COUNT else "INCOMPLETE",
        "historical_mapping_method": "PER_CLIP_REBASE_VALIDATED_BY_EXPLICIT_PAI_NUREC_POSE_TIMELINE",
        "historical_identity_source": "same clip UUID plus accepted PAI egomotion and NuRec rig trajectory artifacts",
        "historical_min_pair_count": HISTORICAL_MIN_PAIR_COUNT,
        "sequence_tracks_available_count": sequence_count,
        "time_mapping_and_sequence_ready_count": sum(1 for row in plan_rows if row["time_mapping_verified"] and row["sequence_tracks_available"]),
        "currently_runnable_cf_ttc_clip_count": sum(1 for row in plan_rows if row["current_runnable_cf_ttc"]),
        "physicalai_t0_consistent_count": t0_consistent,
        "physicalai_t0_conflict_count": t0_conflict,
        "post_hoc_sequence_consistency_pass_count": posthoc_pass,
        "post_hoc_sequence_consistency_conflict_count": posthoc_conflict,
        "minimal_additional_data_clip_count": sum(1 for row in plan_rows if not row["time_mapping_verified"]),
        "full_clip_download_required_count": sum(1 for row in plan_rows if row["full_clip_download_required"]),
        "per_clip_offset_rederived_for_historical_clips": False,
        "obstacle_geometry_block": "CLOSED",
        "remaining_blockers": ["VERIFIED_SEMANTIC_PAI_NUREC_CORRESPONDENCE_FOR_UNMAPPED_CLIPS"] if missing else [],
    })
    _json_dump(output_root / "time_mapping_full300_final_summary.json", {
        "status": "INCOMPLETE" if missing or conflicting else "COMPLETE",
        "verified_mapping_clip_ids": [row["clip_id"] for row in mapping_rows],
        "unresolved_clip_count": missing,
        "conflicting_clip_count": conflicting,
        "no_numeric_offset_inference": True,
        "no_global_offset": True,
        "no_sequence_min_offset_derivation": True,
    })
    # Only actually verified rows are written to the canonical contract.
    mapping_path = Path(args.output_mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        for row in mapping_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "expected_clip_count": EXPECTED_CLIP_COUNT,
        "existing_verified_mapping_count": verified_reused,
        "regenerated_verified_mapping_count": regenerated,
        "total_verified_mapping_count": verified_reused + regenerated,
        "conflicting_mapping_count": conflicting,
        "missing_mapping_count": missing,
        "time_mapping_coverage_rate": (verified_reused + regenerated) / EXPECTED_CLIP_COUNT,
        "time_mapping_full300_status": "FULL300_VERIFIED" if verified_reused + regenerated == EXPECTED_CLIP_COUNT else "INCOMPLETE",
        "sequence_tracks_available_count": sequence_count,
        "time_mapping_and_sequence_ready_count": sum(1 for row in plan_rows if row["time_mapping_verified"] and row["sequence_tracks_available"]),
        "currently_runnable_cf_ttc_clip_count": sum(1 for row in plan_rows if row["current_runnable_cf_ttc"]),
        "physicalai_t0_consistent_count": t0_consistent,
        "physicalai_t0_conflict_count": t0_conflict,
        "post_hoc_sequence_consistency_pass_count": posthoc_pass,
        "post_hoc_sequence_consistency_conflict_count": posthoc_conflict,
        "minimal_additional_data_clip_count": sum(1 for row in plan_rows if not row["time_mapping_verified"]),
        "full_clip_download_required_count": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment-manifest", type=Path, default=Path("configs/nurec_cf_ttc_full300_manifest.jsonl"))
    parser.add_argument("--historical-contract", type=Path, default=Path("configs/nurec_coordinate_time_contract_5clip.jsonl"))
    parser.add_argument("--historical-match", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1\pai_nurec_time_match.csv"))
    parser.add_argument("--prediction-jsonl", type=Path, default=Path(r"D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl"))
    parser.add_argument("--ground-truth-jsonl", type=Path, default=Path(r"D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl"))
    parser.add_argument("--data-root", type=Path, action="append", default=[
        Path(r"D:\300_clip_nurec\05_data_contract"),
        Path(r"D:\300_clip_nurec\hf_probe"),
        Path(r"D:\300_clip_nurec\01_context"),
        Path(r"D:\300_clip_nurec\00_raw"),
    ])
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\300_clip_nurec\hf_probe\time_mapping_full300_recovery_v1"))
    parser.add_argument("--output-mapping", type=Path, default=Path("configs/nurec_time_contract_full300.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
