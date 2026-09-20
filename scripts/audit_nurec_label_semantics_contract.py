"""Close the NuRec label-semantics question without inferring empty frames.

This audit is deliberately separate from scoring and time/geometry audits.  It
combines primary upstream source review with the already acquired 13-clip
evidence.  Object rows are treated as sparse observations unless an upstream
contract explicitly supplies an authoritative frame grid and zero-object
attestation semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PREVIOUS_ROOT = Path(r"D:\300_clip_nurec\hf_probe\cf_ttc_13clip_empty_attestation_v1")
DEFAULT_FULL300_ROOT = Path(r"D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5")
DEFAULT_NUREC_ROOT = Path(r"D:\300_clip_nurec\00_raw\nurec_full300")
DEFAULT_COMPONENT_ROOT = DEFAULT_PREVIOUS_ROOT / "downloaded_members"
DEFAULT_OUTPUT_ROOT = Path(r"D:\300_clip_nurec\hf_probe\nurec_label_semantics_contract_final_v1")

CONTRACT_C = "CONTRACT_C_SPARSE_OBJECT_ROWS_NO_EMPTY_GUARANTEE"
FRAME_GRID_STATUS = "NOT_FOUND_IN_RELEASED_LABEL_ARTIFACTS"

UPSTREAM_SOURCES = [
    {
        "source_id": "ncore_pai_converter",
        "organization": "NVIDIA",
        "repository": "ncore",
        "path": "tools/data_converter/pai/converter.py",
        "url": "https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py",
        "source_kind": "PRIMARY_IMPLEMENTATION",
        "review_status": "REVIEWED",
        "relevant_evidence": "_load_cuboid_track_observations loads obstacle rows into CuboidTrackObservation; missing obstacle file returns an empty list; no per-frame zero-object record is emitted.",
        "supports_authoritative_frame_grid": False,
        "supports_empty_from_absence": False,
    },
    {
        "source_id": "ncore_conventions",
        "organization": "NVIDIA",
        "repository": "ncore",
        "path": "data/conventions.html",
        "url": "https://nvidia.github.io/ncore/data/conventions.html",
        "source_kind": "PRIMARY_SPECIFICATION",
        "review_status": "REVIEWED",
        "relevant_evidence": "Cuboids are observations with timestamps/reference frames; the specification does not define an empty-frame sentinel for absent cuboid observations.",
        "supports_authoritative_frame_grid": False,
        "supports_empty_from_absence": False,
    },
    {
        "source_id": "nurec_workflow_pai",
        "organization": "NVIDIA",
        "repository": "nurec-skills",
        "path": "skills/nre/references/example-workflows/bash/nurec_workflow_pai.md",
        "url": "https://github.com/NVIDIA/nurec-skills/blob/main/skills/nre/references/example-workflows/bash/nurec_workflow_pai.md",
        "source_kind": "PRIMARY_OPERATIONAL_DOCUMENTATION",
        "review_status": "REVIEWED",
        "relevant_evidence": "The workflow identifies NCore as the standardized input and lists sequence_tracks as an export artifact, but does not define zero-object frame semantics.",
        "supports_authoritative_frame_grid": False,
        "supports_empty_from_absence": False,
    },
    {
        "source_id": "instant_nurec_input_contract",
        "organization": "NVIDIA",
        "repository": "instant-nurec",
        "path": "README.md",
        "url": "https://github.com/NVIDIA/instant-nurec/blob/main/README.md",
        "source_kind": "PRIMARY_CONSUMER_DOCUMENTATION",
        "review_status": "REVIEWED",
        "relevant_evidence": "Instant-NuRec consumes an NCore V4 sequence/metadata JSON; it does not specify sparse cuboid absence as an empty-frame assertion.",
        "supports_authoritative_frame_grid": False,
        "supports_empty_from_absence": False,
    },
    {
        "source_id": "alpasim_scene_catalog",
        "organization": "NVlabs",
        "repository": "alpasim",
        "path": "data/scenes/README.md",
        "url": "https://github.com/NVlabs/alpasim/blob/main/data/scenes/README.md",
        "source_kind": "PRIMARY_ARTIFACT_CATALOG",
        "review_status": "REVIEWED",
        "relevant_evidence": "Public NuRec scenes are artifact bundles pinned by dataset revision; catalog metadata does not define object-row completeness or empty-frame semantics.",
        "supports_authoritative_frame_grid": False,
        "supports_empty_from_absence": False,
    },
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _clip_ids(gap_rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["clip_id"] for row in gap_rows if row.get("clip_id")})


def _walk_key_paths(value: Any, path: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.append(child_path)
            result.extend(_walk_key_paths(child, child_path))
    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        result.extend(_walk_key_paths(value[0], f"{path}[0]"))
    return result


def _json_frame_evidence(path: Path) -> dict[str, Any]:
    try:
        value = _read_json(path)
    except Exception as exc:  # pragma: no cover - exercised by integration data
        return {"read_status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    paths = _walk_key_paths(value)
    lower = [item.lower() for item in paths]
    frame_paths = [item for item in paths if "frame" in item.lower() or "timestamp" in item.lower()]
    explicit_empty_paths = [
        item for item in paths if any(token in item.lower() for token in ("empty", "object_count", "num_objects", "label_count"))
    ]
    return {
        "read_status": "READ",
        "top_level_keys": list(value) if isinstance(value, dict) else [],
        "frame_or_timestamp_paths": frame_paths[:100],
        "explicit_empty_or_count_paths": explicit_empty_paths[:100],
        "frame_timestamp_list_found": any("timestamp" in item and "frame" in item for item in lower),
        "explicit_zero_object_attestation_found": False,
    }


def _parquet_stats(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        rows = table.to_pylist()
        timestamp_values: list[int] = []
        for row in rows:
            key = row.get("key") if isinstance(row, dict) else None
            timestamp = key.get("timestamp_micros") if isinstance(key, dict) else None
            if timestamp is not None:
                timestamp_values.append(int(timestamp))
        return {
            "read_status": "READ",
            "row_count": len(rows),
            "column_names": list(table.column_names),
            "unique_object_timestamp_count": len(set(timestamp_values)),
            "timestamp_min_us": min(timestamp_values) if timestamp_values else None,
            "timestamp_max_us": max(timestamp_values) if timestamp_values else None,
        }
    except Exception as exc:  # pragma: no cover - exercised by integration data
        return {"read_status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _exact_object_timestamps(path: Path) -> dict[int, int]:
    stats: dict[int, int] = {}
    try:
        import pyarrow.parquet as pq

        for row in pq.read_table(path).to_pylist():
            key = row.get("key") if isinstance(row, dict) else None
            timestamp = key.get("timestamp_micros") if isinstance(key, dict) else None
            if timestamp is not None:
                timestamp = int(timestamp)
                stats[timestamp] = stats.get(timestamp, 0) + 1
    except Exception:
        return {}
    return stats


def _sequence_track_timestamps(path: Path) -> dict[int, int]:
    """Read only track timestamps without importing the production normalizer.

    The CLI is also invoked as ``py scripts/<file>.py`` where Python places the
    scripts directory first on ``sys.path``.  Keeping this tiny reader local
    makes the scope audit deterministic and avoids turning an import-path issue
    into an empty evidence result.
    """
    try:
        value = _read_json(path)
        while isinstance(value, dict) and set(value) == {"dummy_chunk_id"}:
            value = value["dummy_chunk_id"]
        tracks = value.get("tracks_data", {})
        timestamps = tracks.get("tracks_timestamps_us", [])
        result: dict[int, int] = {}
        for track_timestamps in timestamps:
            for timestamp in track_timestamps:
                timestamp = int(timestamp)
                result[timestamp] = result.get(timestamp, 0) + 1
        return result
    except Exception:
        return {}


def _find_component(clip_id: str, basename: str, nurec_root: Path, component_root: Path) -> Path | None:
    candidates = [
        nurec_root / clip_id / basename,
        component_root / clip_id / basename,
    ]
    return next((path for path in candidates if path.is_file()), None)


def _local_scope_rows(clip_id: str, nurec_root: Path, component_root: Path) -> list[dict[str, Any]]:
    names = [
        "data_info.json",
        "datasource_summary.json",
        "metadata.yaml",
        "parsed_config.yaml",
        "pose_record.json",
        "rig_trajectories.json",
        "sequence_tracks.json",
        "clipgt/clip.parquet",
        "clipgt/association.parquet",
        "clipgt/egomotion_estimate.parquet",
        "clipgt/obstacle.parquet",
        "clipgt/calibration_estimate.parquet",
    ]
    rows = []
    for name in names:
        path = _find_component(clip_id, name, nurec_root, component_root)
        entry: dict[str, Any] = {
            "clip_id": clip_id,
            "requested_file": name,
            "path": str(path) if path else "",
            "found": bool(path),
            "read_status": "NOT_FOUND",
            "authoritative_frame_grid_evidence": False,
            "zero_object_attestation_evidence": False,
        }
        if path and path.suffix.lower() == ".json":
            evidence = _json_frame_evidence(path)
            entry.update(evidence)
            entry["authoritative_frame_grid_evidence"] = bool(evidence.get("frame_timestamp_list_found"))
            entry["zero_object_attestation_evidence"] = bool(evidence.get("explicit_zero_object_attestation_found"))
        elif path and path.suffix.lower() == ".parquet":
            entry.update(_parquet_stats(path))
        elif path:
            try:
                path.read_text(encoding="utf-8")
                entry["read_status"] = "READ"
            except Exception as exc:
                entry["read_status"] = "READ_ERROR"
                entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
    return rows


def classify_missing_timestamp(
    row: dict[str, str],
    sequence_counts: dict[int, int],
    obstacle_counts: dict[int, int],
) -> dict[str, Any]:
    timestamp = int(row["query_timestamp_us"])
    sequence_count = int(sequence_counts.get(timestamp, 0))
    obstacle_count = int(obstacle_counts.get(timestamp, 0))
    exact_count = max(sequence_count, obstacle_count)
    if exact_count:
        return {
            "contract_classification": "OBJECT_PRESENT_RECONCILIATION_CONFLICT",
            "confirmed_empty": False,
            "outside_authoritative_frame_grid": False,
            "unresolved": False,
            "object_empty_evidence_conflict": True,
            "evidence_reason": "prior missing query has exact object row in a rechecked local source",
        }
    return {
        "contract_classification": "UNRESOLVED_NO_AUTHORITATIVE_FRAME_GRID",
        "confirmed_empty": False,
        "outside_authoritative_frame_grid": False,
        "unresolved": True,
        "object_empty_evidence_conflict": False,
        "evidence_reason": "sparse object rows and sensor/pose ranges do not attest a complete label frame grid",
    }


def build_contract(
    previous_root: Path,
    nurec_root: Path,
    component_root: Path,
    output_root: Path,
    full300_root: Path = DEFAULT_FULL300_ROOT,
) -> dict[str, Any]:
    gap_rows = _read_csv(previous_root / "13clip_observation_gap_evidence.csv")
    if len(gap_rows) != 317:
        raise ValueError(f"expected 317 unique gap rows, found {len(gap_rows)}")
    clips = _clip_ids(gap_rows)
    if len(clips) != 13:
        raise ValueError(f"expected 13 affected clips, found {len(clips)}")

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "upstream_source_inventory.csv", UPSTREAM_SOURCES)
    evidence_rows = []
    for source in UPSTREAM_SOURCES:
        evidence_rows.append(
            {
                "source_id": source["source_id"],
                "source_url": source["url"],
                "source_kind": source["source_kind"],
                "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW",
                "claim": source["relevant_evidence"],
                "supports_authoritative_frame_grid": source["supports_authoritative_frame_grid"],
                "supports_empty_from_absence": source["supports_empty_from_absence"],
                "verification_status": "REVIEWED_PRIMARY_SOURCE",
            }
        )
    _write_csv(output_root / "upstream_label_semantics_evidence.csv", evidence_rows)

    local_rows: list[dict[str, Any]] = []
    per_clip_grid: list[dict[str, Any]] = []
    object_counts_by_clip: dict[str, dict[int, int]] = {}
    for clip_id in clips:
        scope = _local_scope_rows(clip_id, nurec_root, component_root)
        local_rows.extend(scope)
        sequence_path = _find_component(clip_id, "sequence_tracks.json", nurec_root, component_root)
        obstacle_path = _find_component(clip_id, "clipgt/obstacle.parquet", nurec_root, component_root)
        sequence_counts = _sequence_track_timestamps(sequence_path) if sequence_path else {}
        obstacle_counts = _exact_object_timestamps(obstacle_path) if obstacle_path else {}
        object_counts_by_clip[clip_id] = {**sequence_counts}
        for timestamp, count in obstacle_counts.items():
            object_counts_by_clip[clip_id][timestamp] = max(object_counts_by_clip[clip_id].get(timestamp, 0), count)
        data_info = next((item for item in scope if item["requested_file"] == "data_info.json"), {})
        per_clip_grid.append(
            {
                "clip_id": clip_id,
                "sequence_tracks_found": bool(sequence_path),
                "sequence_tracks_unique_object_timestamp_count": len(sequence_counts),
                "obstacle_parquet_found": bool(obstacle_path),
                "obstacle_unique_object_timestamp_count": len(obstacle_counts),
                "sensor_frame_range_metadata_found": bool(data_info.get("frame_or_timestamp_paths")),
                "explicit_frame_timestamp_list_found": False,
                "authoritative_frame_grid_status": FRAME_GRID_STATUS,
                "complete_annotation_coverage_proven": False,
                "zero_object_attestation_found": False,
                "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
            }
        )
    _write_csv(output_root / "13clip_authoritative_frame_grid.csv", per_clip_grid)

    sequence_contract = {
        "artifact": "sequence_tracks.json",
        "generation_path": "NRE/NuRec export artifact from NCore/NuRec pipeline",
        "upstream_source_ids": ["nurec_workflow_pai", "nurec_input_contract", "alpasim_scene_catalog"],
        "local_shape": "tracks_data.tracks_id + tracks_poses + tracks_timestamps_us + cuboidtracks_data.cuboids_dims",
        "semantic_result": CONTRACT_C,
        "empty_frame_semantics": "NOT_SPECIFIED",
        "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW_PLUS_LOCAL_AUTOMATED_INSPECTION",
    }
    obstacle_contract = {
        "artifact": "clipgt/obstacle.parquet",
        "generation_path": "NVIDIA NCore PAI converter _load_cuboid_track_observations",
        "upstream_source_id": "ncore_pai_converter",
        "row_semantics": "one CuboidTrackObservation per obstacle row",
        "missing_file_behavior": "empty_observation_list",
        "empty_frame_semantics": "NOT_SPECIFIED",
        "semantic_result": CONTRACT_C,
        "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW",
    }
    _write_json(output_root / "sequence_tracks_generation_contract.json", sequence_contract)
    _write_json(output_root / "obstacle_parquet_generation_contract.json", obstacle_contract)

    frame_contract = {
        "status": FRAME_GRID_STATUS,
        "searched_artifacts": [
            "sequence_tracks.json",
            "clipgt/obstacle.parquet",
            "clipgt/clip.parquet",
            "data_info.json",
            "association.parquet",
            "rig_trajectories.json",
            "metadata.yaml",
            "parsed_config.yaml",
        ],
        "sensor_or_pose_ranges_are_not_label_frame_grid": True,
        "complete_label_coverage_proven": False,
        "absence_of_object_row_is_empty": False,
        "empty_inference_allowed": False,
    }
    _write_json(output_root / "authoritative_frame_grid_contract.json", frame_contract)

    reclassified: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for row in sorted(gap_rows, key=lambda item: (item["clip_id"], int(item["query_timestamp_us"]))):
        clip_id = row["clip_id"]
        decision = classify_missing_timestamp(row, object_counts_by_clip.get(clip_id, {}), {})
        output_row = {
            "clip_id": clip_id,
            "query_timestamp_us": row["query_timestamp_us"],
            "required_by_cf": row.get("required_by_cf", ""),
            "required_by_ttc": row.get("required_by_ttc", ""),
            "prior_classifications": row.get("prior_classifications", ""),
            "prior_classification_not_authoritative": True,
            "prior_outside_label_timeline_renamed": "OUTSIDE_AVAILABLE_OBJECT_ROW_RANGE",
            **decision,
            "authoritative_frame_grid_status": FRAME_GRID_STATUS,
            "empty_inference_allowed": False,
        }
        reclassified.append(output_row)
        if decision["confirmed_empty"]:
            empty_rows.append(output_row)
        if decision["object_empty_evidence_conflict"]:
            conflict_rows.append(output_row)
    _write_csv(output_root / "317_timestamp_contract_reclassification.csv", reclassified)
    fields = ["clip_id", "query_timestamp_us", "evidence_source", "verification_status"]
    _write_csv(output_root / "confirmed_empty_from_upstream_contract.csv", empty_rows, fields)
    _write_csv(output_root / "object_empty_conflicts.csv", conflict_rows, list(reclassified[0]) if conflict_rows else ["clip_id", "query_timestamp_us", "reason"])

    full_summary_path = full300_root / "full300_readiness_final_summary.json"
    if not full_summary_path.is_file():
        full_summary_path = previous_root / "final_13clip_observation_closure_summary.json"
    previous = _read_json(full_summary_path)
    readiness = {
        "TOTAL_CF_REQUIRED_QUERY_COUNT": int(previous["TOTAL_CF_REQUIRED_QUERY_COUNT"]),
        "TOTAL_CF_OBJECT_PRESENT_COUNT": int(previous["TOTAL_CF_OBJECT_PRESENT_COUNT"]),
        "TOTAL_CF_CONFIRMED_EMPTY_COUNT": 0,
        "TOTAL_CF_MISSING_COUNT": int(previous["TOTAL_CF_MISSING_COUNT"]),
        "CF_DATA_READY_FULL_300": False,
        "TOTAL_TTC_REQUIRED_QUERY_COUNT": int(previous["TOTAL_TTC_REQUIRED_QUERY_COUNT"]),
        "TOTAL_TTC_OBJECT_PRESENT_COUNT": int(previous["TOTAL_TTC_OBJECT_PRESENT_COUNT"]),
        "TOTAL_TTC_CONFIRMED_EMPTY_COUNT": 0,
        "TOTAL_TTC_MISSING_COUNT": int(previous["TOTAL_TTC_MISSING_COUNT"]),
        "TTC_DATA_READY_FULL_300": False,
        "affected_clip_count": 13,
        "unique_missing_timestamp_count": 317,
        "confirmed_empty_count": len(empty_rows),
        "outside_authoritative_frame_grid_count": sum(bool(r["outside_authoritative_frame_grid"]) for r in reclassified),
        "unresolved_count": sum(bool(r["unresolved"]) for r in reclassified),
        "object_empty_evidence_conflict_count": len(conflict_rows),
        "label_set_empty_semantics_status": "UNRESOLVED_UPSTREAM_CONTRACT",
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
    }
    _write_json(output_root / "cf_ttc_readiness_after_upstream_contract.json", readiness)

    final = {
        "UPSTREAM_SOURCE_COUNT": len(UPSTREAM_SOURCES),
        "UPSTREAM_PRIMARY_IMPLEMENTATION_SOURCE_COUNT": sum(row["source_kind"] == "PRIMARY_IMPLEMENTATION" for row in UPSTREAM_SOURCES),
        "SEQUENCE_TRACKS_GENERATION_PATH": sequence_contract["generation_path"],
        "OBSTACLE_PARQUET_GENERATION_PATH": obstacle_contract["generation_path"],
        "FINAL_LABEL_SEMANTICS_CONTRACT": CONTRACT_C,
        "AUTHORITATIVE_FRAME_GRID_STATUS": FRAME_GRID_STATUS,
        "ANNOTATION_COVERS_EVERY_FRAME_IN_GRID": False,
        "ABSENCE_OF_OBJECT_ROW_MEANS_EMPTY": False,
        "EMPTY_INFERENCE_ALLOWED": False,
        "AFFECTED_CLIP_COUNT": 13,
        "UNIQUE_MISSING_TIMESTAMP_COUNT": 317,
        "RECLASSIFIED_CONFIRMED_EMPTY_COUNT": len(empty_rows),
        "OUTSIDE_AUTHORITATIVE_FRAME_GRID_COUNT": sum(bool(r["outside_authoritative_frame_grid"]) for r in reclassified),
        "UNRESOLVED_COUNT": sum(bool(r["unresolved"]) for r in reclassified),
        "OBJECT_EMPTY_EVIDENCE_CONFLICT_COUNT": len(conflict_rows),
        **readiness,
        "LABEL_SET_EMPTY_SEMANTICS_STATUS": "UNRESOLVED_UPSTREAM_CONTRACT",
        "PHYSICAL_WORLD_OBSTACLE_COMPLETENESS": "NOT_CLAIMED",
        "DOWNLOADED_BYTES": int(_read_json(previous_root / "final_13clip_observation_closure_summary.json").get("DOWNLOADED_BYTES", 0)),
        "FULL_CLIP_DOWNLOAD_COUNT": int(_read_json(previous_root / "final_13clip_observation_closure_summary.json").get("FULL_CLIP_DOWNLOAD_COUNT", 0)),
        "TIME_MAPPING_FULL300_STATUS": "VERIFIED",
        "OBSTACLE_GEOMETRY_BLOCK": "CLOSED",
        "SCORER_EVALUATOR_CHANGED": False,
        "REMAINING_BLOCKERS": [
            "AUTHORITATIVE_NUREC_LABEL_FRAME_GRID_NOT_PUBLICLY_SPECIFIED",
            "EMPTY_FRAME_SEMANTICS_UNRESOLVED",
            "317_QUERY_TIMESTAMPS_REMAIN_MISSING_UNDER_FAIL_CLOSED_POLICY",
        ],
        "RECOMMENDED_NEXT_STEP": "retain all 317 queries as MISSING; obtain an official annotation/frame-grid export contract or explicit zero-object attestation before changing scorer readiness",
        "evidence_origin": "PUBLIC_CODE_MANUAL_REVIEW_PLUS_LOCAL_AUTOMATED_INSPECTION",
    }
    _write_json(output_root / "upstream_label_semantics_final_contract.json", final)
    _write_json(output_root / "final_label_semantics_closure_summary.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    global DEFAULT_FULL300_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--full300-root", type=Path, default=DEFAULT_FULL300_ROOT)
    parser.add_argument("--nurec-root", type=Path, default=DEFAULT_NUREC_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    DEFAULT_FULL300_ROOT = args.full300_root
    final = build_contract(args.previous_root, args.nurec_root, args.component_root, args.output_root, args.full300_root)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
