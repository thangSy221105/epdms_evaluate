#!/usr/bin/env python3
"""Build the additive XODR DDC profile without rerunning the scorer.

The current NuRec XODR evidence contains map geometry and a geoReference, but
does not establish an authoritative XODR-to-NuRec scene transform or legal
direction binding.  This script therefore produces a complete, reproducible
fail-closed profile: it preserves the existing 4,800-row partial vector and
keeps both official ``ddc`` and ``ddc_proxy`` null.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


FAIL_STATUS = "XODR_DIRECTION_OR_COORDINATE_CONTRACT_UNRESOLVED"
PROFILE = "epdms_partial_vector_lk_ddc_xodr_v2"
OFFICIAL_COMPONENTS = ("nc", "dac", "ddc", "tlc", "ttc", "ep", "lk", "hc", "ec")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_identity(records: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    record_keys = [str(row.get("record_key")) for row in records]
    baseline_keys = [str(row.get("record_key")) for row in baseline]
    return {
        "records_count": len(records),
        "baseline_count": len(baseline),
        "records_unique": len(record_keys) == len(set(record_keys)),
        "baseline_unique": len(baseline_keys) == len(set(baseline_keys)),
        "same_record_key_set": set(record_keys) == set(baseline_keys),
        "missing_from_baseline": sorted(set(record_keys) - set(baseline_keys)),
        "extra_in_baseline": sorted(set(baseline_keys) - set(record_keys)),
    }


def build_fail_closed_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["metric_profile"] = PROFILE
    result["ddc"] = None
    result["ddc_proxy"] = None
    result["ddc_proxy_status"] = FAIL_STATUS
    result["ddc_proxy_implemented"] = False
    result["ddc_proxy_enabled"] = False
    result["xodr_direction_contract_status"] = "UNRESOLVED"
    result["xodr_coordinate_contract_status"] = "UNRESOLVED"
    result["missing_components"] = list(dict.fromkeys([*(row.get("missing_components") or []), *OFFICIAL_COMPONENTS]))
    return result


def _xodr_rows(schema_rows: list[dict[str, str]], member_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    members = {row.get("clip_id"): row for row in member_rows}
    geometry: list[dict[str, Any]] = []
    direction: list[dict[str, Any]] = []
    coordinate: list[dict[str, Any]] = []
    for row in schema_rows:
        clip_id = row.get("clip_id", "")
        member = members.get(clip_id, {})
        geometry.append({
            "clip_id": clip_id,
            "map_xodr_present": member.get("xodr_member_found") == "True",
            "parse_status": row.get("parse_status"),
            "road_count": row.get("road_count"),
            "junction_count": row.get("junction_count"),
            "geometry_count": row.get("geometry_count"),
            "geometry_types": row.get("geometry_types"),
            "geometry_status": "AVAILABLE" if row.get("parse_status") == "PARSED" and int(row.get("geometry_count") or 0) > 0 else "UNRESOLVED",
        })
        direction.append({
            "clip_id": clip_id,
            "lane_direction_attr_count": row.get("lane_direction_attr_count"),
            "lane_direction_values": row.get("lane_direction_values"),
            "road_rule_values": row.get("road_rule_values"),
            "lane_id_positive_count": row.get("lane_id_positive_count"),
            "lane_id_negative_count": row.get("lane_id_negative_count"),
            "direction_semantics_status": "UNRESOLVED_NO_EXPLICIT_DIRECTION_BINDING",
            "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
        })
        coordinate.append({
            "clip_id": clip_id,
            "geo_reference_present": row.get("geo_reference_present"),
            "geo_reference_sha256": row.get("geo_reference_sha256"),
            "xodr_coordinate_frame": "GEOREFERENCED_OPENDRIVE_UNBOUND",
            "nurec_coordinate_frame": "NCORE_LOCAL_WORLD",
            "transform_source": "",
            "coordinate_validation_status": "UNRESOLVED_NO_EXPLICIT_XODR_TO_NUREC_TRANSFORM",
            "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
        })
    return geometry, direction, coordinate


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.records_jsonl)
    baseline = read_jsonl(args.baseline_vector_jsonl)
    identity = validate_identity(records, baseline)
    if not identity["same_record_key_set"] or not identity["records_unique"] or not identity["baseline_unique"]:
        raise ValueError(f"4,800 identity contract failed: {identity}")

    schema_rows = read_csv(args.xodr_dir / "xodr_schema_inventory.csv")
    member_rows = read_csv(args.xodr_dir / "xodr_member_inventory_full300.csv")
    geometry_rows, direction_rows, coordinate_rows = _xodr_rows(schema_rows, member_rows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "xodr_geometry_inventory.csv", geometry_rows)
    write_csv(output / "xodr_direction_semantics_evidence.csv", direction_rows)
    write_csv(output / "xodr_coordinate_validation.csv", coordinate_rows)
    write_csv(output / "xodr_lane_direction_validation.csv", direction_rows)
    # Keep the XODR inventory self-contained as well as colocated with the
    # successor vector.  The source recovery inventory remains untouched.
    write_csv(args.xodr_dir / "xodr_geometry_inventory.csv", geometry_rows)
    write_csv(args.xodr_dir / "xodr_direction_semantics_evidence.csv", direction_rows)
    write_csv(args.xodr_dir / "xodr_coordinate_validation.csv", coordinate_rows)
    write_csv(args.xodr_dir / "xodr_lane_direction_validation.csv", direction_rows)
    write_csv(args.xodr_dir / "ddc_readiness_full300.csv", [{
        "clip_id": row.get("clip_id"),
        "xodr_geometry_available": row.get("geometry_status") == "AVAILABLE",
        "legal_direction_verified": False,
        "xodr_to_nurec_coordinate_verified": False,
        "ddc_proxy_ready": False,
        "status": FAIL_STATUS,
    } for row in geometry_rows])

    vector_rows = [build_fail_closed_row(row) for row in baseline]
    write_jsonl(output / "partial_metric_vector_full4800.jsonl", vector_rows)
    validation = [{
        "record_key": row.get("record_key"),
        "clip_id": row.get("clip_id"),
        "mode": row.get("mode"),
        "alpha": row.get("alpha"),
        "ddc_proxy": None,
        "ddc_proxy_status": FAIL_STATUS,
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
    } for row in vector_rows]
    write_csv(output / "ddc_proxy_validation.csv", validation)
    write_csv(output / "gt_ddc_validation.csv", [{
        "clip_id": row.get("clip_id"),
        "gt_ddc": None,
        "status": "NOT_RUN_FAIL_CLOSED_XODR_CONTRACT",
        "physical_world_completeness_claim": "NOT_CLAIMED",
    } for row in geometry_rows])
    write_csv(output / "ddc_vs_lane_alignment_diagnostic.csv", [{
        "clip_id": row.get("clip_id"),
        "ddc_proxy": None,
        "lane_direction_comparison": "NOT_RUN",
        "status": FAIL_STATUS,
    } for row in geometry_rows])

    ddc_summary = {
        "metric_profile": PROFILE,
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "official_ddc": None,
        "official_ddc_in_missing_components": True,
        "partial_metric_vector_record_count": len(vector_rows),
        "ddc_proxy_valid_count": 0,
        "ddc_proxy_null_count": len(vector_rows),
        "xodr_clip_count": len(schema_rows),
        "xodr_geometry_verified_count": sum(row["geometry_status"] == "AVAILABLE" for row in geometry_rows),
        "xodr_direction_verified_count": 0,
        "xodr_coordinate_verified_count": 0,
        "identity_contract": identity,
        "fail_closed_reason": "XODR_DIRECTION_AND_COORDINATE_CONTRACT_NOT_VERIFIED",
    }
    write_json(output / "partial_metric_vector_summary.json", ddc_summary)
    write_json(output / "ddc_mode_alpha_summary.json", {
        "metric_profile": PROFILE,
        "modes": sorted({str(row.get("mode")) for row in vector_rows}),
        "alphas": sorted({row.get("alpha") for row in vector_rows}),
        "ddc_proxy_enabled": False,
        "ddc_proxy": None,
    })
    contract = {
        "metric_profile": PROFILE,
        "xodr_member_status": "300_OF_300_FOUND_AND_PARSED",
        "xodr_direction_semantics_status": "UNRESOLVED",
        "xodr_to_nurec_coordinate_status": "UNRESOLVED",
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "ddc_proxy": None,
        "official_ddc": None,
        "missing_components": ["ddc"],
        "physical_world_obstacle_completeness": "NOT_CLAIMED",
        "evidence_origin": "LOCAL_AUTOMATED_INSPECTION",
        "fail_closed_reason": "XODR_DIRECTION_AND_COORDINATE_CONTRACT_NOT_VERIFIED",
    }
    write_json(output / "xodr_direction_contract.json", contract)
    final = {
        "metric_profile": PROFILE,
        "source_records": str(args.records_jsonl),
        "baseline_vector": str(args.baseline_vector_jsonl),
        "output_vector": str(output / "partial_metric_vector_full4800.jsonl"),
        "record_count": len(vector_rows),
        "identity_contract": identity,
        "ddc_proxy_implemented": False,
        "ddc_proxy_enabled": False,
        "ddc_proxy_status": "FAIL_CLOSED",
        "official_ddc": None,
        "ddc_in_missing_components": True,
        "xodr_direction_contract_status": "UNRESOLVED",
        "xodr_coordinate_contract_status": "UNRESOLVED",
        "recommended_next_step": "Obtain explicit XODR-to-NCORE_LOCAL_WORLD binding and legal-direction provenance before implementing DDC.",
    }
    write_json(output / "ddc_proxy_final_summary.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-vector-jsonl", type=Path, required=True)
    parser.add_argument("--xodr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
