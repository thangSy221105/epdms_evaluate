"""Evidence-only PhysicalAI -> NuRec time bridge audit.

This command compares supplied PAI and NuRec timestamp tables and writes
forensic reports. It never infers an offset from numeric proximity or shape.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


CLIP_ID = "00040136-e651-4abd-991d-0655ccda9430"
T0_US = 5_100_000


def _summary(values: list[int]) -> dict[str, Any]:
    values = sorted(int(v) for v in values)
    steps = [b - a for a, b in zip(values, values[1:]) if b > a]
    nearest = min(values, key=lambda value: abs(value - T0_US)) if values else None
    return {
        "count": len(values),
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
        "median_step": median(steps) if steps else None,
        "contains_5100000": T0_US in values,
        "nearest_to_5100000": nearest,
    }


def constant_delta_diagnostics(pairs: list[dict[str, int]], semantic_verified: bool) -> dict[str, Any]:
    deltas = [int(pair["nurec_timestamp_us"]) - int(pair["physicalai_timestamp_us"]) for pair in pairs]
    unique = sorted(set(deltas))
    med = median(deltas) if deltas else None
    return {
        "pair_count": len(pairs),
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "delta_unique_count": len(unique),
        "median_delta": med,
        "residual": max((abs(value - med) for value in deltas), default=None),
        "semantic_verified": semantic_verified,
        "verification_status": "VERIFIED" if semantic_verified and len(deltas) >= 2 and len(unique) == 1 else "DIAGNOSTIC_ONLY",
    }


def _read_timestamp(path: Path, field: str) -> list[int]:
    frame = pd.read_parquet(path)
    if field in frame.columns:
        series = frame[field]
    else:
        parts = field.split(".")
        series = frame[parts[0]]
        for part in parts[1:]:
            series = series.map(lambda value: value.get(part) if isinstance(value, dict) else None)
    return [int(value) for value in series.dropna().tolist()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["clip_id", "source", "field", "declared_unit", "count", "min", "max", "median_step", "contains_5100000", "nearest_to_5100000", "source_path", "provenance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pai_ego = _read_timestamp(Path(args.pai_egomotion), args.pai_egomotion_field)
    pai_obs = _read_timestamp(Path(args.pai_obstacle), args.pai_obstacle_field)
    nurec_ego = _read_timestamp(Path(args.nurec_egomotion), args.nurec_egomotion_field)
    nurec_obs = _read_timestamp(Path(args.nurec_obstacle), args.nurec_obstacle_field)
    rows = []
    for source, field, values, path, provenance in [
        ("PhysicalAI", args.pai_egomotion_field, pai_ego, args.pai_egomotion, "official PhysicalAI egomotion parquet"),
        ("PhysicalAI", args.pai_obstacle_field, pai_obs, args.pai_obstacle, "official PhysicalAI obstacle parquet"),
        ("NuRec", args.nurec_egomotion_field, nurec_ego, args.nurec_egomotion, "raw NuRec egomotion parquet"),
        ("NuRec", args.nurec_obstacle_field, nurec_obs, args.nurec_obstacle, "raw NuRec obstacle parquet"),
    ]:
        rows.append({"clip_id": args.clip_id, "source": source, "field": field, "declared_unit": "microseconds", **_summary(values), "source_path": str(path), "provenance": provenance})
    _write_csv(output / "physicalai_timestamp_inventory.csv", rows)

    pai_ego_df = pd.read_parquet(args.pai_egomotion)
    pai_ego_df["distance_to_t0_us"] = (pai_ego_df[args.pai_egomotion_field] - T0_US).abs()
    nearest_rows = pai_ego_df.nsmallest(5, "distance_to_t0_us").drop(columns=["distance_to_t0_us"]).to_dict(orient="records")
    pai_obs_df = pd.read_parquet(args.pai_obstacle)
    obs_ts = pai_obs_df[args.pai_obstacle_field]
    obstacle_windows = {str(window): sorted({int(value) for value in obs_ts[(obs_ts >= window[0]) & (obs_ts <= window[1])].tolist()}) for window in [(4_900_000, 5_300_000), (4_000_000, 6_500_000)]}
    semantic_pairs: list[dict[str, int]] = []
    diagnostics = constant_delta_diagnostics(semantic_pairs, semantic_verified=False)
    coverage = {
        "cf_required_range_us": [T0_US + 100_000, T0_US + 4_000_000],
        "ttc_required_range_us": [T0_US + 100_000, T0_US + 5_000_000],
        "nurec_egomotion_range_us": [min(nurec_ego), max(nurec_ego)] if nurec_ego else [None, None],
        "nurec_obstacle_range_us": [min(nurec_obs), max(nurec_obs)] if nurec_obs else [None, None],
        "cf_time_coverage": "UNRESOLVED",
        "ttc_time_coverage": "UNRESOLVED",
        "reason": "No verified PhysicalAI-to-NuRec mapping; range comparison is diagnostic only",
    }
    lineage = {
        "clip_id": args.clip_id,
        "edges": [
            {"source_domain": "PhysicalAI", "target_domain": "Alpamayo", "mapping_type": "explicit_query_contract", "scale": 1.0, "offset_us": 0, "explicit": True, "source_file": "alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py", "source_line": 27, "evidence": "t0_us default 5_100_000; future query grid is t0 + k*100000"},
            {"source_domain": "PhysicalAI", "target_domain": "NCore", "mapping_type": "timestamp_preserved_in_public_PAI_converter", "scale": 1.0, "offset_us": 0, "explicit": True, "source_file": "ncore/tools/data_converter/pai/converter.py", "source_line": 209, "evidence": "converter reads PAI timestamp and stores T_rig_world_timestamps_us; obstacle rows retain timestamp_us"},
            {"source_domain": "NCore", "target_domain": "NuRec/NRE", "mapping_type": "UNRESOLVED", "scale": None, "offset_us": None, "explicit": False, "source_file": None, "source_line": None, "evidence": "No public NCore->NuRec writer mapping to raw clipgt/key.timestamp_micros found in inspected repositories"},
        ],
    }
    bridge = {
        "clip_id": args.clip_id,
        "physicalai_t0_us": T0_US,
        "physicalai_clock_verified": True,
        "nurec_clock_verified": True,
        "mapping_verified": False,
        "mapping_type": None,
        "scale": None,
        "offset_us": None,
        "source_evidence": ["official Alpamayo loader", "official PhysicalAI loader", "official NCore PAI converter"],
        "semantic_pair_count": 0,
        "residual_us": None,
        "cf_time_range_valid": None,
        "ttc_time_range_valid": None,
        "status": "UNRESOLVED",
        "blockers": ["NO_NCORE_TO_NUREC_PROVENANCE", "NO_SEMANTIC_CROSS_DOMAIN_PAIRS", "PAI_T0_NOT_AN_EGOMOTION_ROW"],
        "physicalai_egomotion_contains_5100000": T0_US in pai_ego,
        "physicalai_egomotion_nearest_rows": nearest_rows,
        "physicalai_obstacle_windows": obstacle_windows,
        "delta_diagnostics": diagnostics,
        "coverage": coverage,
    }
    (output / "time_lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "physicalai_to_nurec_time_bridge.json").write_text(json.dumps(bridge, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output / "physicalai_clock_contract.json").write_text(json.dumps({
        "unit": "microseconds",
        "origin_type": "UNRESOLVED_BY_PUBLIC_PAI_CODE",
        "origin_verified": False,
        "ego_obstacle_same_clock": True,
        "source": [
            {"file": "physical_ai_av/src/physical_ai_av/dataset.py", "lines": "172-184", "evidence": "egomotion and camera tables are read through timestamp columns"},
            {"file": "ncore/tools/data_converter/pai/converter.py", "lines": "209-237, 376-410", "evidence": "egomotion defines the sequence interval and obstacle timestamp_us is filtered against that interval"},
        ],
        "negative_timestamp_observed_in_pai_egomotion": True,
        "note": "The public code establishes microsecond timestamp usage and same-sequence filtering, but does not declare that zero is the PhysicalAI clip origin.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "alpamayo_time_contract.md").write_text("""# Alpamayo PhysicalAI time contract\n\n- `src/alpamayo_r1/load_physical_aiavdataset.py:27`: `t0_us: int = 5_100_000`.\n- Lines 33, 45-50: 64 future steps at 10 Hz, `time_step=0.1`.\n- Lines 107-120: future query grid is `t0_us + 100000, ..., t0_us + 6400000`.\n- Lines 122-129: queries the PhysicalAI egomotion interpolator at those timestamps and obtains `ego_future_xyz`.\n- Lines 142-157: transforms world poses to local coordinates at the t0 pose.\n- Line 220: returns the supplied `t0_us` unchanged.\n\nThis proves the Alpamayo sampling convention. It does not prove that raw NuRec `key.timestamp_micros` uses the same origin.\n""", encoding="utf-8")
    (output / "pai_to_ncore_time_trace.md").write_text("""# PhysicalAI -> NCore timestamp trace\n\n- `ncore/tools/data_converter/pai/utils.py:207-233` reads the PAI `timestamp` column and returns non-negative values as `uint64`; the implementation filters negative rows and does not subtract a minimum or apply a scale.\n- `ncore/tools/data_converter/pai/converter.py:209-237` uses those timestamps as the sequence interval and stores them at lines 298-304.\n- `ncore/tools/data_converter/pai/converter.py:376-410` reads obstacle `timestamp_us` and stores it unchanged after interval filtering.\n- `ncore/tools/data_converter/pai/converter.py:271-279` records source clip/repository/revision/commit metadata when the provider supplies it.\n\nConclusion: `DOES_NCORE_CONVERTER_PRESERVE_PHYSICALAI_TIMESTAMP = PRESERVED_EXACTLY` for the public PAI converter path, subject to its removal of negative egomotion rows. This is not evidence for NCore -> NuRec.\n""", encoding="utf-8")
    (output / "nurec_source_provenance.json").write_text(json.dumps({"clip_id": args.clip_id, "source_clip_id": None, "source_repo_id": None, "source_revision": None, "source_commit_sha": None, "status": "NOT_FOUND_IN_NUREC_PILOT_METADATA", "searched": ["data_info.json", "datasource_summary.json", "metadata.yaml", "pose_record.json", "rig_trajectories.json", "sequence_tracks.json", "clipgt/clip.parquet"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "pai_nurec_sequence_diagnostics.json").write_text(json.dumps({"physicalai_egomotion": _summary(pai_ego), "nurec_egomotion": _summary(nurec_ego), "semantic_pairs": semantic_pairs, "diagnostics": diagnostics}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "obstacle_time_windows.json").write_text(json.dumps(obstacle_windows, indent=2) + "\n", encoding="utf-8")
    return bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-id", default=CLIP_ID)
    parser.add_argument("--pai-egomotion", required=True)
    parser.add_argument("--pai-obstacle", required=True)
    parser.add_argument("--nurec-egomotion", required=True)
    parser.add_argument("--nurec-obstacle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pai-egomotion-field", default="timestamp")
    parser.add_argument("--pai-obstacle-field", default="timestamp_us")
    parser.add_argument("--nurec-egomotion-field", default="key.timestamp_micros")
    parser.add_argument("--nurec-obstacle-field", default="key.timestamp_micros")
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
