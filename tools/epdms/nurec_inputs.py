"""Read-only NuRec input contracts used by the production proxy adapter.

The adapter decorates in-memory rows. It never edits raw prediction, GT, or
NuRec files and never derives a clock offset or geometric correction.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ObservationReadiness:
    clip_id: str
    cf_required: int
    cf_object_present: int
    cf_confirmed_empty: int
    cf_missing: int
    ttc_required: int
    ttc_object_present: int
    ttc_confirmed_empty: int
    ttc_missing: int

    @property
    def cf_ready(self) -> bool:
        return self.cf_missing == 0

    @property
    def ttc_ready(self) -> bool:
        return self.ttc_missing == 0

    @property
    def observation_ready(self) -> bool:
        return self.cf_ready and self.ttc_ready


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_observation_readiness(path: Path) -> Dict[str, ObservationReadiness]:
    result: Dict[str, ObservationReadiness] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            def integer(name: str) -> int:
                return int(row.get(name) or 0)
            clip_id = str(row["clip_id"])
            result[clip_id] = ObservationReadiness(
                clip_id=clip_id,
                cf_required=integer("cf_required_query_count"),
                cf_object_present=integer("cf_object_present_count"),
                cf_confirmed_empty=integer("cf_confirmed_empty_count"),
                cf_missing=integer("cf_missing_count"),
                ttc_required=integer("ttc_required_query_count"),
                ttc_object_present=integer("ttc_object_present_count"),
                ttc_confirmed_empty=integer("ttc_confirmed_empty_count"),
                ttc_missing=integer("ttc_missing_count"),
            )
    return result


def load_time_mapping(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        normalized = dict(row)
        if "verified" not in normalized and "time_mapping_verified" in normalized:
            normalized["verified"] = bool(normalized["time_mapping_verified"])
        result[str(normalized["clip_id"])] = normalized
    return result


def load_dac_readiness(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a generated DAC readiness CSV without embedding map policy."""
    result: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[str(row["clip_id"])] = row
    return result


def load_coordinate_contract(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("status") != "VERIFIED_FROZEN_CONTRACT":
        raise ValueError("coordinate contract is not the accepted frozen contract")
    if contract.get("world_to_nre_applied_to_ego") is not False:
        raise ValueError("world_to_nre_applied_to_ego must remain false")
    if contract.get("time_mapping", {}).get("per_clip_offset_rederived") is not False:
        raise ValueError("per-clip offsets must be reused, never rederived")
    return contract


def decorate_inputs(
    pred_row: Dict[str, Any],
    gt_row: Dict[str, Any],
    context_row: Dict[str, Any],
    contract: Dict[str, Any],
    time_record: Dict[str, Any],
    readiness: ObservationReadiness,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return deep-copied rows with explicit, accepted provenance metadata."""
    pred = decorate_prediction_gt(pred_row, contract, "prediction")
    gt = decorate_prediction_gt(gt_row, contract, "ground_truth")
    context = decorate_context(context_row, contract, time_record, readiness)
    return pred, gt, context


def decorate_prediction_gt(row: Dict[str, Any], contract: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Decorate one prediction/GT row without copying the large obstacle context."""

    result = copy.deepcopy(row)
    common = str(contract["common_evaluation_frame"])
    result.update({
        "coordinate_frame": common,
        "reference_point": contract["anchor_semantics"][role],
        "source_includes_t0": False,
        "trajectory_origin_policy": "future_only",
        "coordinate_contract": {"status": contract["status"], "source": "accepted_frozen_contract"},
    })
    return result


def decorate_context(
    context_row: Dict[str, Any],
    contract: Dict[str, Any],
    time_record: Dict[str, Any],
    readiness: ObservationReadiness,
) -> Dict[str, Any]:
    """Decorate a context once per clip; the scorer treats it as read-only."""

    context = copy.deepcopy(context_row)
    common = str(contract["common_evaluation_frame"])
    context["coordinate_contract"] = {
        "status": contract["status"],
        "source": "accepted_frozen_contract",
        "common_frame": common,
        "obstacle_source_frame": contract["obstacle_source_frame"],
        "map_source_frame": contract["map_source_frame"],
        "obstacle_transform_status": contract["obstacle_transform_status"],
        "world_to_nre_applied_to_ego": False,
        "anchor_alignment_status": contract["anchor_semantics"]["alignment_status"],
    }
    context["coordinate_frame"] = common
    context["obstacle_frame"] = common
    context["map_frame"] = common
    context["obstacle_anchor"] = contract["anchor_semantics"]["obstacle"]
    context["map_anchor"] = contract["anchor_semantics"]["map"]
    context["coordinate_alignment_verified"] = True
    context["coordinate_alignment_status"] = "VERIFIED_FROZEN_CONTRACT"
    context["time_mapping_status"] = "VERIFIED_REUSED_EXISTING"
    context["time_mapping_source"] = time_record.get("source")
    context["per_clip_offset_rederived"] = False
    # The scorer consumes the prediction/GT clip-relative clock.  Reuse the
    # accepted NuRec = PAI + offset contract to express obstacle timestamps on
    # that same clock.  This is a deterministic timeline conversion, not a
    # newly estimated offset.
    offset_us = int(time_record["offset_us"])
    obstacle_block = context.get("semantic_context", {}).get("obstacle", {})
    if isinstance(obstacle_block, dict):
        for obstacle in obstacle_block.get("all_obstacles", []) or []:
            if isinstance(obstacle, dict) and obstacle.get("timestamp_micros") is not None:
                obstacle["timestamp_micros"] = int(obstacle["timestamp_micros"]) - offset_us
        for key in ("confirmed_empty_timestamps_us", "observed_empty_timestamps_us"):
            values = obstacle_block.get(key)
            if isinstance(values, list):
                obstacle_block[key] = [int(value) - offset_us for value in values]
        obstacle_block["timestamp_mapping"] = {
            "source_clock": "NUREC_GLOBAL",
            "target_clock": "PAI_CLIP_RELATIVE",
            "offset_us": offset_us,
            "verified": bool(time_record.get("verified", time_record.get("time_mapping_verified", False))),
            "rederived": False,
        }
    context["observation_readiness"] = {
        "cf_missing": readiness.cf_missing,
        "ttc_missing": readiness.ttc_missing,
        "cf_ready": readiness.cf_ready,
        "ttc_ready": readiness.ttc_ready,
        "observation_ready": readiness.observation_ready,
        "source": "full300_readiness_audit",
    }
    return context
