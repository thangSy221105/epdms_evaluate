"""Partial EPDMS-style metric vector with fail-closed official fields."""

from __future__ import annotations

from typing import Any, Mapping, Optional


OFFICIAL_COMPONENTS = ("nc", "dac", "ddc", "tlc", "ttc", "ep", "lk", "hc", "ec")


def build_partial_metric_vector(
    record: Mapping[str, Any],
    *,
    lk_result: Mapping[str, Any],
    ddc_result: Optional[Mapping[str, Any]] = None,
    lane_direction_diagnostic: Optional[Mapping[str, Any]] = None,
    gt_lk_result: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build additive output; it never changes or reinterprets old records."""

    ddc_result = ddc_result or {}
    vector: dict[str, Any] = {
        "record_key": record.get("record_key"),
        "clip_id": record.get("clip_id"),
        "mode": record.get("mode"),
        "alpha": record.get("alpha"),
        "metric_profile": "epdms_partial_vector_lk_ddc_v1",
    }
    vector.update({component: None for component in OFFICIAL_COMPONENTS})
    vector.update({
        "collision_free_proxy": record.get("collision_free_proxy"),
        "dac_proxy": record.get("dac_proxy"),
        "ttc_proxy": record.get("ttc_proxy"),
        "progress_gt_proxy": record.get("progress_gt_proxy"),
        "future_comfort_proxy": record.get("future_comfort_proxy"),
        "lk_proxy": lk_result.get("lk_proxy"),
        "lk_proxy_status": lk_result.get("status"),
        "lk_proxy_max_lateral_deviation_m": lk_result.get("max_lateral_deviation_m"),
        "lk_proxy_violation_frame_count": lk_result.get("violation_frame_count"),
        "lk_proxy_max_continuous_violation_s": lk_result.get("max_continuous_violation_s"),
        "lk_proxy_associated_lane_ids": lk_result.get("associated_lane_ids", []),
        "ddc_proxy": ddc_result.get("ddc_proxy"),
        "ddc_proxy_status": ddc_result.get("status", "LANE_DIRECTION_UNRESOLVED"),
        "lane_direction_alignment_diagnostic": lane_direction_diagnostic or {},
        "gt_lk_proxy": gt_lk_result.get("lk_proxy") if gt_lk_result else None,
        "gt_max_lateral_deviation_m": gt_lk_result.get("max_lateral_deviation_m") if gt_lk_result else None,
        "gt_max_continuous_violation_s": gt_lk_result.get("max_continuous_violation_s") if gt_lk_result else None,
        "official_epdms_stage1": None,
        "official_epdms_stage1_status": "NOT_READY",
    })
    vector["missing_components"] = list(OFFICIAL_COMPONENTS)
    return vector
