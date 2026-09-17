"""Data schemas and types for EPDMS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VehicleParameters:
    reference_point: str = "rear_axle"
    front_length_m: float = 4.049
    rear_length_m: float = 1.127
    width_m: float = 2.297

    @property
    def length_m(self) -> float:
        return self.front_length_m + self.rear_length_m

    @property
    def rear_axle_to_center_m(self) -> float:
        return (self.front_length_m - self.rear_length_m) / 2.0


@dataclass
class Waypoint:
    t_s: float
    x_m: float
    y_m: float
    heading_rad: float = 0.0
    velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    curvature_inv_m: float = 0.0


@dataclass
class ObstacleBox:
    timestamp_micros: int
    trackline_id: str
    category: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw_rad: float
    valid: bool = True


@dataclass
class EvaluationScoreRecord:
    record_key: str
    clip_id: str
    mode: str
    alpha: float
    rule_group: Optional[str] = None
    metric_profile: str = "nurec_safety_proxy_v1"
    horizon_s: float = 4.0
    frequency_hz: float = 10.0
    trajectory_hash: str = ""
    is_shared_baseline: bool = False
    valid: bool = True
    failure_stage: Optional[str] = None
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    warning_codes: List[str] = field(default_factory=list)

    # Core proxy metrics
    collision_free_proxy: Optional[float] = None
    dac_proxy: Optional[float] = None
    ttc_proxy: Optional[float] = None
    progress_gt_proxy: Optional[float] = None
    future_comfort_proxy: Optional[float] = None
    nurec_safety_proxy_v1: Optional[float] = None

    # Detailed diagnostics
    first_collision_time_s: Optional[float] = None
    collided_track_ids: List[str] = field(default_factory=list)
    collided_object_types: List[str] = field(default_factory=list)
    first_offroad_time_s: Optional[float] = None
    offroad_frame_count: int = 0
    min_ttc_s: Optional[float] = None
    ttc_failure_time_s: Optional[float] = None
    ttc_track_id: Optional[str] = None
    minimum_clearance_m: Optional[float] = None
    gt_progress_m: Optional[float] = None
    pred_projected_progress_m: Optional[float] = None
    endpoint_displacement_error_m: Optional[float] = None
    ade_m: Optional[float] = None
    fde_m: Optional[float] = None

    # Kinematics
    max_abs_longitudinal_accel: Optional[float] = None
    max_abs_lateral_accel: Optional[float] = None
    max_jerk_magnitude: Optional[float] = None
    max_abs_longitudinal_jerk: Optional[float] = None
    max_abs_yaw_rate: Optional[float] = None
    max_abs_yaw_acceleration: Optional[float] = None
    comfort_failure_reasons: List[str] = field(default_factory=list)

    # Metadata & Hash
    runtime_ms: float = 0.0
    config_sha256: str = ""
    source_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_key": self.record_key,
            "clip_id": self.clip_id,
            "mode": self.mode,
            "alpha": self.alpha,
            "rule_group": self.rule_group,
            "metric_profile": self.metric_profile,
            "horizon_s": self.horizon_s,
            "frequency_hz": self.frequency_hz,
            "trajectory_hash": self.trajectory_hash,
            "is_shared_baseline": self.is_shared_baseline,
            "valid": self.valid,
            "failure_stage": self.failure_stage,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "warning_codes": self.warning_codes,
            "collision_free_proxy": self.collision_free_proxy,
            "dac_proxy": self.dac_proxy,
            "ttc_proxy": self.ttc_proxy,
            "progress_gt_proxy": self.progress_gt_proxy,
            "future_comfort_proxy": self.future_comfort_proxy,
            "nurec_safety_proxy_v1": self.nurec_safety_proxy_v1,
            "first_collision_time_s": self.first_collision_time_s,
            "collided_track_ids": self.collided_track_ids,
            "collided_object_types": self.collided_object_types,
            "first_offroad_time_s": self.first_offroad_time_s,
            "offroad_frame_count": self.offroad_frame_count,
            "min_ttc_s": self.min_ttc_s,
            "ttc_failure_time_s": self.ttc_failure_time_s,
            "ttc_track_id": self.ttc_track_id,
            "minimum_clearance_m": self.minimum_clearance_m,
            "gt_progress_m": self.gt_progress_m,
            "pred_projected_progress_m": self.pred_projected_progress_m,
            "endpoint_displacement_error_m": self.endpoint_displacement_error_m,
            "ade_m": self.ade_m,
            "fde_m": self.fde_m,
            "max_abs_longitudinal_accel": self.max_abs_longitudinal_accel,
            "max_abs_lateral_accel": self.max_abs_lateral_accel,
            "max_jerk_magnitude": self.max_jerk_magnitude,
            "max_abs_longitudinal_jerk": self.max_abs_longitudinal_jerk,
            "max_abs_yaw_rate": self.max_abs_yaw_rate,
            "max_abs_yaw_acceleration": self.max_abs_yaw_acceleration,
            "comfort_failure_reasons": self.comfort_failure_reasons,
            "runtime_ms": self.runtime_ms,
            "config_sha256": self.config_sha256,
            "source_sha256": self.source_sha256,
        }
