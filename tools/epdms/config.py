"""Configuration loading and validation for EPDMS evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import VehicleParameters


class EvaluationConfig:
    def __init__(self, raw: Dict[str, Any], config_path: Optional[Path] = None):
        self.raw = raw
        self.config_path = config_path

        self.metric_profile: str = raw.get("metric_profile", "nurec_safety_proxy_v1")
        self.horizon_s: float = float(raw.get("horizon_s", 4.0))
        self.frequency_hz: float = float(raw.get("frequency_hz", 10.0))
        self.dt_s: float = 1.0 / self.frequency_hz
        self.future_poses: int = int(round(self.horizon_s * self.frequency_hz))
        self.strict_mode: bool = bool(raw.get("strict_mode", True))
        self.resume: bool = bool(raw.get("resume", True))
        self.random_seed: int = int(raw.get("random_seed", 2026))

        self.alphas: List[float] = [float(a) for a in raw.get("alphas", [0.0, 0.5, 1.0, 2.0])]
        self.modes: List[str] = list(raw.get("modes", ["cross_scene", "no_reasoning", "noisy", "opposite_action"]))

        veh_dict = raw.get("vehicle", {})
        self.vehicle = VehicleParameters(
            reference_point=veh_dict.get("reference_point", "rear_axle"),
            front_length_m=float(veh_dict.get("front_length_m", 4.049)),
            rear_length_m=float(veh_dict.get("rear_length_m", 1.127)),
            width_m=float(veh_dict.get("width_m", 2.297)),
        )

        proxy_dict = raw.get("proxy", {})
        self.touch_is_collision: bool = bool(proxy_dict.get("touch_is_collision", True))
        self.ttc_horizon_s: float = float(proxy_dict.get("ttc_horizon_s", 1.0))
        self.progress_stationary_threshold_m: float = float(proxy_dict.get("progress_stationary_threshold_m", 5.0))
        self.practical_score_delta: float = float(proxy_dict.get("practical_score_delta", 0.01))
        self.practical_ade_delta_m: float = float(proxy_dict.get("practical_ade_delta_m", 0.05))

        paths_dict = raw.get("paths", {})
        self.prediction_jsonl = Path(paths_dict.get("prediction_jsonl", ""))
        self.context_jsonl = Path(paths_dict.get("context_jsonl", ""))
        self.context_filtered_dir = Path(paths_dict.get("context_filtered_dir", ""))
        self.ground_truth_jsonl = Path(paths_dict.get("ground_truth_jsonl", ""))
        self.score_dir = Path(paths_dict.get("score_dir", ""))
        self.analysis_dir = Path(paths_dict.get("analysis_dir", ""))

        self.sha256 = self._compute_sha256()

    def _compute_sha256(self) -> str:
        s = json.dumps(self.raw, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> EvaluationConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw, config_path=p)
