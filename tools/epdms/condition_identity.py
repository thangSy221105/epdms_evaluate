"""Deterministic condition identity parsing shared by scoring, audit, and resume."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


class InvalidAlphaError(ValueError):
    """Raised/recorded when a condition alpha cannot define an identity."""


@dataclass(frozen=True)
class ConditionIdentity:
    clip_id: str
    mode: str
    alpha: Optional[float]
    record_key: str
    valid: bool
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None


def _fallback_suffix(row: Dict[str, Any], row_index: Optional[int]) -> str:
    if row_index is not None:
        return f"row{int(row_index)}"
    encoded = json.dumps(row, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _format_key(clip_id: str, mode: str, alpha: float) -> str:
    return f"{clip_id}|{mode}|{alpha:.3f}".rstrip("0").rstrip(".") if alpha != 0 else f"{clip_id}|{mode}|0"


def parse_condition_identity(row: Dict[str, Any], row_index: Optional[int] = None) -> ConditionIdentity:
    """Parse identity exactly once; malformed alpha gets a deterministic fallback key."""
    if not isinstance(row, dict):
        row = {"value": repr(row)}
    clip_id = str(row.get("clip_id", "unknown"))
    mode = str(row.get("mode", "unknown"))
    raw_alpha = row.get("alpha") if "alpha" in row else None
    if "alpha" not in row:
        return ConditionIdentity(
            clip_id=clip_id,
            mode=mode,
            alpha=None,
            record_key=f"{clip_id}|{mode}|invalid-alpha-{_fallback_suffix(row, row_index)}",
            valid=False,
            failure_type="InvalidAlphaError",
            failure_reason="Missing required alpha field",
        )
    try:
        alpha = float(raw_alpha)
        if not math.isfinite(alpha):
            raise ValueError(f"alpha must be finite, got {raw_alpha!r}")
    except (TypeError, ValueError) as exc:
        suffix = _fallback_suffix(row, row_index)
        return ConditionIdentity(
            clip_id=clip_id,
            mode=mode,
            alpha=None,
            record_key=f"{clip_id}|{mode}|invalid-alpha-{suffix}",
            valid=False,
            failure_type="InvalidAlphaError",
            failure_reason=f"Invalid alpha {raw_alpha!r}: {exc}",
        )
    return ConditionIdentity(clip_id, mode, alpha, _format_key(clip_id, mode, alpha), True)


def record_key_from_prediction(row: Dict[str, Any], row_index: Optional[int] = None) -> str:
    return parse_condition_identity(row, row_index=row_index).record_key
