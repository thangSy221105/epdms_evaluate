"""Statistical aggregation, bootstrap confidence intervals, and paired delta analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    n_iterations: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 2026,
) -> Tuple[float, float]:
    """Computes bootstrap confidence interval for the mean."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(data[0]), float(data[0])

    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, n, size=(n_iterations, n))
    resampled_means = np.mean(data[indices], axis=1)

    alpha_half = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(resampled_means, alpha_half * 100))
    upper = float(np.percentile(resampled_means, (1.0 - alpha_half) * 100))
    return lower, upper


def aggregate_by_group(
    records: List[Dict[str, Any]],
    group_keys: List[str],
    practical_delta: float = 0.01,
) -> List[Dict[str, Any]]:
    """Groups evaluated records by specified keys (e.g. mode, alpha or rule_group, alpha) and aggregates metrics."""
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        if not r.get("valid", True):
            continue
        g = tuple(r.get(k) for k in group_keys)
        buckets.setdefault(g, []).append(r)

    rows = []
    for g, items in sorted(buckets.items()):
        n = len(items)
        scores = np.array([float(it.get("nurec_safety_proxy_v1", 0.0)) for it in items])
        cfs = np.array([float(it.get("collision_free_proxy", 1.0)) for it in items])
        dacs = np.array([float(it.get("dac_proxy", 1.0)) for it in items])
        ttcs = np.array([float(it.get("ttc_proxy", 1.0)) for it in items])
        fcs = np.array([float(it.get("future_comfort_proxy", 1.0)) for it in items])
        eps = np.array([float(it.get("progress_gt_proxy", 1.0)) for it in items])

        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores))

        collision_rate = float(np.mean(cfs == 0.0) * 100.0)
        offroad_rate = float(np.mean(dacs == 0.0) * 100.0)
        ttc_failure_rate = float(np.mean(ttcs == 0.0) * 100.0)
        comfort_failure_rate = float(np.mean(fcs == 0.0) * 100.0)
        mean_progress = float(np.mean(eps))

        row = {group_keys[i]: g[i] for i in range(len(group_keys))}
        row.update({
            "N": n,
            "mean_safety_proxy": round(mean_score, 4),
            "median_safety_proxy": round(median_score, 4),
            "std_safety_proxy": round(std_score, 4),
            "collision_rate_pct": round(collision_rate, 2),
            "offroad_rate_pct": round(offroad_rate, 2),
            "ttc_failure_rate_pct": round(ttc_failure_rate, 2),
            "comfort_failure_rate_pct": round(comfort_failure_rate, 2),
            "mean_progress": round(mean_progress, 4),
        })
        rows.append(row)

    return rows


def compute_paired_deltas(
    records: List[Dict[str, Any]],
    practical_delta: float = 0.01,
) -> List[Dict[str, Any]]:
    """Pairs each (clip_id, mode, alpha > 0) with (clip_id, mode, alpha == 0) and computes deltas."""
    baseline_map = {}
    for r in records:
        if not r.get("valid", True):
            continue
        cid = r.get("clip_id")
        mode = r.get("mode")
        alpha = float(r.get("alpha", 0.0))
        if alpha == 0.0:
            baseline_map[(cid, mode)] = r

    paired_rows = []
    for r in records:
        if not r.get("valid", True):
            continue
        cid = r.get("clip_id")
        mode = r.get("mode")
        alpha = float(r.get("alpha", 0.0))
        if alpha == 0.0:
            continue

        base = baseline_map.get((cid, mode))
        if not base:
            continue

        base_score = float(base.get("nurec_safety_proxy_v1", 0.0))
        out_score = float(r.get("nurec_safety_proxy_v1", 0.0))
        delta = out_score - base_score

        status = "unchanged"
        if delta > practical_delta:
            status = "improved"
        elif delta < -practical_delta:
            status = "degraded"

        paired_rows.append({
            "clip_id": cid,
            "mode": mode,
            "alpha": alpha,
            "rule_group": r.get("rule_group"),
            "baseline_safety_proxy": round(base_score, 4),
            "output_safety_proxy": round(out_score, 4),
            "delta_safety_proxy": round(delta, 4),
            "safety_status": status,
            "base_cf": base.get("collision_free_proxy"),
            "out_cf": r.get("collision_free_proxy"),
            "base_dac": base.get("dac_proxy"),
            "out_dac": r.get("dac_proxy"),
            "base_ttc": base.get("ttc_proxy"),
            "out_ttc": r.get("ttc_proxy"),
            "base_comfort": base.get("future_comfort_proxy"),
            "out_comfort": r.get("future_comfort_proxy"),
            "base_progress": base.get("progress_gt_proxy"),
            "out_progress": r.get("progress_gt_proxy"),
        })

    return paired_rows
