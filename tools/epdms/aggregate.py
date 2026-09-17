"""Statistical aggregation, bootstrap confidence intervals, and paired delta analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
) -> List[Dict[str, Any]]:
    """Groups evaluated records by specified keys and aggregates metrics.
    
    Robust against None values in group keys (e.g. rule_group).
    Keeps full precision in intermediate data, adding rounded fields for display.
    """
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        if not r.get("valid", True):
            continue
        g = tuple(r.get(k) for k in group_keys)
        buckets.setdefault(g, []).append(r)

    rows = []
    # Sort with safe key that handles None
    sorted_items = sorted(
        buckets.items(),
        key=lambda item: tuple(str(x) if x is not None else "" for x in item[0]),
    )

    for g, items in sorted_items:
        n = len(items)
        scores = np.array([float(it["nurec_safety_proxy_v1"]) for it in items if it.get("nurec_safety_proxy_v1") is not None])
        cfs = np.array([float(it["collision_free_proxy"]) for it in items if it.get("collision_free_proxy") is not None])
        dacs = np.array([float(it["dac_proxy"]) for it in items if it.get("dac_proxy") is not None])
        ttcs = np.array([float(it["ttc_proxy"]) for it in items if it.get("ttc_proxy") is not None])
        fcs = np.array([float(it["future_comfort_proxy"]) for it in items if it.get("future_comfort_proxy") is not None])
        eps = np.array([float(it["progress_gt_proxy"]) for it in items if it.get("progress_gt_proxy") is not None])

        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        median_score = float(np.median(scores)) if len(scores) > 0 else 0.0
        std_score = float(np.std(scores)) if len(scores) > 0 else 0.0

        collision_rate = float(np.mean(cfs == 0.0) * 100.0) if len(cfs) > 0 else 0.0
        offroad_rate = float(np.mean(dacs == 0.0) * 100.0) if len(dacs) > 0 else 0.0
        ttc_failure_rate = float(np.mean(ttcs == 0.0) * 100.0) if len(ttcs) > 0 else 0.0
        comfort_failure_rate = float(np.mean(fcs == 0.0) * 100.0) if len(fcs) > 0 else 0.0
        mean_progress = float(np.mean(eps)) if len(eps) > 0 else 0.0

        row = {group_keys[i]: g[i] for i in range(len(group_keys))}
        row.update({
            "N": n,
            "mean_safety_proxy": mean_score,
            "median_safety_proxy": median_score,
            "std_safety_proxy": std_score,
            "collision_rate_pct": collision_rate,
            "offroad_rate_pct": offroad_rate,
            "ttc_failure_rate_pct": ttc_failure_rate,
            "comfort_failure_rate_pct": comfort_failure_rate,
            "mean_progress": mean_progress,
        })
        rows.append(row)

    return rows


def compute_paired_deltas(
    records: List[Dict[str, Any]],
    practical_delta: float = 0.01,
) -> List[Dict[str, Any]]:
    """Pairs each (clip_id, mode, alpha > 0) with (clip_id, mode, alpha == 0).
    
    Maintains full float precision in delta_safety_proxy_raw.
    """
    baseline_map: Dict[Tuple[str, str, Any, Any, Any], Dict[str, Any]] = {}
    for r in records:
        if not r.get("valid", True):
            continue
        cid = str(r.get("clip_id", ""))
        mode = str(r.get("mode", ""))
        alpha = float(r.get("alpha", 0.0))
        horizon = r.get("horizon_s")
        freq = r.get("frequency_hz")
        profile = r.get("metric_profile")
        if alpha == 0.0:
            baseline_map[(cid, mode, horizon, freq, profile)] = r

    paired_rows = []
    for r in records:
        if not r.get("valid", True):
            continue
        cid = str(r.get("clip_id", ""))
        mode = str(r.get("mode", ""))
        alpha = float(r.get("alpha", 0.0))
        if alpha == 0.0:
            continue
        horizon = r.get("horizon_s")
        freq = r.get("frequency_hz")
        profile = r.get("metric_profile")

        base = baseline_map.get((cid, mode, horizon, freq, profile))
        if not base:
            continue

        base_score_raw = base.get("nurec_safety_proxy_v1")
        out_score_raw = r.get("nurec_safety_proxy_v1")
        if base_score_raw is None or out_score_raw is None:
            continue

        base_score = float(base_score_raw)
        out_score = float(out_score_raw)
        delta = out_score - base_score

        # Distinguish numerical tie from practical tie
        is_numerical_tie = abs(delta) <= 1e-9
        if delta > practical_delta:
            practical_status = "improved"
        elif delta < -practical_delta:
            practical_status = "degraded"
        else:
            practical_status = "unchanged"

        base_ade = base.get("ade_m")
        out_ade = r.get("ade_m")
        delta_ade = (float(out_ade) - float(base_ade)) if (base_ade is not None and out_ade is not None) else None

        base_fde = base.get("fde_m")
        out_fde = r.get("fde_m")
        delta_fde = (float(out_fde) - float(base_fde)) if (base_fde is not None and out_fde is not None) else None

        paired_rows.append({
            "clip_id": cid,
            "mode": mode,
            "alpha": alpha,
            "horizon_s": horizon,
            "frequency_hz": freq,
            "metric_profile": profile,
            "rule_group": r.get("rule_group"),
            "baseline_safety_proxy": base_score,
            "output_safety_proxy": out_score,
            "delta_safety_proxy_raw": delta,
            "delta_safety_proxy": delta,
            "is_numerical_tie": is_numerical_tie,
            "practical_status": practical_status,
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
            "base_ade": base_ade,
            "out_ade": out_ade,
            "delta_ade": delta_ade,
            "base_fde": base_fde,
            "out_fde": out_fde,
            "delta_fde": delta_fde,
        })

    return paired_rows


def compute_paired_summary(
    paired_rows: List[Dict[str, Any]],
    group_keys: List[str] = ["mode", "alpha"],
    bootstrap_iterations: int = 5000,
    random_seed: int = 2026,
) -> List[Dict[str, Any]]:
    """Computes aggregate paired statistics including mean delta, median delta, and bootstrap 95% CI."""
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in paired_rows:
        g = tuple(r.get(k) for k in group_keys)
        buckets.setdefault(g, []).append(r)

    results = []
    sorted_items = sorted(
        buckets.items(),
        key=lambda item: tuple(str(x) if x is not None else "" for x in item[0]),
    )

    for g, items in sorted_items:
        n = len(items)
        deltas = np.array([float(it["delta_safety_proxy_raw"]) for it in items])
        mean_d = float(np.mean(deltas))
        median_d = float(np.median(deltas))
        std_d = float(np.std(deltas))

        ci_low, ci_high = bootstrap_ci(deltas, n_iterations=bootstrap_iterations, random_seed=random_seed)

        improved = sum(1 for it in items if it["practical_status"] == "improved")
        unchanged = sum(1 for it in items if it["practical_status"] == "unchanged")
        degraded = sum(1 for it in items if it["practical_status"] == "degraded")
        num_ties = sum(1 for it in items if it["is_numerical_tie"])

        res = {group_keys[i]: g[i] for i in range(len(group_keys))}
        res.update({
            "n_paired": n,
            "mean_delta": mean_d,
            "median_delta": median_d,
            "std_delta": std_d,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "improved_count": improved,
            "unchanged_count": unchanged,
            "degraded_count": degraded,
            "numerical_tie_count": num_ties,
            "improved_rate_pct": (improved / n) * 100.0 if n > 0 else 0.0,
            "degraded_rate_pct": (degraded / n) * 100.0 if n > 0 else 0.0,
        })
        results.append(res)

    return results


def compute_ade_disagreement_summary(
    paired_rows: List[Dict[str, Any]],
    group_keys: List[str] = ["mode", "alpha"],
    ade_penalty_threshold_m: float = 0.05,
    practical_delta_safety: float = 0.01,
) -> List[Dict[str, Any]]:
    """Analyzes whether increases in ADE error correspond to real safety degradation or harmless deviation.
    
    A condition is considered in 'Disagreement' if ADE got worse (delta_ade > threshold)
    but Safety Proxy stayed unchanged or improved (delta_safety >= -practical_delta).
    """
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in paired_rows:
        g = tuple(r.get(k) for k in group_keys)
        buckets.setdefault(g, []).append(r)

    results = []
    sorted_items = sorted(
        buckets.items(),
        key=lambda item: tuple(str(x) if x is not None else "" for x in item[0]),
    )

    for g, items in sorted_items:
        n = len(items)
        valid_ade_items = [it for it in items if it.get("delta_ade") is not None]
        n_ade = len(valid_ade_items)

        if n_ade > 0:
            delta_ades = np.array([float(it["delta_ade"]) for it in valid_ade_items])
            delta_safeties = np.array([float(it["delta_safety_proxy_raw"]) for it in valid_ade_items])

            mean_d_ade = float(np.mean(delta_ades))
            mean_d_safety = float(np.mean(delta_safeties))

            # Penalized by ADE (error relative to GT increased)
            penalized = [it for it in valid_ade_items if float(it["delta_ade"]) > ade_penalty_threshold_m]
            n_penalized = len(penalized)

            # Disagreement: ADE degraded, but safety did NOT degrade
            disagreement = [it for it in penalized if float(it["delta_safety_proxy_raw"]) >= -practical_delta_safety]
            n_disagreement = len(disagreement)

            # Genuinely safe: output has no collision (CF=1) and no offroad (DAC=1)
            genuinely_safe = [
                it for it in disagreement
                if (it.get("out_cf") is not None and float(it["out_cf"]) >= 1.0)
                and (it.get("out_dac") is not None and float(it["out_dac"]) >= 1.0)
            ]
            n_genuinely_safe = len(genuinely_safe)

            # Safety compromised: safety proxy degraded or gate violated
            def _cf_val(d: Dict[str, Any], k: str) -> float:
                return float(d[k]) if (d.get(k) is not None) else 1.0

            def _dac_val(d: Dict[str, Any], k: str) -> float:
                return float(d[k]) if (d.get(k) is not None) else 1.0

            safety_compromised = [
                it for it in penalized
                if float(it["delta_safety_proxy_raw"]) < -practical_delta_safety
                or _cf_val(it, "out_cf") < _cf_val(it, "base_cf")
                or _dac_val(it, "out_dac") < _dac_val(it, "base_dac")
            ]
            n_safety_compromised = len(safety_compromised)

            # Both degraded: ADE degraded AND safety actually degraded
            both_degraded = [it for it in penalized if float(it["delta_safety_proxy_raw"]) < -practical_delta_safety]
            n_both_degraded = len(both_degraded)

            disagreement_pct = (n_disagreement / n_penalized * 100.0) if n_penalized > 0 else None
            gen_safe_pct = (n_genuinely_safe / n_penalized * 100.0) if n_penalized > 0 else None
        else:
            mean_d_ade = None
            mean_d_safety = None
            n_penalized = 0
            n_disagreement = 0
            n_genuinely_safe = 0
            n_safety_compromised = 0
            n_both_degraded = 0
            disagreement_pct = None
            gen_safe_pct = None

        res = {group_keys[i]: g[i] for i in range(len(group_keys))}
        res.update({
            "n_paired": n,
            "n_with_ade": n_ade,
            "n_excluded_missing_ade": n - n_ade,
            "mean_delta_ade": mean_d_ade,
            "mean_delta_safety": mean_d_safety,
            "ade_penalized_cases": n_penalized,
            "disagreement_count": n_disagreement,
            "disagreement_rate_pct": disagreement_pct,
            "genuinely_safe_count": n_genuinely_safe,
            "genuinely_safe_rate_pct": gen_safe_pct,
            "safety_compromised_count": n_safety_compromised,
            "both_degraded_count": n_both_degraded,
        })
        results.append(res)

    return results

