"""Reporting module: exports aggregate tables in Markdown and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_table_to_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    fieldnames = list(data[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def export_mode_alpha_to_markdown(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Mode", "Alpha", "N", "Mean Proxy", "Median", "Std",
        "Collision %", "Offroad %", "TTC Fail %", "Comfort Fail %", "Mean Progress"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in data:
        row = [
            str(r.get("mode", "")),
            str(r.get("alpha", "")),
            str(r.get("N", "")),
            f"{float(r.get('mean_safety_proxy', 0.0)):.4f}",
            f"{float(r.get('median_safety_proxy', 0.0)):.4f}",
            f"{float(r.get('std_safety_proxy', 0.0)):.4f}",
            f"{float(r.get('collision_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('offroad_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('ttc_failure_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('comfort_failure_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('mean_progress', 0.0)):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_rule_group_to_markdown(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Rule Group", "Alpha", "N", "Mean Proxy", "Collision %", "Offroad %", "Comfort Fail %", "Progress"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in data:
        row = [
            str(r.get("rule_group", "")),
            str(r.get("alpha", "")),
            str(r.get("N", "")),
            f"{float(r.get('mean_safety_proxy', 0.0)):.4f}",
            f"{float(r.get('collision_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('offroad_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('comfort_failure_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('mean_progress', 0.0)):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_paired_summary_to_markdown(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Mode", "Alpha", "N Paired", "Mean Delta", "Median Delta", "Bootstrap 95% CI",
        "Improved %", "Degraded %", "Numerical Tie"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in data:
        ci_str = f"[{float(r.get('ci95_lower', 0.0)):.4f}, {float(r.get('ci95_upper', 0.0)):.4f}]"
        row = [
            str(r.get("mode", "")),
            str(r.get("alpha", "")),
            str(r.get("n_paired", "")),
            f"{float(r.get('mean_delta', 0.0)):+.4f}",
            f"{float(r.get('median_delta', 0.0)):+.4f}",
            ci_str,
            f"{float(r.get('improved_rate_pct', 0.0)):.2f}%",
            f"{float(r.get('degraded_rate_pct', 0.0)):.2f}%",
            str(r.get("numerical_tie_count", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_ade_disagreement_to_markdown(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Mode", "Alpha", "N Paired", "N with ADE", "Mean Delta ADE (m)", "Mean Delta Safety",
        "ADE Penalized", "Disagreement (Safe/Unchanged)", "Disagreement %", "Genuinely Safe (Gate Kept)",
        "Safety Compromised", "Both Degraded"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in data:
        d_ade = r.get("mean_delta_ade")
        d_ade_str = f"{float(d_ade):+.4f}" if d_ade is not None else "N/A"

        d_safe = r.get("mean_delta_safety")
        d_safe_str = f"{float(d_safe):+.4f}" if d_safe is not None else "N/A"

        dis_pct = r.get("disagreement_rate_pct")
        dis_pct_str = f"{float(dis_pct):.2f}%" if dis_pct is not None else "N/A"

        row = [
            str(r.get("mode", "")),
            str(r.get("alpha", "")),
            str(r.get("n_paired", "")),
            str(r.get("n_with_ade", "N/A")),
            d_ade_str,
            d_safe_str,
            str(r.get("ade_penalized_cases", 0)),
            str(r.get("disagreement_count", 0)),
            dis_pct_str,
            str(r.get("genuinely_safe_count", 0)),
            str(r.get("safety_compromised_count", 0)),
            str(r.get("both_degraded_count", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

