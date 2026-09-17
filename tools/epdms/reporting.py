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
        "Mode", "Alpha", "N Paired", "Mean Delta ADE (m)", "Mean Delta Safety",
        "ADE Penalized", "Disagreement (Safe/Unchanged)", "Disagreement %", "Both Degraded"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in data:
        row = [
            str(r.get("mode", "")),
            str(r.get("alpha", "")),
            str(r.get("n_paired", "")),
            f"{float(r.get('mean_delta_ade', 0.0)):+.4f}",
            f"{float(r.get('mean_delta_safety', 0.0)):+.4f}",
            str(r.get("ade_penalized_cases", 0)),
            str(r.get("disagreement_count", 0)),
            f"{float(r.get('disagreement_rate_pct', 0.0)):.2f}%",
            str(r.get("both_degraded_count", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

