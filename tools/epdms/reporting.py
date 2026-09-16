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
            f"{r.get('mean_safety_proxy', 0.0):.4f}",
            f"{r.get('median_safety_proxy', 0.0):.4f}",
            f"{r.get('std_safety_proxy', 0.0):.4f}",
            f"{r.get('collision_rate_pct', 0.0):.2f}%",
            f"{r.get('offroad_rate_pct', 0.0):.2f}%",
            f"{r.get('ttc_failure_rate_pct', 0.0):.2f}%",
            f"{r.get('comfort_failure_rate_pct', 0.0):.2f}%",
            f"{r.get('mean_progress', 0.0):.4f}",
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
            f"{r.get('mean_safety_proxy', 0.0):.4f}",
            f"{r.get('collision_rate_pct', 0.0):.2f}%",
            f"{r.get('offroad_rate_pct', 0.0):.2f}%",
            f"{r.get('comfort_failure_rate_pct', 0.0):.2f}%",
            f"{r.get('mean_progress', 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
