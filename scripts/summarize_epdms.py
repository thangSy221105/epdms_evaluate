#!/usr/bin/env python3
"""CLI aggregator to compute summary statistics, paired deltas, and research reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from tools.epdms.aggregate import (
    aggregate_by_group,
    compute_ade_disagreement_summary,
    compute_paired_deltas,
    compute_paired_summary,
)
from tools.epdms.config import EvaluationConfig
from tools.epdms.io_jsonl import iter_jsonl
from tools.epdms.reporting import (
    export_ade_disagreement_to_markdown,
    export_mode_alpha_to_markdown,
    export_paired_summary_to_markdown,
    export_rule_group_to_markdown,
    export_table_to_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EPDMS evaluation scores and generate research reports.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "epdms_300.json",
        help="Path to evaluation config JSON.",
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        default=None,
        help="Path to evaluated scores JSONL file.",
    )
    args = parser.parse_args()

    config = EvaluationConfig.from_file(args.config)
    score_file = args.score_file or (config.score_dir / "epdms_scores_300.jsonl")
    analysis_dir = config.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not score_file.is_file():
        print(f"[!] Error: Score file not found: {score_file}")
        sys.exit(1)

    print(f"[*] Reading evaluated records from: {score_file}")
    records = list(iter_jsonl(score_file))
    print(f"[*] Loaded {len(records)} evaluated records.")

    # 1. Aggregate by Mode & Alpha
    print("[*] Aggregating by Mode and Alpha...")
    mode_alpha_data = aggregate_by_group(records, group_keys=["mode", "alpha"])
    export_table_to_csv(mode_alpha_data, analysis_dir / "epdms_by_mode_alpha.csv")
    export_mode_alpha_to_markdown(mode_alpha_data, analysis_dir / "epdms_by_mode_alpha.md")
    with (analysis_dir / "epdms_by_mode_alpha.json").open("w", encoding="utf-8") as f:
        json.dump(mode_alpha_data, f, indent=2)

    # 2. Aggregate by Rule Group & Alpha
    print("[*] Aggregating by Rule Group and Alpha...")
    rule_group_data = aggregate_by_group(records, group_keys=["rule_group", "alpha"])
    export_table_to_csv(rule_group_data, analysis_dir / "epdms_by_rule_group_alpha.csv")
    export_rule_group_to_markdown(rule_group_data, analysis_dir / "epdms_by_rule_group_alpha.md")
    with (analysis_dir / "epdms_by_rule_group_alpha.json").open("w", encoding="utf-8") as f:
        json.dump(rule_group_data, f, indent=2)

    # 3. Paired Deltas vs Alpha 0
    print("[*] Computing Paired Deltas (alpha > 0 vs alpha == 0)...")
    paired_data = compute_paired_deltas(records, practical_delta=config.practical_score_delta)
    export_table_to_csv(paired_data, analysis_dir / "paired_delta_vs_alpha0.csv")

    paired_summary = compute_paired_summary(paired_data)
    export_table_to_csv(paired_summary, analysis_dir / "paired_summary_by_mode_alpha.csv")
    export_paired_summary_to_markdown(paired_summary, analysis_dir / "paired_summary_by_mode_alpha.md")
    with (analysis_dir / "paired_summary_by_mode_alpha.json").open("w", encoding="utf-8") as f:
        json.dump(paired_summary, f, indent=2)

    # 4. ADE vs Safety Disagreement Analysis
    print("[*] Computing ADE vs Safety Disagreement Analysis...")
    ade_disagreement = compute_ade_disagreement_summary(
        paired_data,
        practical_delta_safety=config.practical_score_delta,
    )
    export_table_to_csv(ade_disagreement, analysis_dir / "ade_safety_disagreement.csv")
    export_ade_disagreement_to_markdown(ade_disagreement, analysis_dir / "ade_safety_disagreement.md")
    with (analysis_dir / "ade_safety_disagreement.json").open("w", encoding="utf-8") as f:
        json.dump(ade_disagreement, f, indent=2)

    # 5. Generate Research Summary Markdown
    summary_md = analysis_dir / "final_research_summary.md"
    lines = [
        "# Final Research Summary — NuRec 300 EPDMS Safety Analysis",
        "",
        f"**Evaluated Records:** {len(records)}",
        f"**Profile:** `{config.metric_profile}`",
        f"**Horizon:** `{config.horizon_s}s`",
        "",
        "## 1. Summary by Mode and Alpha",
        "",
    ]

    with (analysis_dir / "epdms_by_mode_alpha.md").open("r", encoding="utf-8") as f:
        lines.append(f.read())

    lines.extend([
        "",
        "## 2. Summary by Rule Group and Alpha",
        "",
    ])

    with (analysis_dir / "epdms_by_rule_group_alpha.md").open("r", encoding="utf-8") as f:
        lines.append(f.read())

    lines.extend([
        "",
        "## 3. Paired Delta Analysis vs Alpha 0 (Bootstrap 95% CI)",
        "",
    ])

    with (analysis_dir / "paired_summary_by_mode_alpha.md").open("r", encoding="utf-8") as f:
        lines.append(f.read())

    lines.extend([
        "",
        "## 4. ADE vs Safety Disagreement (Research Question: Does Alpha Degrade Safety or Just Deviate from GT?)",
        "",
        "> **Key Insight:** When alpha increases, trajectory deviation from Ground Truth increases (higher ADE).",
        "> This table examines whether cases penalized by ADE are genuinely less safe or merely alternative safe trajectories.",
        "",
    ])

    with (analysis_dir / "ade_safety_disagreement.md").open("r", encoding="utf-8") as f:
        lines.append(f.read())

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[+] Summaries generated successfully:")
    print(f"    - {analysis_dir / 'epdms_by_mode_alpha.md'}")
    print(f"    - {analysis_dir / 'epdms_by_rule_group_alpha.md'}")
    print(f"    - {analysis_dir / 'paired_delta_vs_alpha0.csv'}")
    print(f"    - {analysis_dir / 'paired_summary_by_mode_alpha.md'}")
    print(f"    - {analysis_dir / 'ade_safety_disagreement.csv'}")
    print(f"    - {analysis_dir / 'ade_safety_disagreement.md'}")
    print(f"    - {summary_md}")


if __name__ == "__main__":
    main()
