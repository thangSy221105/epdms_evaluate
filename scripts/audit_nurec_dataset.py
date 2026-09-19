#!/usr/bin/env python3
"""Audit NuRec availability and EPDMS data contracts without transformation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prep.nurec import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and audit a NuRec dataset for EPDMS readiness.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--context-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(args.dataset_root, args.prediction_jsonl, args.ground_truth_jsonl, args.output_dir, args.context_jsonl)
    summary = result["summary"]
    print(f"DATASET_AUDIT clips={summary['total_clips']} proxy_ready={summary['proxy_ready_clip_count']} status={summary['status']}")
    print(f"PARQUET_ENGINE status={summary['parquet_engine']['status']} errors={summary['audit_error_count']}")


if __name__ == "__main__":
    main()

