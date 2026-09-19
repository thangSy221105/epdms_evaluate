#!/usr/bin/env python3
"""Audit NuRec availability and EPDMS data contracts without transformation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prep.nurec import audit_dataset
from tools.epdms.config import EvaluationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and audit a NuRec dataset for EPDMS readiness.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--context-jsonl", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Effective EPDMS evaluation config JSON.")
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--frequency", type=float, default=None)
    parser.add_argument("--ttc-horizon", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation_config = EvaluationConfig.from_file(args.config) if args.config else None
    if evaluation_config is not None:
        if args.horizon is not None:
            evaluation_config.horizon_s = args.horizon
        if args.frequency is not None:
            evaluation_config.frequency_hz = args.frequency
        if args.horizon is not None or args.frequency is not None:
            evaluation_config.future_poses = int(round(evaluation_config.horizon_s * evaluation_config.frequency_hz))
        if args.ttc_horizon is not None:
            evaluation_config.ttc_horizon_s = args.ttc_horizon
    result = audit_dataset(args.dataset_root, args.prediction_jsonl, args.ground_truth_jsonl, args.output_dir, args.context_jsonl, evaluation_config)
    summary = result["summary"]
    print(f"DATASET_AUDIT clips={summary['total_clips']} proxy_ready={summary['proxy_ready_clip_count']} status={summary['status']}")
    print(f"PARQUET_ENGINE status={summary['parquet_engine']['status']} errors={summary['audit_error_count']}")
    if result["contracts"]:
        print(f"QUERY_GRID {json.dumps(next(iter(result['contracts'].values())).get('query_grid', {}), ensure_ascii=True, sort_keys=True)}")


if __name__ == "__main__":
    main()
