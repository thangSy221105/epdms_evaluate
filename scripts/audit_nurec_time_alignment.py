#!/usr/bin/env python3
"""Audit NuRec timestamp and clock-domain evidence without applying mappings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prep.time_alignment import audit_time_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensically audit NuRec time alignment without guessing offsets.")
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository to search for upstream t0 provenance candidates.")
    args = parser.parse_args()
    result = audit_time_alignment(args.clip_dir, args.clip_id, args.prediction_jsonl, args.ground_truth_jsonl, args.output_dir, args.repo_root)
    evidence = result["evidence"]
    print(f"TIME_ALIGNMENT clip={args.clip_id} status={evidence['mapping']['status']} verified={evidence['mapping']['verified']} output={args.output_dir}")


if __name__ == "__main__":
    main()
