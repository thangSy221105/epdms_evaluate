#!/usr/bin/env python3
"""Inspect one NuRec clip without dumping or modifying raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prep.nurec import write_schema_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one NuRec clip schema and write a bounded report.")
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = write_schema_report(args.clip_dir, args.output_dir)
    print(f"SCHEMA_REPORT clip={report['clip_id']} parquet={report['parquet_engine']['status']} output={args.output_dir}")


if __name__ == "__main__":
    main()

