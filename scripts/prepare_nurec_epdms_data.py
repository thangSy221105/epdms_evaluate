#!/usr/bin/env python3
"""Create non-destructive NuRec EPDMS staging manifests from audit output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_prep.nurec import prepare_staging


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical NuRec EPDMS staging manifests without editing raw files.")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_staging(args.audit_dir, args.output_dir)
    print(f"STAGING_DONE clips={result['clips_staged']} proxy_ready={result['proxy_ready_clip_count']} output={result['output_dir']}")


if __name__ == "__main__":
    main()

