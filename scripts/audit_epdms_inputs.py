#!/usr/bin/env python3
"""Phase 0 Audit CLI: Runs environment and data contracts audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.epdms.audit import run_and_save_audit
from tools.epdms.config import EvaluationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit environment and input datasets for EPDMS evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "epdms_300.json",
        help="Path to evaluation config JSON.",
    )
    args = parser.parse_args()

    print(f"[*] Loading config from: {args.config}")
    config = EvaluationConfig.from_file(args.config)

    output_dir = config.analysis_dir
    print(f"[*] Running audit and writing reports to: {output_dir}")
    result = run_and_save_audit(config, output_dir)

    env = result["environment"]
    readiness = result["contracts"]["readiness"]

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY & READINESS CONCLUSION:")
    print("=" * 60)
    print(f"  Python Version:     {sys.version.split()[0]}")
    print(f"  NumPy Available:    {env['numpy_available']} ({env.get('numpy_version', 'N/A')})")
    print(f"  Pandas Available:   {env['pandas_available']} ({env.get('pandas_version', 'N/A')})")
    print(f"  NAVSIM Available:   {env['navsim_available']}")
    print(f"  nuPlan Available:   {env['nuplan_available']}")
    print("-" * 60)
    print(f"  Profile 'navsim_v2_full':       {readiness['navsim_v2_full']}")
    print(f"  Profile 'navsim_v2_stage1':     {readiness['navsim_v2_stage1']}")
    print(f"  Profile 'nurec_safety_proxy_v1': {readiness['nurec_safety_proxy_v1']}")
    print("=" * 60)
    print(f"[+] Files generated:")
    for f in result["files_written"]:
        print(f"    - {f}")
    print("\nPhase 0 Audit completed successfully.")


if __name__ == "__main__":
    main()
