# Round 5 Technical Handover — Time Alignment and Run Integrity

## Status

| Area | Status | Evidence |
|---|---|---|
| Code contract tests | PASS | `tests/epdms/test_contracts_r5.py` — 15/15 |
| Existing regression suite | BLOCKED/EXPECTED REVIEW | Strict empty-observation behavior intentionally invalidates legacy tests that treated `all_obstacles=[]` as safe |
| Prediction timeline | PASS for configured 4.0 s window | 64-pose NuRec input is explicitly cropped to the first 40 future poses |
| GT/prediction common grid | ENFORCED | Both use `normalize_trajectory_timeline`; ADE/FDE compares normalized future timestamps |
| Time alignment | BLOCKED | Real NuRec obstacle timestamps are outside the prediction clip clock |
| Coordinate alignment | BLOCKED | Real rows do not declare frame/anchor metadata |
| Map | BLOCKED | 156 clips have no map parquet; parquet validation also requires a parquet engine |
| Real-data pilot | GATING ONLY | 3 clips / 48 conditions, 0 valid, 48 invalid; this is not a scoring pilot |

Generated evidence is in `round5_audit_v2/` and `round5_pilot_v2/` in the workspace.

## Implemented contracts

- Added one parser for prediction and GT timestamp/coordinate validation.
- Explicit relative and absolute timestamps are distinguished by field schema.
- Absolute timestamps must be anchored to `t0_us`; shifted clocks raise `TimelineOriginMismatchError`.
- GT timeline errors are no longer swallowed by `except ...: pass`.
- ADE/FDE uses the common normalized future grid.
- Strict empty obstacle lists require explicit `confirmed_empty_scene` evidence.
- Fingerprinting now has one source of truth: `run_identity.METRIC_IMPLEMENTATION_VERSION = 2.3.0-r5`.
- Resume reads valid and invalid keys separately; `--retry-invalid` writes a separate attempt file.
- Overlapping target/tmp checkpoints raise `AmbiguousCheckpointError`.
- Phase 0 emits `time_alignment_report.csv/md` and `coordinate_alignment_report.csv/md`.

## Reproduction commands

```powershell
$py = 'C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m unittest discover -s tests/epdms -p 'test*.py' -v
& $py scripts/audit_epdms_inputs.py --config configs/epdms_300.json
& $py scripts/evaluate_epdms.py --config configs/epdms_300.json --score-dir round5_pilot_v2 --max-clips 3 --no-resume
& $py scripts/evaluate_epdms.py --config configs/epdms_300.json --score-dir round5_pilot_v2 --max-clips 3 --resume
& $py scripts/evaluate_epdms.py --config configs/epdms_300.json --score-dir round5_pilot_v2 --max-clips 3 --resume --retry-invalid
```

## Remaining blockers

1. Verify the NuRec clock relationship across prediction, GT, obstacle and egomotion records. No global offset is applied by this branch.
2. Declare and verify prediction/GT/obstacle/map frame plus rear-axle/vehicle-center anchor.
3. Provide the missing map parquet files and a parquet engine in the execution environment.
4. Re-run Phase 0; only then run a real valid-condition end-to-end pilot and the full evaluation.

The branch must not be labelled `DATASET_READY` or `READY_FOR_FULL_RESEARCH_RUN` while these blockers remain.
