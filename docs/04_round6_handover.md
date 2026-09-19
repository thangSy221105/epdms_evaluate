# Round 6 Handover — Contract Consistency & Readiness Integrity

BASE_COMMIT: `13dda6067bd6343746baeadd6d7e754acfd69174`
NEW_BRANCH: `fix/round6-contract-consistency`
NEW_COMMIT: `42e9086c6e8cde0db3a395bd1f847f29cdbe7c40`

## Scope

This round closes only the remaining findings from the Round 5 review. Metric definitions, safety-proxy weights, and official NAVSIM implementations were not changed.

## Findings closed

| Finding | Root cause | Code fix | Regression test | Result |
|---|---|---|---|---|
| GT crop | Preflight and scoring used different t0/crop logic | Added shared `prepare_trajectory_window()` for prediction and GT | 40/41/64/65 pose cases | PASS |
| Coordinate contract | Missing metadata could pass strict scoring | Strict scoring now requires all frames and anchors to be explicitly verified | Missing/partial/mismatched metadata cases | PASS |
| Confirmed-empty evidence | Empty obstacle list returned before per-frame coverage | CF/TTC now evaluate object and confirmed-empty states through the common coverage helper | Full/partial empty evidence and out-of-horizon objects | PASS |
| GT readiness | Record presence was treated as GT validity | Audit normalizes every required GT and checks the common grid | Malformed GT audit case | PASS |
| Observation readiness | Audit checked only obstacle-list presence | Audit reports CF/TTC required, observed, confirmed-empty, missing, and ratios | 1/41 and 41/41 coverage cases | PASS |
| Absolute timestamp audit | Magnitude heuristic could misclassify small absolute timestamps | Audit consumes normalized timeline timestamps directly | Relative/absolute representation equivalence | PASS |
| Fingerprint scope | `--max-clips` was applied after fingerprinting | Exact sorted clip IDs are included before fingerprint/resume | Different scope sets/counts | PASS |
| Manifest status | Any valid score could look completed | Added execution/scoring/dataset status and unique current-state counts | COMPLETE/PARTIAL/BLOCKED cases | PASS |
| Retry lineage | Retry attempt was hard-coded to 2 | Latest state loader scans canonical and attempt files with attempt lineage | Attempt 1→2→3, invalid→valid | PASS |
| Checkpoint duplicates | Internal duplicate keys were not rejected | Target/tmp recovery rejects internal duplicates and overlap | target/tmp duplicate/disjoint cases | PASS |

## Verification

`tests/epdms/test_contracts_r6.py`: 24/24 pass.

Round 4 + Round 5 contract tests: pass.

Full discovery: 106 pass, 2 errors caused by the environment lacking both `pyarrow` and `fastparquet`; the affected tests only create temporary parquet fixtures.

`compileall`: pass.

Phase 0 audit on the configured 300-clip scope:

```text
PREDICTION_TIMELINE_READY=True
GT_TIMELINE_READY=True
TIME_ALIGNMENT_READY=False
COORDINATE_ALIGNMENT_READY=False
OBSERVATION_COVERAGE_CONTRACT_READY=False
MAP_READY=False
INPUT_GRID_READY=False
DATASET_STATUS=DATASET_NOT_READY
```

These remaining dataset blockers are reported as real data/environment blockers. No clock offset, coordinate default, fake empty evidence, or synthetic map data was introduced.

CODE_STATUS: `ROUND6_CONTRACT_TESTS_PASS`
DATA_STATUS: `DATASET_NOT_READY`
PILOT_STATUS: `REAL_SCORING_PILOT_BLOCKED`

No real scoring pilot was forced because the mandatory dataset contract is not ready.
