# Round 7 Handover — Run Isolation & Strict Gating

BASE_COMMIT: `7067149` (descendant of Round 6 code commit `9d60800b69bc61fc5bf837fb7b23ed658c29195f`)
NEW_BRANCH: `fix/round7-run-isolation-and-strict-gating`

## Scope

This round addresses only the Round 6 review findings around execution identity, checkpoint isolation, retry ambiguity, malformed condition identity, coordinate-gate ordering, and trajectory-origin ambiguity. Metric definitions, proxy weights, NAVSIM implementations, and dataset values were not changed.

## Findings closed

| Finding | Code fix | Verification |
|---|---|---|
| Fingerprint/run identity conflation | Added deterministic `effective_fingerprint` plus fresh unique `run_id`; resume reuses manifest `run_id` | Round 7 identity tests |
| Fresh-run state mixing | `--no-resume` refuses non-empty score directories unless `--overwrite-new-run` is explicit | Fresh-directory contract test |
| Retry artifact isolation | Attempt filenames include `run_id`; all retry and retry-`.tmp` files are inspected before recovery | Attempt/tmp/wrong-run tests |
| Missing or stale record identity | Strict state loading requires `run_id`, `effective_fingerprint`, `record_key`, and rejects stale rows | Strict loader tests |
| Ambiguous latest state | Same `record_key`/`run_id`/`attempt_number` with different state raises `AmbiguousRetryStateError`; identical duplicates dedupe | Conflict/dedupe tests |
| Malformed alpha batch crash | Shared `parse_condition_identity()` produces deterministic fallback keys and invalid denominator records | Malformed-alpha tests |
| Coordinate gate after geometry scoring | Coordinate alignment now unconditionally gates CF/TTC/DAC before proxy calls | Monkeypatch ordering test |
| Timestamp-less long source ambiguity | Long sources require explicit `future_only`/`includes_t0` provenance; provenance is stored in timeline metadata | Origin-policy tests |
| Trailing timestamp corruption | Crop is resolved before parsing the retained horizon; corruption outside the crop is ignored, corruption inside remains fatal | Inside/outside crop tests |
| Metric implementation identity | Bumped `METRIC_IMPLEMENTATION_VERSION` to `2.4.0-r7` | Fingerprint/version test |

## Verification

- `compileall`: PASS.
- Round 5 + Round 6 + Round 7 focused contracts: `64/64` PASS.
- Full Python unittest discovery: `131/133` PASS; the only two errors are environment limitations because the bundled runtime lacks both `pyarrow` and `fastparquet`, required by existing tests that create temporary parquet fixtures. No test assertion failed.
- Real Phase 0 audit completed against the configured dataset and wrote reports to `D:\300_clip_nurec\04_analysis\epdms`.

Current real-data readiness remains blocked:

```text
DATASET_STATUS=DATASET_NOT_READY
PREDICTION_TIMELINE_READY=True
GT_TIMELINE_READY=False
TIME_ALIGNMENT_READY=False
COORDINATE_ALIGNMENT_READY=False
OBSERVATION_COVERAGE_CONTRACT_READY=False
MAP_READY=False
INPUT_GRID_READY=False
```

The GT timeline result is intentionally stricter in Round 7: timestamp-less long sources without explicit origin provenance are ambiguous. No timestamp offset, coordinate transform, synthetic observation evidence, map data, or fake pilot was introduced.

CODE_STATUS: `ROUND7_CONTRACTS_PASS`
DATA_STATUS: `DATASET_NOT_READY`
PILOT_STATUS: `REAL_SCORING_PILOT_BLOCKED`

The branch is ready for reviewer inspection. No scoring pilot was forced because the mandatory input contracts are not ready.
