# NuRec final contract sync

BASE_COMMIT: `c7597b63c0afbac2187751985c0cd3d0c24814c2`
BRANCH: `fix/nurec-data-prep-final-contract-sync`
NEW_COMMIT: `7c7c85a` (`fix(data): synchronize NuRec coverage with scorer contracts`)

This pass is data-preparation-only. No file under `tools/epdms/`, no scorer,
metric, evaluator, NAVSIM implementation, clock transform, coordinate
transform, DAC conversion, or raw NuRec data was changed.

## Closed findings

- Object observations use scorer-compatible nearest matching: 50 ms for CF and
  100 ms for TTC.
- Confirmed-empty evidence is exact-query only, matching
  `evaluate_query_coverage`; an empty attestation at `query + 1 us` or
  `query + 49 ms` is not accepted as empty.
- Query grids use one effective configuration. The audit accepts an
  `EvaluationConfig` or explicit settings, supports CLI `--config` and runtime
  horizon/frequency/TTC overrides, and records settings plus a grid
  fingerprint in each contract. No broad fallback to 41/51 exists.
- Prediction loader reports raw, unique, and duplicate condition counts. Only
  unique valid conditions enter canonical condition checks; duplicate physical
  rows remain forensic errors and block the clip.
- Coverage remains uncertified whenever time alignment is unresolved, even if
  a query grid can be built and its provenance is known.

## HF v8 validation

Commands used the official clip and `configs/epdms_300.json`; outputs were not
written over v7:

- `D:\300_clip_nurec\hf_probe\schema_v6`
- `D:\300_clip_nurec\hf_probe\audit_v8`
- `D:\300_clip_nurec\hf_probe\prepared_v8`

```text
raw prediction conditions       = 16
unique prediction conditions    = 16
duplicate prediction conditions = 0
audit errors                    = 0
proxy-ready                     = 0 / 1
```

Effective query grid from the config:

```text
horizon_s       = 4.0
frequency_hz    = 10.0
future_poses    = 40
CF queries      = 41
ttc_horizon_s   = 1.0
TTC queries     = 51
```

The remaining evidence-based blockers are:

```text
TIME_ALIGNMENT_UNRESOLVED
COORDINATE_UNRESOLVED
OBSERVATION_COVERAGE_INCOMPLETE
DAC_GEOMETRY_UNVERIFIED
```

## Verification

```text
data-prep tests  = 84 passed
EPDMS tests      = 133 passed
compileall       = passed
```

The HF contract records the query-grid provenance, but CF/TTC counts remain
uncertified because time alignment is still unresolved. No time alignment,
coordinate transform, DAC implementation, or official EPDMS work was started.
