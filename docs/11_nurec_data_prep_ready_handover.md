# NuRec data-prep ready handover

BASE_COMMIT: `cc32908eb062aa28cae11f368dcf0928b9cb32af`
BRANCH: `fix/nurec-data-prep-final-ready-gate`
NEW_COMMIT: `23a21f1` (`fix(data): allow complete object-only observation coverage`)

## Closed finding

`OBJECT_ONLY_COVERAGE_DOES_NOT_REQUIRE_COMPLETENESS`.

Object observations are authoritative when exact or nearest scorer-tolerance
timestamps cover the query. `obstacle_table_complete` is not a global object
readiness gate. It is used only to promote independent exact frame evidence
without an object from `UNKNOWN` to `OBSERVED_EMPTY`.

Therefore:

- full object-only CF coverage can be ready with no frame evidence;
- full object-only TTC coverage can be ready with no frame evidence;
- partial object coverage remains blocked;
- incomplete-table frame evidence without an object remains `UNKNOWN`;
- complete-table exact empty evidence remains `OBSERVED_EMPTY`;
- query-grid verification remains mandatory for proxy readiness.

## Verification

```text
data-prep tests = 107 passed
EPDMS tests     = 133 passed
compileall      = passed
```

HF v10:

```text
schema           = D:\300_clip_nurec\hf_probe\schema_v8
audit            = D:\300_clip_nurec\hf_probe\audit_v10
prepared         = D:\300_clip_nurec\hf_probe\prepared_v10
raw conditions       = 16
unique conditions    = 16
duplicate conditions = 0
audit errors         = 0
query_grid.verified  = true
CF queries           = 41
TTC queries          = 51
proxy-ready          = 0 / 1
```

Remaining HF blockers:

```text
TIME_ALIGNMENT_UNRESOLVED
COORDINATE_UNRESOLVED
OBSERVATION_COVERAGE_INCOMPLETE
DAC_GEOMETRY_UNVERIFIED
```

No scorer, evaluator, metric, raw-data, time-alignment, coordinate-transform,
DAC, or official EPDMS implementation was changed.
