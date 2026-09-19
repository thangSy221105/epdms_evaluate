# NuRec merge-gate handover

BASE_COMMIT: `6eac07e830fcab6f823b5791d2aba9b4ecca007a`
BRANCH: `fix/nurec-data-prep-final-parity`
NEW_COMMIT: `<filled after commit>`

## Scope

This merge-gate fix is limited to NuRec data-preparation parity and readiness
provenance. It does not modify `tools/epdms/`, scorer behavior, evaluator
behavior, metric formulas, proxy weights, NAVSIM, raw NuRec files, time
alignment, coordinate transforms, or DAC conversion.

## Closed findings

- Object observation coverage mirrors `evaluate_query_coverage()` directly:
  CF uses a 50 ms nearest-observation tolerance and TTC uses 100 ms.
- Object observations do not require independent frame evidence.
- Confirmed empty remains exact-timestamp-only and requires verified obstacle
  table completeness; object evidence wins when both cover a query.
- Coverage counts preserve `observed + empty + unknown + missing == required`.
- Query-grid provenance participates in readiness. No-config defaults are
  explicitly unverified; invalid grids fail closed with a distinct blocker.
- Prediction duplicate accounting remains raw/unique/duplicate, with only the
  canonical unique condition set used for consistency checks.

## Verification

```text
data-prep tests = 100 passed
EPDMS tests     = 133 passed
compileall      = passed
```

HF v9 outputs:

```text
schema           = D:\300_clip_nurec\hf_probe\schema_v7
audit            = D:\300_clip_nurec\hf_probe\audit_v9
prepared         = D:\300_clip_nurec\hf_probe\prepared_v9
raw conditions       = 16
unique conditions    = 16
duplicate conditions = 0
audit errors         = 0
proxy-ready          = 0 / 1
query_grid.verified  = true
query_grid.source    = configs\epdms_300.json
CF queries           = 41
TTC queries          = 51
```

Remaining evidence-based blockers on the HF clip:

```text
TIME_ALIGNMENT_UNRESOLVED
COORDINATE_UNRESOLVED
OBSERVATION_COVERAGE_INCOMPLETE
DAC_GEOMETRY_UNVERIFIED
```

No `QUERY_GRID_CONFIG_UNVERIFIED` blocker is present when the explicit config
is supplied.
