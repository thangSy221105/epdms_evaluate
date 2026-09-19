# NuRec EPDMS data-preparation hardening

This document records the hardening pass on top of
`d93144a270d55ae39e9946ac690531168a93a787` (`fix(data): support nested NuRec
parquet contracts`). The work is isolated on
`fix/nurec-data-prep-readiness-hardening` and is limited to data preparation,
tests, and documentation. No scorer, metric, evaluator, NAVSIM, run-identity,
resume, or raw NuRec file was changed.

## Hardening implemented

- DAC readiness is fail-closed. Lane/intersection geometry is reported as a
  structural candidate only; it does not make `map.ready` or
  `dac_geometry_verified` true. Missing `drivable_space.parquet` is therefore
  visible and cannot become a safe DAC result.
- Timestamp resolution uses a single selected physical field. The canonical
  nested `key.timestamp_micros` field is preferred. `frame_id`, `sample_id`,
  and `token` are never used as physical timestamps. Ambiguous or invalid
  timestamp candidates remain unresolved, with per-field counts and summaries.
- Numeric range overlap is recorded only as a diagnostic. It is never an
  alignment proof. Explicit time mapping requires both `verified == true` and
  a non-empty provenance source.
- Obstacle validation is row-level and strict: finite integer-compatible
  timestamp, non-empty track ID/category, finite center, strictly positive
  size, and finite non-degenerate quaternion. Invalid rows are counted and
  never silently dropped.
- Malformed JSONL, non-object rows, missing clip IDs, and duplicate clip IDs
  are written to `audit_errors.csv`; duplicates block readiness.
- Nested schema discovery is bounded to 64 non-null samples and supports
  list-of-struct paths such as `obstacle.center.x` and map point lists.
- Map checks use `structurally_valid_geometry_count` and
  `structurally_invalid_geometry_count`. They do not claim semantic polygon
  or DAC validity without an explicit contract.
- Egomotion reports actual nested position/rotation fields and separates
  transform metadata availability from a verified transform chain.
- Observation coverage uses independent frame evidence only. Obstacle row or
  timestamp counts are not treated as frame counts, and empty frames are not
  fabricated from missing obstacle rows.

## Official Hugging Face probe

Clip:

`00040136-e651-4abd-991d-0655ccda9430`

Raw staging root:

`D:\300_clip_nurec\hf_probe\dataset\00040136-e651-4abd-991d-0655ccda9430`

Outputs from this hardening pass (v5 was not overwritten):

- `D:\300_clip_nurec\hf_probe\schema_v4`
- `D:\300_clip_nurec\hf_probe\audit_v6`
- `D:\300_clip_nurec\hf_probe\prepared_v6`

Verified raw facts:

```text
obstacle schema              = key.timestamp_micros + nested obstacle fields
obstacle rows                = 3,287
valid obstacle rows          = 3,287
invalid obstacle rows        = 0
unique obstacle timestamps   = 3,287
unique obstacle tracks       = 78
obstacle categories          = automobile, person, rider, trailer
egomotion rows               = 202
lane geometry                = 235 / 235 structurally valid
intersection geometry        = 4 / 4 structurally valid
road boundary geometry       = 246 / 246 structurally valid
drivable_space.parquet       = missing
DAC result                   = lane_plus_intersection, candidate only
proxy-ready clips            = 0 / 1
parquet engine               = pyarrow 25.0.1, READY
```

The selected obstacle timestamp field is `key.timestamp_micros`. The clip
interval is approximately `27,563,309,000` to `27,583,309,000` microseconds;
the prediction/GT `t0_us` is `5,100,000`. No explicit mapping was found and no
offset was applied.

## Readiness result and true blockers

The prepared contract for the probe is intentionally not ready:

```text
ready_for_proxy                 = false
TIME_ALIGNMENT_UNRESOLVED       = true
COORDINATE_UNRESOLVED           = true
OBSERVATION_COVERAGE_INCOMPLETE = true
DAC_GEOMETRY_UNVERIFIED         = true
PREDICTION_DUPLICATE            = true
```

`rig_trajectories.json` and `calibration_estimate.parquet` provide transform
metadata candidates, but the chain and frame/anchor semantics are not proven,
so `transform_metadata_available` is not treated as
`transform_chain_verified`.

The prediction JSONL contains repeated rows for the same clip ID. The audit
does not guess a hidden condition-level identity: every duplicate is visible
in `audit_errors.csv` and the clip is blocked until the source identity
contract is clarified. This is a source-data blocker, not a reason to discard
rows silently.

## Tests and scope checks

```text
data-prep tests       = 49 passed
EPDMS regression      = 133 passed
compileall            = passed
```

The new data-preparation tests cover canonical and ambiguous timestamps,
invalid obstacle rows, nested/list schema access, malformed and duplicate
JSONL, structural map checks, transform metadata versus verification,
observation evidence, clock mismatch, and fail-closed readiness. Existing
EPDMS tests remain green.

The pipeline still preserves raw data, performs no coordinate transform, does
not alter clock domains, and does not modify evaluator/scorer behavior. The
next phase may consume the prepared contracts only after explicit time,
coordinate, observation, and DAC semantic contracts are established.
