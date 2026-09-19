# NuRec EPDMS Data Preparation — Round 1

Branch: `feat/nurec-data-preparation`

This branch contains only local NuRec inspection, inventory, contract audit,
and non-destructive staging. It does not modify the scorer, metrics,
`evaluate_single_condition`, run identity, resume logic, EPDMS formula, or
NAVSIM code.

## Implemented commands

```powershell
python scripts/inspect_nurec_clip.py `
  --clip-dir D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2\<clip_id> `
  --output-dir D:\300_clip_nurec\05_data_contract\schema\<clip_id>

python scripts/audit_nurec_dataset.py `
  --dataset-root D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2 `
  --prediction-jsonl D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl `
  --ground-truth-jsonl D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl `
  --context-jsonl D:\300_clip_nurec\01_context\full\nurec_context_full_300.jsonl `
  --output-dir D:\300_clip_nurec\05_data_contract

python scripts/prepare_nurec_epdms_data.py `
  --audit-dir D:\300_clip_nurec\05_data_contract `
  --output-dir D:\300_clip_nurec\05_prepared_epdms
```

## Real clip inspected

Clip: `00040136-e651-4abd-991d-0655ccda9430`

Sample outputs:

- `D:\300_clip_nurec\05_data_contract\schema\00040136-e651-4abd-991d-0655ccda9430\schema_report.json`
- `D:\300_clip_nurec\05_data_contract\schema\00040136-e651-4abd-991d-0655ccda9430\schema_report.md`

The first Round-1 run was recorded before a parquet engine was available, so
that historical report did not guess parquet columns. The current runtime has
`pyarrow 25.0.1`, and the subsequent official Hugging Face probe verified the
raw nested parquet schema. The original Round-1 JSON inspection had already
established:

- `pose_record.json` contains `alignment_origin` and `record[].timestamp_microseconds`.
- `rig_trajectories.json` contains `T_world_base`, `world_to_nre`, camera/lidar frame timestamp arrays, world/rig transforms, and camera/lidar calibration transforms.
- The derived full-context summary reports obstacle rows with `timestamp_micros`, `trackline_id`, `category`, `center`, `size`, and `orientation`.
- The derived context sample for this clip reports 3,287 obstacle rows and categories automobile/person/trailer/rider. This is context-summary evidence, not a substitute for reading raw parquet.
- The selected filtered clip does not contain every requested optional file; the report marks each absent file explicitly.

## Dataset-wide result

Reports were written to:

`D:\300_clip_nurec\05_data_contract`

Staging manifests were written to:

`D:\300_clip_nurec\05_prepared_epdms`

Observed summary:

```text
total_clips                     = 300
prediction_ready                = 300
gt_ready                        = 300
time_ready                      = 0
coordinate_ready                = 0
observation_cf_ready            = 0
observation_ttc_ready           = 0
map_ready                       = 0
proxy_ready_clip_count          = 0
dataset_status                  = DATASET_NOT_READY
parquet_engine                  = PARQUET_ENGINE_UNAVAILABLE (historical Round-1 run)
```

Blocker counts:

```text
TIME_ALIGNMENT_UNRESOLVED       = 300
COORDINATE_UNRESOLVED           = 300
OBSERVATION_COVERAGE_INCOMPLETE = 300
OBSTACLE_SCHEMA_INVALID         = 300
MAP_MISSING                     = 156
PARQUET_ENGINE_UNAVAILABLE      = 144 (historical Round-1 run)
```

The audit does not turn `prediction_t0_us - obstacle_min_timestamp` into a
verified offset. It records such a difference only as a diagnostic candidate.

## Answers to the eight handover questions

1. Obstacle schema observed from the available derived context is
   `timestamp_micros`, `trackline_id`, `category`, `center{x,y,z}`,
   `size{x,y,z}`, and `orientation{x,y,z,w}`. Raw parquet schema remains
   pending a parquet engine.
2. Obstacle timestamps in the available context sample are approximately
   `27,563,348,644` microseconds, while AR1 prediction `t0_us` is `5,100,000`.
   The clock relationship is therefore not verified; no offset was applied.
3. No field proving the AR1-to-NuRec timestamp mapping was found by the
   conservative audit. `pose_record.json` and `rig_trajectories.json` expose
   timestamp/pose evidence, but the relation to AR1 `t0_us` still needs an
   explicit schema mapping.
4. `rig_trajectories.json` exposes pose/transform-like data. Whether
   `egomotion_estimate.parquet` is sufficient for the required transform cannot
   be concluded until parquet columns are readable.
5. Obstacle/map coordinate frames are not verified. No complete common
   frame+anchor declaration was available in the joined audit inputs.
6. Empty-frame evidence is not established. An empty obstacle table alone is
   classified as `UNKNOWN`; the pipeline only accepts independent frame
   evidence as a candidate for `OBSERVED_EMPTY`.
7. `drivable_space.parquet` cannot yet be approved as a direct DAC source.
   It is inventoried as a candidate only; geometry validity and frame metadata
   require parquet inspection.
8. Truly READY clips for the current proxy: **0 / 300**.

## Raw-data policy

No raw NuRec file was edited. No timestamp was shifted, no frame was renamed,
no transform was applied, no empty evidence was fabricated, and no missing map
was converted into a safe DAC result.

## Next data action

The initial 300-clip run used a runtime without a parquet engine and is kept as
historical evidence. The current runtime has `pyarrow 25.0.1`; the official
Hugging Face probe below is the first raw-parquet validation. The evaluator
should remain unchanged until the audit identifies verified time and
coordinate contracts.

## Follow-up: official Hugging Face real-clip probe

After authenticating to Hugging Face, the official NuRec clip matching the
local 300-clip set was probed:

`00040136-e651-4abd-991d-0655ccda9430`

Only the USDZ annotation/metadata entries were fetched through HTTP Range;
checkpoint, volume, mesh, camera frames, and video were not downloaded. The
probe staging root is:

`D:\300_clip_nurec\hf_probe`

With `pyarrow 25.0.1`, the real nested parquet schema is now readable:

- obstacle: `key.timestamp_micros`, `obstacle.trackline_id`,
  `obstacle.category`, `obstacle.center{x,y,z}`, `obstacle.size{x,y,z}`,
  `obstacle.orientation{x,y,z,w}`;
- egomotion: `key.timestamp_micros`,
  `egomotion_estimate.location{x,y,z}` and
  `egomotion_estimate.orientation{x,y,z,w}`;
- lane geometry: `lane.left_rail` and `lane.right_rail`;
- intersection/road-boundary geometry: `*.location` point lists.

The nested-field audit fix is covered by the data-preparation tests. The
historical Round-1 reports are:

- `D:\300_clip_nurec\hf_probe\schema_v3\schema_report.md`
- `D:\300_clip_nurec\hf_probe\audit_v5\dataset_readiness_summary.md`
- `D:\300_clip_nurec\hf_probe\prepared_v5\00040136-e651-4abd-991d-0655ccda9430\contract.json`

The subsequent hardening reports are `schema_v4`, `audit_v6`, and
`prepared_v6`. The current condition/coverage pass uses `schema_v5`,
`audit_v7`, and `prepared_v7`; none of these reports overwrite the historical
outputs.

Observed for this clip:

```text
obstacle rows                  = 3,287
obstacle timestamps            = 3,287 unique
obstacle tracks                = 78
invalid obstacle rows          = 0
egomotion rows                 = 202
lane geometry                  = 235 / 235 valid
intersection geometry          = 4 / 4 valid
road boundary geometry         = 246 / 246 valid
drivable_space.parquet         = missing
recommended DAC source         = lane_plus_intersection (candidate only)
proxy-ready                    = false
```

The remaining blockers are intentionally conservative: AR1 `t0_us` is still
not explicitly mapped to NuRec global timestamps, frame/anchor agreement is
not verified although transform metadata is available as a candidate, and
observation coverage cannot be certified until that time contract is proven.
