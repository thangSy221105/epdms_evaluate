# CF/TTC 13-clip empty-attestation audit

This branch audits only the 13 clips identified by the full-300 readiness
report. It does not reopen time mapping, NuRec world/ego binding, cuboid
geometry, acquisition completeness, or scorer semantics.

## Input gate

The orchestration script first reads:

`D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5`

and stops with `TRIAGE_INPUT_REGRESSION` unless it reproduces:

- 13 affected clips;
- 257 CF missing queries;
- 305 TTC missing queries.

Missing queries are deduplicated by `(clip_id, query_timestamp_us)`. The run
found 317 unique timestamps: 12 CF-only, 60 TTC-only, and 245 shared.

## Evidence inspection

For every affected clip the script inventories the complete USDZ central
directory using HTTP Range requests. It downloads only component members under
the output root, never a full USDZ. The inspected sources include:

- existing local `sequence_tracks.json`;
- `clipgt/obstacle.parquet`;
- `clipgt/clip.parquet`;
- `clipgt/association.parquet`;
- `data_info.json`, `datasource_summary.json`, `metadata.yaml`,
  `parsed_config.yaml`, `pose_record.json`, and `rig_trajectories.json`.

Sensor/pose/clip ranges are recorded separately from label evidence. A sensor
or pose timestamp does not attest that an annotation set exists.

## Exact classification rule

`CONFIRMED_EMPTY` requires an exact query timestamp and an authoritative
zero-object label-set record. A nearby timestamp, a gap between object rows,
or a sensor frame is insufficient. Object evidence remains higher priority;
an object/empty conflict is recorded and fails closed.

The run produced:

| Classification | Unique timestamps |
|---|---:|
| `CONFIRMED_EMPTY` | 0 |
| `OBJECT_LABEL_GAP` | 0 |
| `OUTSIDE_LABEL_TIMELINE` | 180 |
| `ANNOTATION_INPUT_ERROR` | 0 |
| `UNRESOLVED` | 137 |

Because no exact empty attestation was found, the repository sidecar
`configs/nurec_confirmed_empty_full300.jsonl` was not created.

## Final result

Output root:

`D:\300_clip_nurec\hf_probe\cf_ttc_13clip_empty_attestation_v1`

- Component downloads: 117
- Downloaded bytes: 61,821,049
- Full clip downloads: 0
- CF: 12,043 object-present, 0 confirmed-empty, 257 missing
- TTC: 14,995 object-present, 0 confirmed-empty, 305 missing
- Full-300 evaluated clips: 300
- `CF_DATA_READY_FULL_300 = false`
- `TTC_DATA_READY_FULL_300 = false`
- `LABEL_SET_EMPTY_SEMANTICS_STATUS = UNRESOLVED_NO_AUTHORITATIVE_EMPTY_ATTESTATION`
- `PHYSICAL_WORLD_OBSTACLE_COMPLETENESS = NOT_CLAIMED`
- `OBSTACLE_GEOMETRY_BLOCK = CLOSED`

The authoritative files are:

- `final_13clip_observation_closure_summary.json`;
- `13clip_observation_gap_evidence.csv`;
- `13clip_observation_gap_summary.csv`;
- `remaining_unresolved_queries.csv`;
- `confirmed_empty_evidence.csv`;
- `nurec_13clip_container_member_inventory.csv`;
- `annotation_source_inventory.csv`.

The remaining blocker is label-set evidence for 137 in-label-range unresolved
timestamps; the other 180 queries are outside the available object-label
timeline. Neither class is converted to empty or safe.
