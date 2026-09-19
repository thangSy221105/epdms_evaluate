# NuRec time-alignment evidence

Branch: `feat/nurec-time-alignment-evidence`
Pilot clip: `00040136-e651-4abd-991d-0655ccda9430`
Output: `D:\300_clip_nurec\hf_probe\time_alignment_v1`

## Pilot v2 scientific status

```text
TIME_ALIGNMENT_STATUS = UNRESOLVED
TIME_ALIGNMENT_VERIFIED = false
```

The hardened forensic tool does not infer an offset from numeric ranges,
obstacle minimums, egomotion minimums, first sensor frames, or curve fitting.
No verified context patch was created and no raw NuRec file was modified.

## Answers to the ten required questions

1. Prediction has 16 conditions, common `t0_us=5100000`, and actual relative
   waypoint timelines at `clean_waypoints[*].t_s` and
   `guided_waypoints[*].t_s`: 64 points, 0.1 to 6.4 seconds, median step 0.1 s.
   This is relative trajectory evidence, not a NuRec global-origin proof.
2. GT has `t0_us=5100000` and `ego_future_xyz` with 64 timestamp-less points;
   no GT waypoint timestamp field is present. The tool reports
   `TIMESTAMP_IMPLICIT_BY_PIPELINE` and does not infer timestamps from index.
3. NuRec obstacle time is `clipgt/obstacle.parquet:key.timestamp_micros`, with
   microseconds explicit in the field name.
4. Egomotion time is
   `clipgt/egomotion_estimate.parquet:key.timestamp_micros`, also explicitly
   microseconds.
5. No `frames/` directory was present in the pilot raw clip, so no camera
   filename timestamp clock was available as evidence.
6. `data_info.json` contains raw sensor/pose range and sequence offset fields,
   but the inspected report does not establish that prediction/GT `t0_us` is
   the same origin as those NuRec ranges.
7. No verified prediction-to-NuRec mapping was found.
8. Mapping source: none. `timestamp_correspondences.csv` records zero
   semantically identified frame/event pairs.
9. Units are explicit for `t0_us`, `timestamp_micros`,
   `timestamp_microseconds`, and similarly suffixed fields. Unsuffixed
   timestamp-like fields remain `UNIT_AMBIGUOUS`; unit was never inferred from
   magnitude.
10. Mapping is not verified.

## Exact blockers

```text
MISSING_EXPLICIT_RELATIVE_TO_GLOBAL_ORIGIN
MISSING_FRAME_CORRESPONDENCE
CLOCK_DOMAIN_NOT_PROVEN
```

Numeric range relationships are retained as diagnostics only. In particular,
`obstacle_min_timestamp` and `egomotion_min_timestamp` were not used as an
origin, and no `prediction_t0 - obstacle_min` correction was generated.

## Pilot outputs

```text
timestamp_inventory.csv
clock_domain_matrix.csv
timestamp_correspondences.csv
time_schema_report.json
time_schema_report.md
time_alignment_evidence.json
time_alignment_report.md
```

The pilot had 16 prediction conditions, one GT record, zero verified mapping
pairs, zero JSONL/parquet read errors, and final status `UNRESOLVED`. Since time
alignment was not verified, the existing main audit was not rerun into
audit_v11/prepared_v11.
