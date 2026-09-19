# NuRec time-alignment evidence

Branch: `feat/nurec-time-alignment-evidence`
Pilot clip: `00040136-e651-4abd-991d-0655ccda9430`
Output: `D:\300_clip_nurec\hf_probe\time_alignment_v1`

## Final status

```text
TIME_ALIGNMENT_STATUS = UNRESOLVED
TIME_ALIGNMENT_VERIFIED = false
```

The forensic tool does not infer an offset from numeric ranges, obstacle
minimums, egomotion minimums, first sensor frames, or curve fitting. No
verified context patch was created and no raw NuRec file was modified.

## Answers to the ten required questions

1. Prediction `t0_us` is explicitly a microsecond-valued field by its field
   name. The prediction record does not declare a relationship between this
   value and the NuRec clock origin.
2. GT `t0_us` is explicitly a microsecond-valued field by its field name. The
   GT record does not contain a waypoint timestamp contract or declared origin.
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
pairs, and final status `UNRESOLVED`. Since time alignment was not verified,
the existing main audit was not rerun into audit_v11/prepared_v11.
