# NuRec semantic time correspondence

The alignment resolver now accepts semantic pairs only when prediction, GT, and raw NuRec candidates share an explicit identity and each identity is colocated with a timestamp in the same source record (or an explicit mapping declares that relationship).

Equal numeric timestamps, equal array positions, matching `t0_us` values, and accidental repeated IDs are not sufficient. With fewer than two constant-offset verified pairs, the result remains unresolved. Inconsistent offsets produce `CONFLICTING_TIME_ORIGIN`.

The pilot emits:

- `prediction_gt_identity_candidates.json`
- `nurec_identity_candidates.json`
- `semantic_correspondences.csv`
- `timeline_consistency.csv`
- `t0_origin_report.json`
- `upstream_assignment_trace.json`
- `time_alignment_evidence.json`

Clean and guided trajectories are compared by role. Matching clean/guided grids are reported as consistent; they are not treated as a conflict. A different timestamp vector for corresponding conditions is a `PREDICTION_TIMELINE_CONFLICT` blocker.

The current NuRec pilot has timestamped raw pose/obstacle records, but no verified semantic bridge from the prediction/GT `t0_us` or timestamp-less `ego_future_xyz` arrays to those records. It must therefore remain `UNRESOLVED` until upstream provenance or a raw frame correspondence is supplied.
