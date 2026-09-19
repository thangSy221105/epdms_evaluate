# NuRec coordinate alignment audit

This block audits coordinate provenance without changing the scorer, evaluator,
raw NuRec files, or prediction/GT data.

The hardening branch consumes an upstream per-clip time sidecar. It does not
derive an offset from NuRec timestamps. Pose queries are strict in-range only:
out-of-range queries raise `POSE_INTERPOLATION_OUT_OF_RANGE` with the query and
source bounds. Translation is linearly interpolated and rotation uses
quaternion SLERP. Localization uses the full transform
`inverse(T_rig_world(t0)) @ T_rig_world(t)`, which is equivalent for translation
to the upstream GT formula `R_t0^-1 @ (xyz_world - xyz_t0)`.

## Pilot evidence

The pilot is `00040136-e651-4abd-991d-0655ccda9430`. The NuRec package contains
`rig_trajectories.json` with `T_rig_worlds` and timestamps, plus an explicit
`world_to_nre` matrix. The checked NCore converter source documents
`T_rig_worlds` as rig-to-anchor transforms. The upstream Alpamayo loader creates
GT by querying future poses and applying
`R_t0^-1 @ (xyz_world - xyz_t0)`, so the GT contract is
`EGO_AT_T0`.

The numerical pilot check uses only the metadata-backed dynamic pose chain. It
does not fit translation, rotation, ICP, Procrustes, or a pilot-specific
matrix. The output is `coordinate_alignment_summary.json` in the external
pilot report directory. `prediction_frame_status` remains unresolved unless
the selected prediction row carries explicit frame provenance.

## Current status

The pilot is **PARTIALLY_VERIFIED**:

- GT frame: `SUPPORTED_BY_UPSTREAM_EGO_AT_T0_CONTRACT`; this wording does not
  overclaim exact file-generation lineage.
- NuRec rig pose chain: verified as a dynamic rig-to-anchor/world chain from
  `T_rig_worlds` and the upstream converter documentation.
- Prediction frame: unresolved because the prediction JSONL has no explicit
  coordinate-frame declaration in the inspected row.
- Obstacle frame: unresolved. Flattened `obstacle.parquet` preserves center,
  size, orientation, and `egomotion_label_class_id`, but not
  `reference_frame_id` or `reference_frame_timestamp_us`.
- Map frame: `PROVENANCE_AVAILABLE_NOT_INTEGRATED` when `T_world_base` and
  `map.xodr` georeference are present; lane/intersection/road-boundary
  geometry is still not wired into a scoring transform in this branch.
- `drivable_space.parquet` is absent in the pilot package, so DAC coordinate
  readiness is false.

Therefore `COORDINATE_ALIGNMENT_VERIFIED` remains false. This is intentional:
the pose validation supports a transform algorithm, but missing obstacle/map
frame metadata prevents a complete scoring contract.

`sequence_tracks.json` is inspected for track pose/timestamp availability but
is not consumed by the scorer or obstacle adapter in this branch.

## Reproducible command

```powershell
$py = "C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/audit_nurec_coordinate_alignment.py `
  --nurec-clip-dir "D:\300_clip_nurec\hf_probe\official_nvidia_00040136" `
  --prediction-jsonl "D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl" `
  --ground-truth-jsonl "D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl" `
  --time-alignment-jsonl "configs/nurec_coordinate_time_contract_5clip.jsonl" `
  --output-dir "D:\300_clip_nurec\hf_probe\coordinate_alignment_v1\00040136-e651-4abd-991d-0655ccda9430" `
  --clip-id "00040136-e651-4abd-991d-0655ccda9430"
```

Coordinate code readiness is separate from coordinate data readiness. The
former is true for the audited pose chain; the latter remains false until the
prediction, obstacle, and map contracts are explicit.
