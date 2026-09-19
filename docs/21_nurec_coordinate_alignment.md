# NuRec coordinate alignment audit

This block audits coordinate provenance without changing the scorer, evaluator,
raw NuRec files, or prediction/GT data.

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
pilot report directory.

## Current status

The pilot is **PARTIALLY_VERIFIED**:

- GT frame: `VERIFIED_EGO_AT_T0`.
- NuRec rig pose chain: verified as a dynamic rig-to-anchor/world chain from
  `T_rig_worlds` and the upstream converter documentation.
- Prediction frame: unresolved because the prediction JSONL has no explicit
  coordinate-frame declaration in the inspected row.
- Obstacle frame: unresolved. Flattened `obstacle.parquet` preserves center,
  size, orientation, and `egomotion_label_class_id`, but not
  `reference_frame_id` or `reference_frame_timestamp_us`.
- Map frame: unresolved. Lane/intersection/road-boundary geometry is present,
  but those parquet records do not declare the geometry reference frame.
- `drivable_space.parquet` is absent in the pilot package, so DAC coordinate
  readiness is false.

Therefore `COORDINATE_ALIGNMENT_VERIFIED` remains false. This is intentional:
the pose validation supports a transform algorithm, but missing obstacle/map
frame metadata prevents a complete scoring contract.

## Reproducible command

```powershell
$py = "C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/audit_nurec_coordinate_alignment.py `
  --nurec-clip-dir "D:\300_clip_nurec\hf_probe\official_nvidia_00040136" `
  --prediction-jsonl "D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl" `
  --ground-truth-jsonl "D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl" `
  --output-dir "D:\300_clip_nurec\hf_probe\coordinate_alignment_v1\00040136-e651-4abd-991d-0655ccda9430" `
  --clip-id "00040136-e651-4abd-991d-0655ccda9430"
```

Coordinate code readiness is separate from coordinate data readiness. The
former is true for the audited pose chain; the latter remains false until the
prediction, obstacle, and map contracts are explicit.
