# NuRec obstacle normalization and CF/TTC readiness

This branch supersedes the earlier candidate-only obstacle normalization
audit for the production data-preparation path.

## Frozen geometry contract

`sequence_tracks.json` is consumed as the authoritative obstacle geometry
source in the verified target frame:

```text
coordinate_frame = NCORE_LOCAL_WORLD
obstacle_transform_status = VERIFIED
obstacle_geometry_block = CLOSED
```

The normalizer copies the full `tracks_data.tracks_poses` timeline and the
track-constant `cuboidtracks_data.cuboids_dims` values. It applies no SE(3)
transform, fitted correction, or new time offset. The seven accepted center
anomaly rows remain in the output with
`geometry_quality_status=RETAINED_LOCALIZED_ANOMALY`.

The generated context contains `semantic_context.obstacle.all_obstacles` and
explicit obstacle-frame provenance. It deliberately does not assert global
prediction/GT coordinate compatibility; that remains a separate contract.

## Observation semantics

The readiness audit uses the scorer's CF timeline (t0 plus 40 future poses at
10 Hz) and imports
`tools.epdms.observation_contract.build_ttc_projection_timestamps` for TTC.
Object rows are classified as `OBJECTS_PRESENT`. A query without an object row
is `MISSING` unless an explicit frame-level empty attestation exists. Sensor
timestamps, pose timestamps, cadence gaps, and absence of object rows are never
converted into `CONFIRMED_EMPTY`.

The five-clip pilot currently has 205 CF queries and 255 TTC queries, all with
object evidence within the existing tolerances. Therefore pilot strict
readiness is true, while empty-label semantics remain unresolved because no
explicit empty-frame attestation was found. Full-300 readiness is
`NOT_EVALUATED` and must be audited before 4,800-condition scoring.

Run the production normalizer and readiness audit with:

```powershell
$py = "C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/prepare_nurec_obstacles.py `
  --manifest configs/nurec_coordinate_multiclip_manifest.jsonl `
  --time-alignment-jsonl configs/nurec_coordinate_time_contract_5clip.jsonl `
  --geometry-audit-dir "D:\300_clip_nurec\hf_probe\cuboid_final_outlier_closure_v1" `
  --output-root "D:\300_clip_nurec\hf_probe\cf_ttc_observation_readiness_v1"

& $py scripts/audit_nurec_observation_completeness.py `
  --manifest configs/nurec_coordinate_multiclip_manifest.jsonl `
  --time-alignment-jsonl configs/nurec_coordinate_time_contract_5clip.jsonl `
  --geometry-audit-dir "D:\300_clip_nurec\hf_probe\cuboid_final_outlier_closure_v1" `
  --output-root "D:\300_clip_nurec\hf_probe\cf_ttc_observation_readiness_v1"
```

No scorer, evaluator, raw data, or strict-mode semantics are changed by this
preparation/audit path.
