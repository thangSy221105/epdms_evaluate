# NuRec obstacle normalization

This block audits NuRec obstacle sources without modifying the scorer,
evaluator, raw NuRec/PAI files, or map code. It reads `sequence_tracks.json`
and `clipgt/obstacle.parquet`, reuses the accepted per-clip time sidecar, and
writes new sidecars under the external obstacle-normalization output root.

The local NuRec package contains track poses as seven-tuples
`[x,y,z,qx,qy,qz,qw]`, per-track timestamps, labels, flags, and fixed cuboid
dimensions. `sequence_tracks.json` is therefore the strongest local candidate
for authoritative track geometry. The flattened obstacle parquet contains
centres, dimensions, quaternion orientation and timestamps, but does not retain
`reference_frame_id` or `reference_frame_timestamp_us`.

The NCore public contract defines cuboid observations as relative to an
explicit reference frame and exposes pose-graph transforms. The local
`sequence_tracks.json` does not carry an explicit source-frame declaration or
the consolidation provenance needed to prove that its poses are NuRec
world/scene poses. Therefore this block records
`sequence_tracks.json_candidate`, not a verified authoritative source, and
does not apply a guessed transform.

If the source frame is later proven to be NuRec world/scene, the intended
transform is:

```text
T_object_EGO_AT_T0 = inverse(T_rig_world(t0)) @ T_object_world
```

The implementation already provides this rigid-transform helper, including
full orientation composition, but keeps `normalization_verified=false` until
the frame contract is available.

Run the five-clip audit with:

```powershell
$py = "C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/prepare_nurec_obstacles.py `
  --manifest configs/nurec_coordinate_multiclip_manifest.jsonl `
  --time-alignment-jsonl configs/nurec_coordinate_time_contract_5clip.jsonl `
  --output-root "D:\300_clip_nurec\hf_probe\obstacle_normalization_v1"
```

The report deliberately distinguishes geometry rows from observation
attestation. Pose timestamps are not treated as evidence of an empty obstacle
frame, so `EMPTY_SCENE_ATTESTATION_STATUS=UNRESOLVED` keeps CF/TTC readiness
false even when every query has an object row nearby.
