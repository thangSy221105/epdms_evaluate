# NuRec obstacle frame provenance

The public trace now separates three claims:

1. NCore `CuboidTrackObservation` carries an explicit reference frame and
   timestamp.
2. Instant-NuRec `consolidate_cuboid_tracks()` evaluates that reference frame
   to `world`, applies `T_world_world_base`, and stores track poses described as
   world-frame poses.
3. The local extracted `sequence_tracks.json` has matching field structure,
   but its exact serializer/revision is not pinned in the local package.

Therefore the current result is `STRONGLY_SUPPORTED_NOT_VERIFIED` for the local
sequence-track frame and `PARTIALLY_VERIFIED` for the obstacle transform. The
audit does not apply `inverse(T_rig_scene(t0)) @ T_object_scene` until local
serialization lineage and the shared NuRec scene-frame relation are pinned.

The local artifact reports NuRec version `26.4.96-91b06fb8`, with
`sequence_tracks` export enabled. The public repositories are inspected at
their current `main` revisions, so compatibility is `LIKELY_COMPATIBLE`, not
an exact-version claim.

The official PAI obstacle parquet is not present in the cached five-clip
egomotion chunks. Consequently the cross-check uses direct local
`sequence_tracks.json` versus flattened NuRec obstacle rows only where exact
track ID and timestamp pairs exist; it does not infer identity from position.

Completeness remains unresolved. Camera/LiDAR/pose ranges show that source
frames exist, but they do not prove cuboid annotation completeness. Nearby
object timestamps are reported as temporal object evidence only, never as full
observation coverage or confirmed-empty evidence.
