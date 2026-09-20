# NuRec authoritative map transform and DAC readiness

This round freezes the map adapter for `nurec_safety_proxy_v1`.  ClipGT lane
and intersection geometry is read without modifying the raw NuRec package.

## Contract

The accepted NuRec coordinate contract declares map geometry in
`NCORE_LOCAL_WORLD` and scoring in `EGO_AT_T0`.  The deterministic transform is:

```text
p_EGO_AT_T0 = inverse(T_rig_world(nurec_t0_us)) @ [p_NCORE_LOCAL_WORLD, 1]
```

`T_rig_world` is interpolated at the already verified NuRec `t0` using linear
translation and quaternion SLERP.  `world_to_nre` is not used for ego or map
scoring.  No prediction/GT fit and no new time offset are allowed.

The 2D DAC adapter transforms the polygon x/y coordinates with z=0 and keeps
polygon topology.  Partial, malformed, or absent map sources remain
fail-closed; they are never interpreted as drivable space.

## Evidence boundary

NCore publicly documents the rig-to-local-world SE(3) convention.  The NuRec
dataset manifest publicly lists `lane.parquet`, `intersection_area.parquet`,
and `drivable_space.parquet` as ClipGT components.  The public repositories
reviewed for this round do not expose a writer that separately declares the
ClipGT map frame, so the map-frame binding is explicitly recorded as the frozen
local NuRec contract plus NCore convention, not as a claim that a public map
writer was found.

The audit therefore emits both `map_frame_upstream_evidence.csv` and
`map_coordinate_contract.json`.  These preserve the distinction between
public source evidence and local automated schema inspection.

## Readiness policy

`DAC_READY` requires strict valid map geometry, the verified map transform,
verified time mapping, and both existing observation readiness flags.  Missing
map data is reported as `MAP_SOURCE_MISSING`/`MAP_TRANSFORM_NOT_AVAILABLE`; it
does not produce DAC=1.  Physical-world obstacle completeness remains
`NOT_CLAIMED`, and the existing scorer fields `CF_DATA_READY` and
`TTC_DATA_READY` are unchanged by this round.

Run the audit with `scripts/audit_nurec_map_dac.py`.  Passing
`--run-pilot` additionally invokes the production CLI for five deterministic
DAC-ready clips and stores its output below the requested output directory.

