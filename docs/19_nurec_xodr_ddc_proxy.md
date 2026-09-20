# NuRec OpenDRIVE DDC proxy audit

This round inventories the released NuRec USDZ central directories and
recovers only the `map.xodr` member.  All 300 clips in the full-300 manifest
were scanned and their XODR files parsed.  NVIDIA's published NuRec coordinate
chain provides the per-clip binding:

```text
T_map_ncore = inverse(T_ecef_enu @ T_world_base)
```

where `T_world_base` is the frame-0 rig-to-ECEF pose and `T_ecef_enu` is
constructed from the XODR `geoReference`.  The audit implements this formula
without fitting a correction.  It separately records source availability,
implementation, and measured lane-geometry diagnostics.  The coordinate
contract is promoted to `VERIFIED` only when the validation gate itself is
accepted; a finite nearest-lane diagnostic is not silently promoted to proof.

The released files are OpenDRIVE 1.4.  They contain no explicit lane
`direction` attributes, and all 35,003 road records omit `road.rule`.  For
this exact version, ASAM specifies that a missing `road/@rule` defaults to
RHT, so the audit records `effective_road_rule=RHT` with source
`ASAM_DEFAULT` for regular roads.  This is distinct from
`road_rule_attribute_available=false`.  Junction roads are audited explicitly
rather than treated as ordinary lanes.  `connection.incomingRoad` is the
incoming road, `connection.connectingRoad` is the connecting road, and
`laneLink.from` / `laneLink.to` are resolved against exact lane IDs and lane
types.  A `contactPoint=start` connecting road runs along the lane-link
direction; `contactPoint=end` runs opposite to it.  The audit also checks the
predecessor/successor `road.link` records and the junction attributes on both
roads.  It never repairs, swaps, flips, or geometry-matches an inconsistent
topology.  Only `driving` lanes are eligible; non-driving and `bidirectional`
lanes are explicitly excluded from DDC readiness.

The aggregate legal-direction contract is therefore `PARTIAL`, not
`UNRESOLVED` merely because the XML attribute is absent.  The full-300
junction evidence is written to `xodr_junction_inventory_full300.csv`,
`xodr_junction_lanelink_full300.csv`,
`xodr_junction_topology_consistency_full300.csv`, and
`xodr_lane_type_inventory_full300.csv`; provenance rows are in
`upstream_junction_direction_evidence.csv`.

The interpretation follows the [ASAM 1.4/1.45 junction and lane-link
source](https://www.asam.net/fileadmin/Standards/OpenDRIVE/OpenDRIVEFormatSpecDelta_1.5M_vs_1.4H_VIRES.pdf),
the [ASAM official contact-point and road-link
semantics](https://www.asam.net/fileadmin/Standards/OpenDRIVE/ASAM_OpenDRIVE_BS_V1-7-0.html),
the [ASAM road-rule semantics, including the
1.4.0 missing-rule default](https://simulation.pages.asam.net/opendrive-group/opendrive-antora-gen/ASAM_OpenDRIVE_Specification/v1.9.0/specification/10_roads/10_01_introduction.html),
the [ASAM lane-group direction semantics](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.8.1/specification/11_lanes/11_02_lane_groups.html),
and the [ASAM 1.4.0 road-reference-line semantics](https://simulation.pages.asam.net/opendrive-group/opendrive-antora-gen/ASAM_OpenDRIVE_Specification/v1.8.1/specification/09_geometries/09_02_road_reference_line.html).
In particular, reference-line orientation is not assumed to be driving
direction, and newer lane-direction features are not used as evidence for a
1.4 file.

The successor profile is deliberately fail-closed:

* `ddc_proxy_implemented = false`;
* `ddc_proxy_enabled = false`;
* `ddc_proxy = null` for all 4,800 rows;
* official `ddc` remains null and remains in `missing_components`;
* the exact 4,800 `record_key` set is preserved from the existing vector.

The audit measures the frozen vector rather than hard-coding its counts.  It
does not claim that LK or `nurec_safety_proxy_v1` are unchanged by comparing
unavailable artifacts; it records `NOT_MUTATED_BY_THIS_AUDIT_SCRIPT` with
explicit provenance.  The required evidence for a future implementation is
an accepted XODR-to-NuRec coordinate validation gate, a provenance-backed
legal-direction contract including junction handling, and a later scorer
contract decision.  Only after those are verified should lane association and
the one-second DDC window be implemented.
