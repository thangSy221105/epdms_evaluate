# NuRec OpenDRIVE DDC proxy audit

This round inventories the released NuRec USDZ central directories and
recovers only the `map.xodr` member.  All 300 clips in the full-300 manifest
were scanned and their XODR files parsed.  Geometry is present, and each file
contains a `geoReference`, but this is not by itself a proof that XODR
coordinates are `NCORE_LOCAL_WORLD` coordinates.

The current files contain no explicit OpenDRIVE lane `direction` attributes.
The audit therefore does not turn lane sign/order into legal travel direction,
does not fit a transform from GT or local lane geometry, and does not infer a
DDC value from the existing `lane_direction` field.

The interpretation follows the [ASAM OpenDRIVE lane-group direction
semantics](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/1.8.0/specification/11_lanes/11_02_lane_groups.html)
and its [lane-numbering rules](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.8.1/specification/16_annexes/map_rules.html).
In particular, reference-line orientation is not assumed to be driving
direction; this is also called out by the [OSI logical-lane
contract](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/gen/structosi3_1_1LogicalLane.html).

The successor profile is deliberately fail-closed:

* `ddc_proxy_implemented = false`;
* `ddc_proxy_enabled = false`;
* `ddc_proxy = null` for all 4,800 rows;
* official `ddc` remains null and remains in `missing_components`;
* the exact 4,800 `record_key` set is preserved from the existing vector.

The required explicit evidence for a future implementation is an authoritative
XODR-to-NuRec coordinate binding and a provenance-backed legal-direction
contract.  Only after both are verified should lane association and the
one-second DDC window be implemented.
