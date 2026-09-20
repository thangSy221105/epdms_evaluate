# NuRec partial EPDMS-style metric vector

This round adds an additive `epdms_partial_vector_lk_ddc_v1` output. It does
not rename or modify `nurec_safety_proxy_v1`, and it does not claim official
NAVSIM/EPDMS compatibility.

## Metric families

The official-compatible fields remain fail-closed:

```text
nc, dac, ddc, tlc, ttc, ep, lk, hc, ec
```

They are `null` until the corresponding official implementation and input
contract are verified. Proxy fields remain separate:

```text
collision_free_proxy
dac_proxy
ttc_proxy
progress_gt_proxy
future_comfort_proxy
lk_proxy
ddc_proxy
```

The existing `nurec_safety_proxy_v1` formula is unchanged.

## LK proxy

`tools/epdms/lane_metrics.py` reads `clipgt/lane.parquet`, validates
`left_rail`/`right_rail`, resamples both rails by arclength, and derives a
midpoint centerline. Rails and intersection polygons are transformed using the
existing verified NuRec `NCORE_LOCAL_WORLD -> EGO_AT_T0` transform. No fitted
correction is used.

The proxy uses a 0.5 m lateral-deviation threshold and a 2.0 s continuous
violation window. Intersection frames are excluded and reset the continuous
counter. Ambiguous lane association returns `null` rather than a violation.

## DDC direction contract

NuRec exposes a `lane_direction` field, but the inspected values (`STRAIGHT`,
`LEFT_TURN`, `RIGHT_TURN`, and branch/merge variants) do not by themselves
prove legal traffic direction. Therefore `ddc_proxy` is disabled and the
output contains only `lane_direction_alignment_diagnostic`.

## Pilot

The pilot consumes the existing 5-clip/80-record proxy artifact and writes new
files under:

```text
D:\300_clip_nurec\hf_probe\epdms_partial_vector_lk_ddc_v1
```

Observed pilot result:

```text
records                 = 80/80
lk_proxy valid          = 32
lk_proxy null           = 48 (ambiguous lane association)
lk_proxy zero           = 0
lk_proxy one            = 32
ddc_proxy valid         = 0
ddc_proxy null          = 80
official fields filled  = 0
official stage1 score   = null
```

The old pilot score directory is not overwritten. The stopped 4,800-condition
run is not resumed because its schema predates this additive vector.
