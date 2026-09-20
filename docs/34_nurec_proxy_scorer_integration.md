# NuRec proxy scorer integration

This branch integrates the frozen NuRec contracts into the custom
`nurec_safety_proxy_v1` path. It does not implement official NAVSIM profiles
and it does not reopen time, obstacle geometry, or empty-frame semantics.

## Input adapter

`tools/epdms/nurec_inputs.py` decorates copies of prediction, GT, and context
rows. It reuses the verified per-clip mapping

```text
NuRec timestamp = PAI clip-relative timestamp + offset_us
```

to put obstacle timestamps on the scorer's clip-relative clock. The offset is
read from the accepted full-300 sidecar; it is never re-estimated. The adapter
also carries the accepted frozen coordinate contract and explicitly records
that `world_to_nre` is not applied to ego trajectories.

Raw files are not edited.

## Fail-closed statuses

Every pilot logical condition retains `clip_id|mode|alpha`. Missing CF/TTC
evidence produces `INCOMPLETE_OBSERVATION_EVIDENCE` with null CF/TTC/proxy;
absence of obstacle rows is never treated as an empty scene. Missing or
unvalidated map transform produces `MAP_NOT_READY` with null DAC/proxy.

The status fields are serialized by `EvaluationScoreRecord`:

```text
metric_status
cf_status
ttc_status
dac_status
coordinate_status
time_mapping_status
observation_status
overall_score_status
```

## Readiness result

The bounded audit writes versioned artifacts to
`D:\\300_clip_nurec\\hf_probe\\proxy_scorer_integration_v1`.

The current result is:

```text
OBSERVATION_READY_CLIP_COUNT = 287
OBSERVATION_INCOMPLETE_CLIP_COUNT = 13
MAP_SOURCE_AVAILABLE_CLIP_COUNT = 144
DAC_READY_CLIP_COUNT = 0
```

The 144 map-source clips contain local lane/intersection geometry, but the
production map transform into the scorer's `EGO_AT_T0` frame is not validated
by this branch. Consequently the 5-clip pilot emits 80 explicit
`MAP_NOT_READY` records and does not run a full 4,800-condition experiment.

## Reproduction

```powershell
py -3 scripts/audit_nurec_proxy_scorer_integration.py
```

To run the regular evaluator with the adapter, pass the coordinate contract,
observation readiness, time mapping, and (only when proven) DAC readiness
manifests through the corresponding `--nurec-*` options. Without a DAC
readiness manifest the evaluator fails closed for DAC.

The custom formula remains:

```text
S_proxy = CF * DAC_p * (5*TTC_p + 5*EP_GT + 2*FC) / 12
```

It must not be described as official NAVSIM EPDMS.
