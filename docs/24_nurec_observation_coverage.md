# NuRec observation coverage

Coverage is evaluated on the existing repository contract: CF uses the
4-second, 10-Hz future grid and a 50,000-us nearest-observation tolerance;
TTC extends those grid points by 0.2-second projections through a 1-second
horizon with a 100,000-us tolerance. These values mirror the existing
`tools/epdms/observation_contract.py` behavior and are not scorer changes.

Each query is classified as `EXACT_OBJECT`,
`NEAREST_OBJECT_WITHIN_TOLERANCE`, `EXACT_CONFIRMED_EMPTY`, `UNKNOWN`, or
`OUT_OF_RANGE`. A missing obstacle row is never converted to confirmed empty.
Confirmed empty requires an independent complete-frame attestation, which is
not present in the current five extracted packages. Ego-pose timestamps alone
are explicitly excluded as empty-scene evidence.

Consequently the current result is:

```text
EMPTY_SCENE_ATTESTATION_STATUS = UNRESOLVED_EMPTY_ATTESTATION
CF_DATA_READY = false
TTC_DATA_READY = false
```

The output includes per-query CSVs, track sanity flags, source cross-checks,
and per-clip readiness summaries. The next block should obtain or prove the
NCore frame/consolidation provenance and an independent complete observation
timeline before wiring normalized obstacles into CF/TTC.
