# Full-300 CF/TTC minimal-input readiness audit

Branch scope: `feat/cf-ttc-full300-minimal-readiness`.

This round audits observation availability for all experiment clip IDs without
downloading full NuRec clips. Stage 1 consumes only:

- `sequence_tracks.json` as the frozen authoritative obstacle source;
- an existing, explicitly verified per-clip `nurec_t0_us`/`offset_us` mapping.

The audit calls the production query-grid helper. It produces 41 CF queries per
clip (`t0` plus 40 poses at 10 Hz) and the production TTC projection helper
produces 51 unique TTC queries per clip. An object row within the existing
tolerance is `OBJECTS_PRESENT`; otherwise the query is `MISSING`. No missing
query is converted to `CONFIRMED_EMPTY`.

The audit writes the full-300 manifest and reports to:

`D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v1`

The fallback CSV lists only affected clips and keeps
`full_clip_download_required=false`. The audit never derives a global offset,
rederives a pilot offset, changes scorer code, or claims physical-world obstacle
completeness.

Run from the repository root:

```powershell
py -3 scripts/audit_cf_ttc_full300_minimal.py `
  --prediction-jsonl D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl `
  --ground-truth-jsonl D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl `
  --nurec-root D:\300_clip_nurec\01_context\reasoning_filtered\nurec_reasoning_filtered_300_v2 `
  --nurec-root D:\300_clip_nurec\hf_probe\coordinate_alignment_v1\nurec_full_metadata `
  --time-mapping-jsonl configs\nurec_coordinate_time_contract_5clip.jsonl `
  --manifest-output configs\nurec_cf_ttc_full300_manifest.jsonl `
  --output-root D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v1 `
  --pilot-regression-summary D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v1\pilot_regression\cf_ttc_readiness_final_summary.json
```

`CF_DATA_READY_FULL_300` and `TTC_DATA_READY_FULL_300` are true only when all
300 clips were evaluated and their required query sets contain zero missing
queries. `LABEL_SET_EMPTY_SEMANTICS_STATUS` remains `UNRESOLVED` unless an
independent frame-level empty attestation is supplied.
