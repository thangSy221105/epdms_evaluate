# Full-300 NuRec acquisition and CF/TTC readiness

This round acquires the minimum raw components required to replay the accepted
per-clip time contract and audit the CF/TTC observation grid for the complete
300-clip experiment manifest. It does not change the scorer, evaluator,
coordinate transforms, or DAC implementation.

## Sources and acquisition policy

- PAI source: `labels/egomotion.offline` from
  `nvidia/PhysicalAI-Autonomous-Vehicles`.
- NuRec source: `sequence_tracks.json` inside each
  `sample_set/26.04_release/<clip_id>/<clip_id>.usdz` member from
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`.
- Acquisition is component-level. ZIP central directories and individual
  members are read with HTTP Range requests; full NuRec USDZ files are not
  downloaded.
- ZIP64 central directories are supported.
- Joins use exact UUIDs. Existing files are schema-validated and reused.
- Raw data is stored outside the repository under
  `D:\300_clip_nurec\00_raw` and is not committed.

The official full-rate PAI `labels/egomotion` component was accidentally
downloaded during an early probe. It is not consumed by replay or audit. Its
size is recorded separately as `other_downloaded_bytes` in the acquisition
summary so the final accounting is reproducible.

## Reproducible commands

Acquire the two required components:

```powershell
python scripts/acquire_full300_data.py `
  --manifest configs/nurec_cf_ttc_full300_manifest.jsonl `
  --raw-root D:\300_clip_nurec\00_raw `
  --output-root D:\300_clip_nurec\hf_probe\full300_data_acquisition_v1
```

Replay the frozen historical time method:

```powershell
python scripts/replay_nurec_historical_time_mapping_full300.py `
  --repo-root . `
  --pai-root D:\300_clip_nurec\00_raw `
  --nurec-root D:\300_clip_nurec\00_raw\nurec_full300 `
  --historical-nurec-root D:\300_clip_nurec\hf_probe\pai_nurec_multiclip_v1 `
  --output-root D:\300_clip_nurec\hf_probe\time_mapping_full300_historical_replay_v5
```

The replay uses exactly two semantic boundary pairs, scale `1.0`, and the
existing per-clip offset contract. It never estimates a new global offset.

Run the full observation audit:

```powershell
python scripts/audit_cf_ttc_full300_minimal.py `
  --prediction-jsonl D:\300_clip_nurec\00_raw\ar1_output\reasoning_intervention_nurec_selected_300.jsonl `
  --ground-truth-jsonl D:\300_clip_nurec\00_raw\ground_truth\ego_future_gt_nurec_300.jsonl `
  --nurec-root D:\300_clip_nurec\00_raw\nurec_full300 `
  --time-mapping-jsonl configs/nurec_time_contract_full300.jsonl `
  --manifest-output configs/nurec_cf_ttc_full300_manifest.jsonl `
  --output-root D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5
```

## Acquisition result

The final acquisition report is:

`D:\300_clip_nurec\hf_probe\full300_data_acquisition_v1\full300_acquisition_summary.json`

It records:

- PAI `egomotion.offline`: 300/300 valid files;
- NuRec `sequence_tracks.json`: 300/300 valid files;
- exact component-level acquisition; no full NuRec USDZ download;
- no raw data committed to Git.

The replay report is:

`D:\300_clip_nurec\hf_probe\time_mapping_full300_historical_replay_v5\historical_method_full300_summary.json`

It verifies 300/300 mappings, with zero missing inputs, inconsistent offsets,
duration failures, or pose-validation failures. The canonical mapping file
contains a non-empty provenance `source` for all 300 rows; the accepted pilot
row remains byte-identical to the base contract.

## CF/TTC result

The audit report is:

`D:\300_clip_nurec\hf_probe\cf_ttc_full300_readiness_v5\full300_readiness_final_summary.json`

The full query grid was evaluated:

| Contract | Required | Object-present | Missing | Availability |
|---|---:|---:|---:|---:|
| CF | 12,300 | 12,043 | 257 | 0.9791056911 |
| TTC | 15,300 | 14,995 | 305 | 0.9800653595 |

All 300 clips had sequence tracks and verified time mappings. Thirteen clips
have localized query gaps. The audit keeps those queries as `MISSING`; it does
not infer `CONFIRMED_EMPTY` from absent obstacle rows. Therefore:

- `CF_DATA_READY_FULL_300 = false`;
- `TTC_DATA_READY_FULL_300 = false`;
- `PHYSICAL_WORLD_OBSTACLE_COMPLETENESS = NOT_CLAIMED`;
- `OBSTACLE_GEOMETRY_BLOCK = CLOSED` remains unchanged;
- `PER_CLIP_OFFSET_REDERIVED = false`.

The authoritative triage is in
`cf_ttc_missing_query_triage.csv` and
`minimal_fallback_data_plan.csv` under the audit output directory. The current
blocker is observation-label coverage/empty-frame evidence for those localized
gaps, not acquisition of the 300 time or sequence inputs.

## Tests

`tests/data_prep/test_full300_acquisition.py` covers exact member selection,
ZIP central-directory parsing, ZIP component identity, valid-file reuse,
NuRec's `dummy_chunk_id` wrapper, and fail-closed PAI schema validation.

The next allowed step is a separate decision about obtaining frame-level
empty-attestation evidence or defining a label-set-only scorer contract. This
branch must not convert the remaining missing queries to safe/empty states.
