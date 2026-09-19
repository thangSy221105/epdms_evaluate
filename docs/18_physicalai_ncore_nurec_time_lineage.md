# PhysicalAI → NCore → NuRec time lineage audit

This document records only timestamp lineage. It does not modify coordinate transforms, DAC, scorers, metrics, evaluator logic, or raw data.

## Proven facts

- Alpamayo queries PhysicalAI egomotion at `t0_us=5_100_000`; an exact source row is not required when the query is inside the official interpolator range.
- The pilot has no exact `5_100_000` egomotion row, but the query is in range and therefore interpolatable.
- `PAI_TO_NCORE_NUMERIC_RETIMING = NONE_FOR_RETAINED_ROWS`.
- `PAI_TO_NCORE_NEGATIVE_EGO_ROWS = FILTERED`.
- PAI→NCore scale is `1.0`, offset `0` for retained rows.
- The pilot's negative first egomotion timestamp does not prove a zero reference or clip-start origin.

## Unresolved edge

No public NCore→NuRec/NRE writer, manifest, source identity, or semantic cross-domain timestamp pair was found. Numeric proximity between timestamp tables is retained only as diagnostic evidence and is never promoted to an offset. The resulting time alignment remains `UNRESOLVED`.

The generated `nurec_source_provenance.json` uses only `FOUND`, `NOT_FOUND_AFTER_INSPECTION`, and `NOT_INSPECTED`.

The final local inspection scope is explicit: `data_info.json`, `datasource_summary.json`, `metadata.yaml`, `parsed_config.yaml`, pose/trajectory metadata, and the six selected `clipgt` parquet files. Each provenance candidate includes its identity class and actual value summary. Generic `clip_id` is `CANDIDATE_ONLY`; only an explicit matching `source_clip_id` can verify source lineage, and even verified source lineage does not verify time mapping.
