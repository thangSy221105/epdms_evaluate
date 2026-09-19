# PhysicalAI to NuRec time bridge

## Pilot result

For clip `00040136-e651-4abd-991d-0655ccda9430`, the official Alpamayo query contract is present and `5_100_000` is interpolatable because it lies inside the egomotion range. There is no exact `5_100_000` row; the nearest rows are `5_099_189` and `5_109_194`. The first egomotion timestamp is negative, so timestamp-zero reference, clip-start-is-zero, and spatial-origin-at-zero remain unresolved.

The raw NuRec egomotion/obstacle timestamps are around `27_563_309_000`–`27_583_309_000` microseconds. No source clip ID, converter revision, explicit offset, or semantic cross-domain row identity was found in the pilot metadata. The exact numeric difference is intentionally not promoted to an offset.

## Lineage conclusion

`PhysicalAI → NCore` is `PAI_TO_NCORE_NUMERIC_RETIMING = NONE_FOR_RETAINED_ROWS` with `PAI_TO_NCORE_NEGATIVE_EGO_ROWS = FILTERED`, scale 1.0 and offset 0. `NCore → NuRec/NRE` is `UNRESOLVED`: no public export implementation connecting the NCore timestamp to the raw NuRec `key.timestamp_micros` was found.

Therefore:

`TIME_ALIGNMENT_STATUS = UNRESOLVED`

`TIME_ALIGNMENT_VERIFIED = false`

No `time_alignment_patch.jsonl` is created, and no scorer/evaluator/raw data is modified.
