# PhysicalAI to NuRec time bridge

## Pilot result

For clip `00040136-e651-4abd-991d-0655ccda9430`, PhysicalAI egomotion does not contain an exact `5_100_000` row. The nearest rows are `5_099_189` and `5_109_194`. The obstacle table does contain observations in `[4_900_000, 5_300_000]`, but this is only a PhysicalAI-domain observation and is not a NuRec alignment proof.

The raw NuRec egomotion/obstacle timestamps are around `27_563_309_000`–`27_583_309_000` microseconds. No source clip ID, converter revision, explicit offset, or semantic cross-domain row identity was found in the pilot metadata. The exact numeric difference is intentionally not promoted to an offset.

## Lineage conclusion

`PhysicalAI → NCore` is `PRESERVED_EXACTLY` in the inspected public PAI converter. `NCore → NuRec/NRE` is `UNRESOLVED`: no public export implementation connecting the NCore timestamp to the raw NuRec `key.timestamp_micros` was found.

Therefore:

`TIME_ALIGNMENT_STATUS = UNRESOLVED`

`TIME_ALIGNMENT_VERIFIED = false`

No `time_alignment_patch.jsonl` is created, and no scorer/evaluator/raw data is modified.
