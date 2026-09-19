# Multi-clip NCore → NuRec time-pattern experiment

## Dataset inventory

The official manifests contain 1,147 NCore clip UUIDs and 1,607 NuRec sample-set clip UUIDs. Their exact UUID intersection contains 51 clips, so the experiment selects 30 clips and includes the required pilot only if it is in the intersection.

The pilot `00040136-e651-4abd-991d-0655ccda9430` is present in the official NuRec manifest but absent from the official NCore manifest. It therefore cannot inherit a multi-clip pattern automatically.

## Metadata availability

The NCore manifest exposes small per-clip JSON metadata, including `sequence_id`, `source_clip_id`, converter version, and a sequence timestamp interval. The official NuRec overlap manifest exposes `.usdz` and `.mp4` payload entries but does not expose the requested `data_info.json`, pose, egomotion, or obstacle timestamp metadata in the manifest. No video, LiDAR, mesh, or USDZ payload was downloaded for this experiment.

Consequently, 30 exact-identity overlaps were inventoried, but zero clips had both NCore and NuRec timestamp vectors available for semantic or interval comparison.

## Pattern result

`PATTERN_STATUS = INSUFFICIENT_EVIDENCE` and `VERIFICATION_STATUS = UNVERIFIED`. Duration invariants, relative-clock comparison, semantic pairs, global offsets, per-sequence offsets, and per-clip offsets cannot be evaluated from the exposed minimal metadata. No numeric offset is inferred.

The prior scientific conclusion remains unchanged: `NCORE_TO_NUREC_MAPPING = UNRESOLVED`, `TIME_ALIGNMENT_VERIFIED = false`, and `OFFSET_US = null`. The multi-clip search is exhausted for the publicly exposed minimal manifests; additional evidence would require a source export/manifest or NuRec timestamp metadata for the same clips.
