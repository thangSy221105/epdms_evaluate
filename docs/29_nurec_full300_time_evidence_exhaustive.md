# Exhaustive NuRec full-300 time evidence

Run the exhaustive local evidence audit with:

```powershell
python scripts/recover_nurec_time_evidence_exhaustive.py
```

The audit explicitly parses `nurec_context_full_300.jsonl` one JSONL record at
a time, inventories the known per-clip metadata files, and searches lightweight
JSON/JSONL/CSV/YAML contents under the configured local roots. Filename matches
are not verification criteria. The discovery registry may contain numeric-only,
declared-timestamp, explicit-origin, or semantic-pair candidates, while the
canonical contract contains only verified rows.

`sequence_tracks.json` is used only for the post-hoc scorer-support window
(`t0 + 5.0 s`). It cannot create a clock offset. The historical five-clip
contract remains immutable, and the current output root is excluded from its
own discovery scan so reruns cannot self-verify.
