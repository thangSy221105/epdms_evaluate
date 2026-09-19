# NuRec time forensic hardening

BASE_COMMIT: `7776020708cf0492d14b9960c657f786acc8737c`
BRANCH: `fix/nurec-time-alignment-forensic-hardening`
NEW_COMMIT: `<filled after commit>`

## Forensic tool capability

The resolver is no longer hard-coded to return `UNRESOLVED`. It can return
only evidence-supported statuses:

- `ALIGNED_DIRECT` for explicit shared clock domain, origin, and unit;
- `ALIGNED_BY_EXPLICIT_METADATA` for explicit origin formulas or semantic
  frame/event correspondence with at least two constant-offset pairs;
- `CONFLICTING_TIME_ORIGIN` for conflicting explicit origins or offsets;
- `UNIT_AMBIGUOUS` when a required unit is not declared;
- `MISSING_TIME_METADATA` when required time evidence is absent;
- `UNRESOLVED` when evidence exists but does not prove a mapping.

Numeric range overlap, minimum timestamps, spacing, and filename magnitude are
diagnostic only. They never verify a mapping.

The tool now also:

- inspects prediction relative/absolute waypoint timestamp fields;
- distinguishes GT `t0_us` from timestamp-less waypoint arrays;
- reports prediction condition/t0/timeline consistency;
- marks camera filename units ambiguous unless metadata declares them;
- reports malformed JSONL rows with source and line number;
- writes `input_integrity.json`, `t0_provenance_candidates.json`, and an
  upstream repository search report;
- computes explicit correspondence status and conflict evidence.

## Pilot scientific status

Pilot v2 remains deliberately:

```text
TIME_ALIGNMENT_STATUS = UNRESOLVED
TIME_ALIGNMENT_VERIFIED = false
```

Evidence remains insufficient because no explicit prediction/GT-to-NuRec
origin mapping or semantic frame/event correspondence was found. The pilot
blockers are:

```text
MISSING_EXPLICIT_RELATIVE_TO_GLOBAL_ORIGIN
MISSING_FRAME_CORRESPONDENCE
CLOCK_DOMAIN_NOT_PROVEN
```

New timeline evidence is recorded, but relative prediction `t_s` values and
timestamp-less GT `ego_future_xyz` arrays do not by themselves prove NuRec
clock alignment.

## Verification

```text
data-prep tests = 134 passed
EPDMS tests     = 133 passed
compileall      = passed
pilot output    = D:\300_clip_nurec\hf_probe\time_alignment_v2
```

No scorer, evaluator, metric, coordinate, DAC, or raw data file was changed.
