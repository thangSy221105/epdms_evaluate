# NuRec `t0_us` upstream trace

This document records the line-level repository search performed by the time-alignment forensic audit. It does not treat tests, configuration, or a repeated numeric literal as generation provenance.

## Required conclusion

For the inspected prediction/GT contract, `t0_us=5100000` is present as a record field. The repository contains no verified assignment chain connecting that value to a raw NuRec frame, pose, sample token, or `timestamp_micros`. Therefore `WHY_IS_T0_5100000` is `NOT_FOUND` for a raw NuRec bridge (or `DERIVED_BUT_NO_NUREC_BRIDGE` if a future upstream generator match is found).

## Trace record format

The machine-readable report uses:

`FILE / LINE / FUNCTION / MATCH / UPSTREAM INPUT / DERIVED VALUE / DOWNSTREAM OUTPUT / PROVENANCE STATUS`

See `upstream_assignment_trace.json` in the generated pilot output for every matched line and its surrounding context. Candidate lines are evidence for review only; they do not verify alignment.

## Search scope

The trace searches source and project metadata for `t0_us`, `5100000`, waypoint/timestamp fields, frame/sample identity, NuRec parquet readers, pose/egomotion sources, and the requested indexing/frequency patterns. It also records the nearest enclosing Python function where available.

## Forensic rule

An index-derived timestamp, a fixed horizon (`64` points), or a generic sequence offset is not a NuRec bridge unless an explicit frame/index-to-timestamp mapping is present. No guessed offset is applied.
