# NuRec label semantics contract

This block audits whether NuRec `sequence_tracks.json` and ClipGT
`clipgt/obstacle.parquet` provide enough evidence to classify a missing object
row as an observed empty frame.

The NVIDIA NCore PAI converter consumes obstacle parquet rows as individual
`CuboidTrackObservation` objects and returns an empty observation list when the
obstacle file is unavailable. The public converter and NCore conventions do not
publish a complete label-frame grid or a zero-object sentinel for every frame.
The released NuRec artifacts inspected here expose object-track timestamps and
sensor/pose ranges, but no authoritative label frame list or exact zero-object
attestation.

Therefore the final contract is:

`CONTRACT_C_SPARSE_OBJECT_ROWS_NO_EMPTY_GUARANTEE`

Under this contract, absence of an object row is not an empty-frame assertion.
The 317 previously reported `OUTSIDE_LABEL_TIMELINE` rows are renamed
conceptually to `OUTSIDE_AVAILABLE_OBJECT_ROW_RANGE` and reclassified as
`UNRESOLVED_NO_AUTHORITATIVE_FRAME_GRID`; they are not outside a proven frame
grid. CF/TTC readiness remains fail-closed and scorer/evaluator semantics are
unchanged.

Primary references:

- [NVIDIA NCore PAI converter](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/converter.py)
- [NCore conventions](https://nvidia.github.io/ncore/data/conventions.html)
- [NVIDIA NuRec workflow](https://github.com/NVIDIA/nurec-skills/blob/main/skills/nre/references/example-workflows/bash/nurec_workflow_pai.md)
- [Instant-NuRec input documentation](https://github.com/NVIDIA/instant-nurec/blob/main/README.md)
