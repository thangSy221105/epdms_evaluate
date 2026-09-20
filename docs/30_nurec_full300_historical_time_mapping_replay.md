# NuRec full-300 historical time-mapping replay

This branch replays the accepted five-clip PAI/NuRec time contract before
attempting the current 300-clip batch.

The frozen contract is a per-clip unit-scale rebase:

```text
NuRec_timestamp_us = PAI_relative_timestamp_us + offset_us
scale = 1.0
```

The replay uses only the historical start and end semantic timeline
correspondences, requires two equal integer offsets, and requires equal
relative duration.  Pose trajectory errors are reported as independent
diagnostics; no new numeric threshold or fitted transform is introduced.

The public PAI converter is the provenance for the non-negative PAI interval
and the pose field convention (`timestamp`, `qx`, `qy`, `qz`, `qw`, `x`, `y`,
`z`).  NuRec timestamps and poses come from
`rig_trajectories.json:T_rig_world_timestamps_us` and `T_rig_worlds`.

`sequence_tracks.json` is not used to derive `offset_us` or `nurec_t0_us`.
It remains an obstacle-source artifact for later CF/TTC preparation.

Run:

```powershell
python scripts/replay_nurec_historical_time_mapping_full300.py
```

Reports are written to
`D:\300_clip_nurec\hf_probe\time_mapping_full300_historical_replay_v1`.
The current 300-clip canonical contract is updated only with verified rows.
Historical clips outside the current manifest are replayed as a separate
regression set and do not inflate current batch coverage.

No full NuRec bundle is downloaded and no raw file is modified.
