# NuRec coordinate multi-clip validation

This report records the five-clip validation performed after the pilot
coordinate audit. It compares PAI egomotion against NuRec
`rig_trajectories.json` on the same per-clip rebased timeline and the
`t0 + 0.1 ... 6.4 s` future grid.

No SE(2)/SE(3) fit, ICP, Procrustes transform, first-pose subtraction chosen
by visual inspection, or pilot-specific matrix was used. The NuRec pose chain
comes from `T_rig_worlds`; the PAI source comes from the corresponding official
egomotion parquet chunk.

| clip | per-clip offset (us) | XY RMSE (m) | XY P95 (m) | yaw RMSE (deg) |
|---|---:|---:|---:|---:|
| 028508ba-ef59-48d3-a95b-94eb92e3b063 | 3,033,653,000 | 0.061 | 0.097 | 0.335 |
| d078258b-9339-425d-a040-68346ef0d5bc | 23,487,577,000 | 0.032 | 0.046 | 3.460 |
| 689889c5-95b0-42ce-a1c9-f97a4388cb28 | 16,512,637,000 | 0.231 | 0.292 | 1.036 |
| 37f45f87-dc3b-4425-a388-fa7bfa4a11a6 | 12,111,693,000 | 0.145 | 0.256 | 0.390 |
| bb1b395f-c51d-4a16-87ad-7310a7bbf086 | 17,188,651,000 | 0.045 | 0.066 | 0.724 |

## Interpretation

The five clips support a consistent per-clip timeline rebase and a
metadata-backed NuRec pose comparison. This is numerical support for the
`EGO_AT_T0` common-frame algorithm, not a complete coordinate contract.

The status remains:

```text
TIME_ALIGNMENT = RESOLVED_PER_CLIP_REBASE
NUREC_POSE_ALIGNMENT = PASS_NUMERICAL_SUPPORT
COORDINATE_ALIGNMENT = PARTIALLY_VERIFIED
OBSTACLE_FRAME = UNRESOLVED
MAP_FRAME = UNRESOLVED
DRIVABLE_SPACE = MISSING_IN_THE_FIVE_PACKAGES
```

The machine-readable report is kept outside the repository at:

`D:\300_clip_nurec\hf_probe\coordinate_alignment_v1\coordinate_alignment_multiclip.csv`

This block does not change the scorer, evaluator, map loader, prediction/GT
files, or raw NuRec data.
