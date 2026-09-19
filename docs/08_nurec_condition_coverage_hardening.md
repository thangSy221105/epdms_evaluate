# NuRec condition and observation-coverage hardening

BASE_COMMIT: `5a6c7535e5fb3e54136ac10f924b72890e1885f6`
BRANCH: `fix/nurec-data-prep-condition-and-coverage`
NEW_COMMIT: implementation commit reported after commit

This pass changes only NuRec data preparation, tests, and documentation. It
does not modify scorer/evaluator logic, `tools/epdms` implementations, proxy
formula, NAVSIM, clocks, coordinate transforms, DAC conversion, or raw data.

## Closed findings

- Prediction loading is condition-aware and retains all valid rows per clip.
  The canonical identity is reused from `tools.epdms.condition_identity`:
  `clip_id|mode|alpha`.
- Exact repeated condition identities produce
  `PREDICTION_CONDITION_DUPLICATE`; different modes or alphas on one clip are
  valid. Missing/blank clip IDs, mode, alpha, and non-finite alpha produce
  structured `PREDICTION_IDENTITY_INVALID` errors.
- Prediction condition counts, unique counts, modes, alphas, t0 values, and
  frame/anchor metadata values are included in inventory/contracts. Conflicting
  condition t0 values produce `PREDICTION_T0_CONFLICT`; inconsistent
  frame/anchor metadata produces `PREDICTION_COORDINATE_METADATA_CONFLICT`.
- Observation coverage is calculated against the current 10 Hz/4 s CF grid
  and the scorer's `build_ttc_projection_timestamps` grid. Evidence timestamps
  are normalized to a finite integer set, deduplicated, and matched with the
  scorer's 50 ms CF / 100 ms TTC tolerances.
- Coverage is not certified until time alignment is verified. Partial evidence
  reports observed, empty, unknown, and missing counts separately. Obstacle
  timestamps never substitute for independent frame evidence.
- `docs/06_nurec_data_preparation.md` now labels schema_v3/audit_v5/prepared_v5
  as historical and points to v4/v6 and the current v5/v7 outputs.

## Official HF probe

Clip: `00040136-e651-4abd-991d-0655ccda9430`

Current outputs:

- `D:\300_clip_nurec\hf_probe\schema_v5`
- `D:\300_clip_nurec\hf_probe\audit_v7`
- `D:\300_clip_nurec\hf_probe\prepared_v7`

Derived condition result:

```text
prediction condition count        = 16
unique condition count            = 16
modes                             = cross_scene, no_reasoning, noisy, opposite_action
alphas                            = 0, 0.5, 1, 2
prediction t0 values              = [5100000]
duplicate condition keys          = 0
audit errors                      = 0
proxy-ready                       = 0 / 1
```

The remaining blockers are genuine and unchanged in scope:

```text
TIME_ALIGNMENT_UNRESOLVED
COORDINATE_UNRESOLVED
OBSERVATION_COVERAGE_INCOMPLETE
DAC_GEOMETRY_UNVERIFIED
```

Because time alignment is unresolved, v7 correctly leaves CF/TTC required and
matched counts uncertified rather than inventing coverage. The clip still has
the previously observed valid obstacle/map structural data, but no verified
common clock/frame/DAC semantic contract.

## Verification

```text
data-prep tests       = 77 passed
EPDMS regression      = 133 passed
compileall            = passed
```

No raw NuRec file was edited, and the existing untracked user files were not
staged.
