# Grouped Split Data Foundation Review

## Verdict

`VERIFIED_PASS`

Phase 02 remains not started.

## Active dataset

- OpenML data ID: `1464`
- OpenML file ID: `1586225`
- Dataset: `blood-transfusion-service-center`
- Rows: `748`
- Predictors: `4 numeric`
- Target classes: `2`

## Active split

Location: `data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/`

- Strategy: `stratified_group_5fold_v1`
- Grouping: exact type-aware predictor-vector SHA-256 IDs
- Fold assignment: folds `0,1,2` train; fold `3` calibration; fold `4` test
- Actual sizes: train `449`, calibration `150`, test `149`
- Class counts: train `342/107`, calibration `114/36`, test `114/35`
- Predictor groups: `502` total; `69` duplicated; largest group size `35`
- Conflicting-target groups: `31`, explicitly detected and reported
- Predictor duplicate groups crossing partitions: `0`
- Size deviations: train `0.00026738`, calibration `0.00053476`, test `0.00080214`
- Largest class-proportion deviation: `0.00306859`

## Determinism and coverage

- Every row is assigned exactly once.
- Split sets are disjoint and contain all 748 rows.
- Both target classes occur in every partition.
- Repeated generation is identical.
- Shuffled input order produces identical assignments.
- Group IDs are independent of row order and exclude row IDs and target columns.

## Deprecated baseline

The former row-stratified split remains preserved at
`data/splits/openml/1464/seed_1729/` and is deprecated, not active.

- Assignments SHA-256: `b1c639744599425b530c1162df7bf36e16e3abc47aa2bd51be1e030c28ba5f2f`
- Manifest SHA-256: `6e3ead959286ecfeb255ba4079ac5fdb8f9111aa90458942acbb381abdb5e9ca`

## Verification

- `uv sync --extra dev`: passed
- Ruff: passed
- Mypy: passed
- Non-network tests: `30 passed`
- Network tests: `1 passed`
- Online pipeline: passed
- Offline pipeline: passed
- Independent acceptance gates: `37/37 passed`
- Generated data and cache files are ignored and not staged.

Evidence files are retained locally under `artifacts/phase_01_data_foundation/review/`.
