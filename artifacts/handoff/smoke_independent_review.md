# Independent Ten-Condition Smoke Review

## Verdict

`VERIFIED_PASS` — all 32 independent review gates passed against reviewed commit
`ff92852f98bf482c0ad2b11ab11e9ae290960395`. GitHub Actions run `35291513248` reports success for
all four jobs. The strict machine-readable record and generated schema are
[`smoke_independent_review.json`](smoke_independent_review.json) and
[`smoke_independent_review.schema.json`](smoke_independent_review.schema.json).

This is an implementation and feasibility smoke test, not a benchmark result or evidence of
generalization. No pilot, full experiment, SCNF, COSA, or paper analysis was started.

## Independently reconstructed experiment

The plan reconstructed from the frozen configuration and registries has SHA-256
`de24211de43552175461d23fb08342c368fd9c3ff66c128b16c9011c3d99f7c6`. It contains exactly ten
unique conditions: five frozen models across V00 and V01, with six CPU and four CUDA conditions,
one OpenML 1464 dataset, grouped split `stratified_group_5fold_v1`, and seed 1729. The plan binds
the repaired source commit `d151eb1db2a5196e9e28814508a0c4cd49147641` and was fixed before test-label
access.

The dataset has 748 rows split 449/150/149, 502 predictor groups, 69 duplicated groups, and 31
conflicting-target groups. No predictor group crosses partitions. All 70 split records passed the
offline validator; the protected assignment SHA-256 is
`e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236`.

V01 (`numeric_affine_units`) passed offline validate-only verification, including train-only
fitting, row order, target preservation, certificates, and reconstruction. Its implementation
hash is `00548880e1de331d8b6540778b374a17e61838b3bd0f7e81590ff51ab595b034`, cache identity is
`9191149f3e8097874cb55ef2e19b5409373bed4c1d67667166e9512246b70635`, and manifest SHA-256 is
`d28972842dbc2f8b71967b0afa078284e9c78e7491d7f5c0e590cda5eda467a8`. The three earlier stale V01
materializations remain quarantined and were not used.

## Predictions, metrics, and leakage

All 20 accepted calibration/test prediction artifacts were independently checked for content hash,
condition and source lineage, dataset/split/view/model/checkpoint/dependency identity, partition
membership, stable row alignment, class order, probability validity, and frozen argmax behavior.
The independent review also recomputed all 20 metric records (Brier, log loss, accuracy, balanced
accuracy, AUROC, and ECE) and all 10 paired V00/V01 records (Jensen–Shannon summaries, flips,
probability differences, and metric deltas). Every stored result reproduced within the declared
tolerance. The JSON records each accepted prediction artifact hash without exposing its rows.

Code-path inspection and leakage-focused tests confirm that split assignment precedes view creation,
transform/preprocessing/model fitting use training rows only, calibration/test labels do not affect
planning or execution decisions, and test labels are opened only after all required predictions
exist. Paired outputs align by stable row identity. Cache identity binds the data, split, view, model,
checkpoint, device, seed, dependencies, and source.

The nullable checkpoint-lineage repair is bound to source commit
`d151eb1db2a5196e9e28814508a0c4cd49147641`. The first defective run remains isolated in quarantine:
34 files, 20 prediction files, two metric files, and two run reports. Its plan, conditions, cache
identities, prediction hashes, reports, and metric hashes are disjoint from accepted evidence. Ten
old cache entries still exist locally but are unreferenced by the accepted plan. The repair tests
accept true null lineage and reject mixed-null or mismatched lineage; the accepted ten conditions
were cold-executed after repair, not promoted from quarantine.

## Cache, offline, and resource findings

The cold run planned and executed ten conditions, with zero hits, failures, blocks, or network
attempts. The immediate resume planned ten, performed zero executions, and returned ten validated
cache hits, with no duplicates, rewrites, or network attempts. Prediction hashes and modification
timestamps are unchanged. Resume verification SHA-256 is
`9f071b06e95ff034b60cb0fcce9d1e8e760c5e31797c00f195c985551635a808`.

All four CUDA conditions used the frozen policy sequentially; CPU concurrency remained at two.
Telemetry and cleanup passed. Maximum process-tree RAM was 1953.4 MiB; maximum CUDA allocated and
reserved memory was 322.3 MiB and 368 MiB. The observed run duration was approximately 68.9 seconds,
consistent with the 115.2-second conservative estimate. These measurements establish feasibility,
not scientific performance. Frozen checkpoint hashes match the plan; their digests are in the JSON.

## Numerical findings

Logistic Regression changed probabilities by at most `2.220446049250313e-16`, with zero label flips
and zero Brier change. Six positive/negative ordering comparisons shifted at machine precision,
reproducing the raw AUROC change from `0.6647869674185464` to `0.6640350877192983` (delta
`-0.0007518796992480592`). Treating differences within `1e-15` as ties removes the AUROC change.
This is `NUMERICAL_TIE_SENSITIVITY`, not scientific schema instability. The pilot protocol must
freeze tie-handling and reporting before any pilot execution. The report does not describe the two
probability arrays as exactly identical.

TabPFN’s test result independently reproduces three flips among 149 rows (rate `0.020134228187919462`),
maximum probability difference `0.05588936805725098`, mean difference `0.012981989705347575`, and
mean Jensen–Shannon divergence `0.00025063003001823974`. The requested Brier, accuracy, balanced
accuracy, AUROC, and ECE deltas also reproduce. This is a micro-fixture signal, not evidence of
generalization or improvement. Row-level diagnostics, including the flipped row identities, remain
only in ignored local review evidence.

CatBoost and XGBoost probabilities are identical between views. TabICL has zero test flips, maximum
probability difference approximately `1.55e-6`, and mean test Jensen–Shannon divergence approximately
`2.33e-13`; all comparisons use aligned rows and the same class order.

## Preservation and execution checks

Before/after SHA-256, size, and modification-time comparisons found no changes among 13 protected
Phase 01 files, 140 split artifacts, 20 predictions, two metric files, 30 accepted cache files,
seven V01 materialization files, 34 quarantined repair files, or 123 auxiliary files. The two frozen
checkpoints and three transformation-quarantine directories also remained unchanged.

The root and exact-commit clean-clone unit suites each passed 443 tests; the smoke integration test
passed in both environments. Ruff, Mypy (102 source files), schema generation, naming validation,
repository validation, and `git diff --check` passed. The clean clone correctly treats ignored local
smoke artifacts as `NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT`; it does not claim to reproduce local
predictions or GPU/checkpoint evidence. GitHub Actions independently reports four successful jobs.

To avoid rewriting the protected local validation report, the review invoked the smoke validator’s
read-only `validate_smoke_report()` core directly rather than its CLI wrapper, which writes that
report. The core returned `PASS` for all ten S01–S10 gates. The exact command outputs and detailed
row-level diagnostics remain under ignored local review storage.

The preexisting private untracked reference remained untouched and untracked. No dataset,
transformation, checkpoint, cache, prediction, model parameter, or source implementation was changed
by this review.

## Independent review gates

| Gate | Result | Evidence |
| --- | --- | --- |
| IR01 Repository identity | PASS | Required branch/commit; clean tracked tree before publication |
| IR02 GitHub Actions identity | PASS | Exact workflow run; four successful jobs |
| IR03 Plan reconstruction | PASS | Recomputed canonical plan SHA-256 |
| IR04 Exact condition matrix | PASS | 10 unique; five models × two views; CPU/CUDA split verified |
| IR05 Dataset identity | PASS | OpenML 1464 source, feature, and target identity |
| IR06 Grouped split integrity | PASS | Required counts, assignment hash, zero crossings |
| IR07 V00/V01 transformation integrity | PASS | Offline V01 validation and required hashes |
| IR08 Validator repair provenance | PASS | Repaired source-bound plan and mismatch mutation tests |
| IR09 Quarantine isolation | PASS | 34 files isolated; accepted identities are disjoint |
| IR10 Prediction artifact completeness | PASS | 20/20 independently validated and hashed |
| IR11 Probability validity | PASS | Finite, bounded, normalized probabilities |
| IR12 Row alignment | PASS | Exact partition membership and stable alignment |
| IR13 Class-order integrity | PASS | Canonical `[0, 1]` order |
| IR14 Leakage boundary | PASS | Source inspection and leakage-focused tests |
| IR15 Metric recomputation | PASS | 20/20 records reproduced |
| IR16 Paired-metric recomputation | PASS | 10/10 records reproduced |
| IR17 Logistic Regression tie analysis | PASS | Machine-precision cause and tie-aware AUROC reproduced |
| IR18 TabPFN signal | PASS | Three flips and all requested metrics reproduced |
| IR19 TabICL stability | PASS | No flips; probability/divergence results reproduced |
| IR20 Cold-run accounting | PASS | 10 executed; no hits, errors, blocks, or network attempts |
| IR21 Resume/cache integrity | PASS | 10 validated hits; no writes or identity failures |
| IR22 Resource telemetry | PASS | Worker bounds, RAM/VRAM, telemetry, and cleanup verified |
| IR23 Offline enforcement | PASS | Frozen local checkpoints; zero network attempts |
| IR24 Phase 01 preservation | PASS | All 13 protected artifact hashes unchanged |
| IR25 Split preservation | PASS | 70/70 pass; 140 files unchanged |
| IR26 Transformation evidence preservation | PASS | Accepted V01 and three quarantines unchanged |
| IR27 Adapter evidence preservation | PASS | Inventories and leakage evidence unchanged |
| IR28 Scheduler evidence preservation | PASS | Inventory, faults, probes, and comparison unchanged |
| IR29 Tests, lint, and types | PASS | Root and clone suites; Ruff; Mypy; validators |
| IR30 Clean-clone portability | PASS | Exact-commit clone passes; local outputs explicitly N/A |
| IR31 Schema freshness | PASS | Generated schemas produce no tracked diff |
| IR32 Repository hygiene | PASS | Only permitted review/status files published; private reference untouched |

## Next permitted workstream

Pilot-protocol freezing may begin. This review does not start or authorize pilot execution; the pilot
and main experiment remain unstarted. No commit or push may include local raw outputs, caches,
checkpoints, datasets, temporary audit helpers, or the private untracked reference.
