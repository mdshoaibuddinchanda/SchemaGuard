# SchemaGuard Smoke Experiment Review

**Status:** `PASS_PENDING_REVIEW`

`completed_stage`: `smoke_experiment`
`next_stage`: `independent_smoke_review`
`starting_commit`: `acf5c7fa1315bd8e8727f926b32f488f2d05111f`
`source_commit`: `d151eb1db2a5196e9e28814508a0c4cd49147641`
`plan_sha256`: `de24211de43552175461d23fb08342c368fd9c3ff66c128b16c9011c3d99f7c6`
`run_report_sha256`: `88ee48cfea7ef64be2f942d7f0c2e9305dc95ae23738febb3ed9071e0e79059b`
`runtime_probe_inventory_sha256`: `119ea901a3b81de726b154af1ac4b8601605139bc82b5593eaf7787427a42b56`
`validation_status`: `PASS`

The run is a compatibility/consistency smoke only. It does not establish a scientific effect or authorize the pilot, SCNF, COSA, or the main experiment.

## Frozen data, view, and model identities

- Dataset: OpenML 1464, version `openml_file_1586225`.
- Source-data SHA-256: `ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a`.
- Processed feature SHA-256: `9d3168b0bafc5af610caea42c96f95033f32bf5ae4dc011797f45d1809af458f`.
- Target-artifact SHA-256: `9d4e0f665127dd2c41e990d4b129e08d8a0c55f83d42d2e57b758959a9fc23d0`.
- Grouped split: `stratified_group_5fold_v1`, seed `1729`, assignment SHA-256 `e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236`.
- Dependency lock SHA-256: `0b6b2f3903d4cc16860c0da1c35c77096c04c2306256371b5d8181218000762f`.
- Transformation inventory SHA-256: `6984ecef7342a0b7c6190f4636ccc69684e90a7a0343e88fc686362dd9ada104`.
- V01 repair identity: implementation SHA-256 `00548880e1de331d8b6540778b374a17e61838b3bd0f7e81590ff51ab595b034`; cache identity `9191149f3e8097874cb55ef2e19b5409373bed4c1d67667166e9512246b70635`.
- V00 is the canonical `identity` view; V01 is `numeric_affine_units`. The V01 manifest SHA-256 is `d28972842dbc2f8b71967b0afa078284e9c78e7491d7f5c0e590cda5eda467a8`.

| Model | Package/version | Checkpoint | Checkpoint SHA-256 |
|---|---|---|---|
| LR-1.9 | scikit-learn 1.9.1 | None | None |
| CAT-1.2 | catboost 1.2.10 | None | None |
| XGB-3.4 | xgboost 3.4.1 | None | None |
| TPFN3-8.5 | tabpfn 8.5.0 | `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |
| TICL2-2.2 | tabicl 2.2.0 | `tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |

## Environment

- OS: Windows-11-10.0.26200-SP0
- Python: 3.12.14; Conda environment: P12
- CPU: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel; physical/logical cores: 4/8
- RAM total/available at capture: 32494.78515625/13392.59375 MiB
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU; total/free: 4096.0/3305.7000007629395 MiB; driver: 616.64
- CUDA visible: True; PyTorch: 2.6.0+cu124; CUDA build: 12.4
- Runtime package versions: `{"catboost": "1.2.10", "numpy": "2.5.2", "pandas": "3.0.5", "psutil": "7.2.2", "pyarrow": "25.0.1", "pydantic": "2.13.5", "scikit-learn": "1.9.1", "tabicl": "2.2.0", "tabpfn": "8.5.0", "torch": "2.6.0+cu124", "xgboost": "3.4.1"}`
- Telemetry warnings: None

## Runtime estimate and observation

- Estimated cold-run total: 115.2s; CPU critical path: 4.0s; sequential GPU work: 32.8s.
- Observed cold-run wall time: 68.9s.
- Estimate basis: maximum prior successful binary-numerical adapter runtime per model/device, deterministic two-worker CPU list schedule, 1.5 safety factor, and 60 seconds coordination allowance.

## Condition outcomes

The table is rendered from the independently validated resume report, so its `validated_cache_hit` entries mean that the immediate resume revalidated each cold-run result. The cold report records 10 planned, 10 executed, zero cache hits, zero failed, and zero blocked conditions.

| Model | View | Device | Outcome | Cache | CPU/GPU resource evidence | Failure |
|---|---|---|---|---|---|---|
| LR-1.9 | V00 | cpu | PASS | validated_cache_hit | 0.309s; RAM 191.74609375 MiB; VRAM reserved None MiB; telemetry complete | None |
| LR-1.9 | V01 | cpu | PASS | validated_cache_hit | 0.305s; RAM 191.3203125 MiB; VRAM reserved None MiB; telemetry complete | None |
| CAT-1.2 | V00 | cpu | PASS | validated_cache_hit | 1.067s; RAM 221.31640625 MiB; VRAM reserved None MiB; telemetry complete | None |
| CAT-1.2 | V01 | cpu | PASS | validated_cache_hit | 1.086s; RAM 221.03515625 MiB; VRAM reserved None MiB; telemetry complete | None |
| XGB-3.4 | V00 | cpu | PASS | validated_cache_hit | 0.549s; RAM 313.9296875 MiB; VRAM reserved None MiB; telemetry complete | None |
| XGB-3.4 | V01 | cpu | PASS | validated_cache_hit | 0.552s; RAM 313.4765625 MiB; VRAM reserved None MiB; telemetry complete | None |
| TPFN3-8.5 | V00 | cuda | PASS | validated_cache_hit | 8.941s; RAM 1953.37890625 MiB; VRAM reserved 368.0 MiB; telemetry complete | None |
| TPFN3-8.5 | V01 | cuda | PASS | validated_cache_hit | 8.164s; RAM 1934.35546875 MiB; VRAM reserved 368.0 MiB; telemetry complete | None |
| TICL2-2.2 | V00 | cuda | PASS | validated_cache_hit | 3.761s; RAM 1385.9140625 MiB; VRAM reserved 164.0 MiB; telemetry complete | None |
| TICL2-2.2 | V01 | cuda | PASS | validated_cache_hit | 3.709s; RAM 1359.0859375 MiB; VRAM reserved 164.0 MiB; telemetry complete | None |

## Metric summary

Metrics are reported only when every condition completed and the Phase 01 label-access boundary was satisfied.

| Model | View | Partition | Brier | Log loss | Accuracy | Balanced accuracy | AUROC | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LR-1.9 | V00 | calibration | 0.272746 | 0.423079 | 0.786667 | 0.565058 | 0.8265107212475633 | 0.031947 |
| LR-1.9 | V00 | test | 0.350016 | 0.565381 | 0.798658 | 0.620927 | 0.6647869674185464 | 0.102006 |
| LR-1.9 | V01 | calibration | 0.272746 | 0.423079 | 0.786667 | 0.565058 | 0.8265107212475633 | 0.031947 |
| LR-1.9 | V01 | test | 0.350016 | 0.565381 | 0.798658 | 0.620927 | 0.6640350877192983 | 0.102006 |
| CAT-1.2 | V00 | calibration | 0.372954 | 0.661403 | 0.760000 | 0.633041 | 0.6973684210526315 | 0.128542 |
| CAT-1.2 | V00 | test | 0.416247 | 0.769994 | 0.738255 | 0.601253 | 0.6274436090225564 | 0.118885 |
| CAT-1.2 | V01 | calibration | 0.372954 | 0.661403 | 0.760000 | 0.633041 | 0.6973684210526315 | 0.128542 |
| CAT-1.2 | V01 | test | 0.416247 | 0.769994 | 0.738255 | 0.601253 | 0.6274436090225564 | 0.118885 |
| XGB-3.4 | V00 | calibration | 0.365240 | 0.669551 | 0.766667 | 0.589912 | 0.7055311890838206 | 0.145524 |
| XGB-3.4 | V00 | test | 0.446495 | 0.846213 | 0.724832 | 0.572682 | 0.5637844611528823 | 0.169228 |
| XGB-3.4 | V01 | calibration | 0.365240 | 0.669551 | 0.766667 | 0.589912 | 0.7055311890838206 | 0.145524 |
| XGB-3.4 | V01 | test | 0.446495 | 0.846213 | 0.724832 | 0.572682 | 0.5637844611528823 | 0.169228 |
| TPFN3-8.5 | V00 | calibration | 0.276394 | 0.442778 | 0.826667 | 0.676901 | 0.8111598440545809 | 0.070426 |
| TPFN3-8.5 | V00 | test | 0.335872 | 0.554317 | 0.778523 | 0.607769 | 0.6763157894736843 | 0.101203 |
| TPFN3-8.5 | V01 | calibration | 0.277368 | 0.444772 | 0.820000 | 0.672515 | 0.8094541910331384 | 0.054716 |
| TPFN3-8.5 | V01 | test | 0.333752 | 0.552276 | 0.771812 | 0.613283 | 0.6725563909774437 | 0.091405 |
| TICL2-2.2 | V00 | calibration | 0.275708 | 0.435707 | 0.820000 | 0.701023 | 0.8135964912280701 | 0.054713 |
| TICL2-2.2 | V00 | test | 0.334769 | 0.540599 | 0.778523 | 0.627569 | 0.6763157894736842 | 0.100376 |
| TICL2-2.2 | V01 | calibration | 0.275708 | 0.435707 | 0.820000 | 0.701023 | 0.8135964912280701 | 0.054713 |
| TICL2-2.2 | V01 | test | 0.334769 | 0.540599 | 0.778523 | 0.627569 | 0.6763157894736842 | 0.100376 |

## Paired V00/V01 consistency summary

| Model | Partition | Mean JS | Median JS | P90 JS | Max JS | Label flips | Max/mean |ΔP| | ΔBrier | ΔLog loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR-1.9 | calibration | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | -1.1102230246251565e-16 |
| LR-1.9 | test | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | 0.0 |
| CAT-1.2 | calibration | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | 0.0 |
| CAT-1.2 | test | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | 0.0 |
| XGB-3.4 | calibration | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | 0.0 |
| XGB-3.4 | test | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000000/0.000000 | 0.0 | 0.0 |
| TPFN3-8.5 | calibration | 0.00014470 | 0.00002189 | 0.00027382 | 0.00283827 | 0.006667 | 0.052917/0.007069 | 0.0009739092629788004 | 0.0019933715329676915 |
| TPFN3-8.5 | test | 0.00025063 | 0.00011029 | 0.00050188 | 0.00229241 | 0.020134 | 0.055889/0.012982 | -0.0021201019411460353 | -0.0020411271848407297 |
| TICL2-2.2 | calibration | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000001/0.000000 | 1.2229731671808963e-08 | 4.280811899404213e-08 |
| TICL2-2.2 | test | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 0.000002/0.000000 | 1.1393693033490138e-07 | 2.105521357265161e-07 |

## Cache and resume

- Cold execution: 10/10 conditions executed; 6 CPU and 4 CUDA; all passed.
- Mode: resume; executed: 0; validated cache hits: 10.
- Network attempts: 0; duplicate artifacts: 0; output rewrites: 0.
- Resume hashes unchanged: True; mtimes unchanged: True.
- Maximum observed worker process-tree RAM: 1953.4 MiB. Maximum CUDA allocated/reserved memory: 322.3/368 MiB.
- Cold report SHA-256: bc7fc0c72e3f597484b81f5462f95385bc5ccad23214b893b93639cbedb25d86.
- Resume verification SHA-256: 9f071b06e95ff034b60cb0fcce9d1e8e760c5e31797c00f195c985551635a808.

## Phase 01 preservation

- Protected artifact comparison: `PASS` (13 files); unchanged: 13.
- Comparison source snapshot SHA-256: `775bcb934010c05324efbfe5b8ae72dd3d4a6feb83505262e0429ebd653b3361`.
- Grouped split verification: `PASS`; strategy: `stratified_group_5fold_v1`; rows: 748; partitions: `{"calibration": 150, "test": 149, "train": 449}`.
- Predictor groups: 502; duplicates: 69; conflicting targets: 31; crossing groups: 0.
- Deprecated row-stratified split selected: False.

## Acceptance checks

| Gate | Result | Detail |
|---|---|---|
| S01 | PASS | strict plan, report, scheduler identity, and ten-condition order validated |
| S02 | PASS | all ten conditions are accounted for with canonical cache identities |
| S03 | PASS | 20 prediction artifacts passed independent lineage and probability validation |
| S04 | PASS | 20 metric and 10 paired records match strict report values |
| S05 | PASS | Phase 01 hashes independently validated |
| S06 | PASS | grouped split and duplicate-leakage baseline revalidated |
| S07 | PASS | network access remained unused |
| S08 | PASS | test-label access occurred only after model predictions completed |
| S09 | PASS | cold/resume behavior matches the recorded execution mode |
| S10 | PASS | overall smoke result: PASS_PENDING_REVIEW |

## Pre-execution, preservation, and quality checks

- Pre-execution plan: exactly 10 unique conditions (6 CPU, 4 CUDA); exact frozen model and checkpoint identities; plan and source commit matched; all condition cache identities were cold; offline policy enabled; GPU headroom exceeded the configured minimum.
- Complete non-network pytest selection: `conda run -n P12 python -m pytest -q -m "not network and not gpu and not foundation_model" -rA` — exit code 0, all selected tests passed.
- Ruff: `conda run -n P12 python -m ruff check .` — PASS.
- Mypy: `conda run -n P12 python -m mypy src/schemaguard` — PASS, 102 source files.
- Repository validator at source commit: R01–R11 PASS; 52 generated JSON schemas validated; local evidence contracts passed.
- Grouped split validator: all 70 dataset/seed records PASS; aggregate crossing groups 0; protected 1464 split remains 748 rows with 449/150/149 partitions.
- Transformation validator: offline V01 validate-only check PASS; implementation and cache identities match the accepted repair; no transformation implementation or data was changed.
- Model-adapter evidence: PASS, 37/37 records and 37/37 leakage proofs.
- Cache/scheduler evidence: PASS_PENDING_REVIEW, 30/30 fault checks and protected files unchanged.
- Independent smoke evidence validator: PASS; all 10 smoke gates S01–S10 PASS and 20/20 prediction artifacts validated.

## Execution and validation decisions

- Reused the repository's content-addressed scheduler/cache, existing model adapters, existing split and transformation artifacts, and isolated sequential CUDA worker. No frozen model settings, transformations, preprocessing, or resource limits were changed.
- Kept the cold run and immediate resume in one offline invocation; the plan was finalized before test-label access. Test labels were opened only for the sealed post-prediction evaluation boundary.
- Prediction artifacts, caches, runtime JSON, worker logs, and protected-hash receipts remain local under ignored `results/` and cache paths. They are not staged.

## Files created or modified

Implementation files are committed in `d4321e75ea5320b722c7da329d3efca01baaacfd` and validator repair `d151eb1db2a5196e9e28814508a0c4cd49147641`:

- `.github/workflows/quality.yml`
- `artifacts/handoff/smoke_experiment_decisions.md`
- `configs/runtime/smoke_experiment.yaml`
- `schemas/smoke_condition.schema.json`, `schemas/smoke_condition_resource.schema.json`, `schemas/smoke_config.schema.json`, `schemas/smoke_evidence_inventory.schema.json`, `schemas/smoke_metric.schema.json`, `schemas/smoke_paired_metric.schema.json`, `schemas/smoke_plan.schema.json`, `schemas/smoke_prediction_file.schema.json`, `schemas/smoke_protected_foundation_hash_comparison.schema.json`, `schemas/smoke_protected_split_validation.schema.json`, `schemas/smoke_resume_verification.schema.json`, `schemas/smoke_run_report.schema.json`, `schemas/smoke_runtime_estimate.schema.json`, `schemas/smoke_validation_report.schema.json`
- `scripts/run_smoke_experiment.py`, `scripts/validate_smoke_experiment.py`, `scripts/validate_transformation_engine.py`
- `src/schemaguard/artifact_contracts.py`, `src/schemaguard/experiments/__init__.py`, `src/schemaguard/experiments/contracts.py`, `src/schemaguard/experiments/evaluation.py`, `src/schemaguard/experiments/evidence.py`, `src/schemaguard/experiments/execution.py`, `src/schemaguard/experiments/planning.py`
- `tests/integration/smoke_experiment_support.py`, `tests/integration/test_smoke_experiment.py`, `tests/unit/test_artifact_schemas.py`, `tests/unit/test_smoke_contracts.py`, `tests/unit/test_smoke_evaluation.py`, `tests/unit/test_smoke_evidence.py`, `tests/unit/test_smoke_execution.py`, `tests/unit/test_smoke_leakage.py`, `tests/unit/test_smoke_planning.py`, `tests/unit/test_transformation_validator_policy.py`
- Sanitized evidence files: `artifacts/handoff/smoke_experiment_inventory.json` and this review.

The first validator-only attempt's 34 generated files remain recoverable and ignored under `results/smoke/quarantine/nullable_lineage_validator_fix_3128b04a39647163a012c979dfa88249ad3f426ab439f6ee93aef2ba1c98ac7f/`. Current predictions and runtime artifacts remain ignored under `results/smoke/`; no row-level predictions, checkpoint bytes, or cache payloads are included in tracked evidence.

## Commands executed

```text
conda run -n P12 python scripts/run_smoke_experiment.py --config configs/runtime/smoke_experiment.yaml
conda run -n P12 python scripts/validate_smoke_experiment.py --record-evidence
conda run -n P12 python -m ruff check .
conda run -n P12 python -m mypy src/schemaguard
conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence" -rA
```

## Generated local evidence

Prediction rows, model cache payloads, raw worker tracebacks, and resource samples remain in ignored local outputs. This tracked handoff contains no row-level predictions or credentials.

## Deviations

The first cold run under the original smoke implementation produced passing model predictions, but independent evidence validation exposed a null-comparison defect: pandas `Series.ne(None)` treated nullable classical-model checkpoint lineage as unequal. Those 34 generated files were preserved in the ignored quarantine above and were not reused as accepted evidence. The null-aware validator repair was committed as `d151eb1`; a new source-bound plan (`de24211d…`) then cold-executed all ten conditions and passed independent validation. No model condition failed in the accepted run.

Warnings: scikit-learn emitted a `penalty` deprecation warning and TabPFN emitted an `auto_scale_n_estimators` deprecation warning. Frozen parameters were retained unchanged. Pytest's expected single-class metric and small-class split warnings also appeared in tests; the suite exited successfully.

The private root `.docx` was not opened or modified; Git status only confirmed that it remains untracked. The pilot, SCNF, COSA, main experiments, statistical significance testing, paper figures/tables, and paper claims remain unstarted. Independent review is required. GitHub Actions runs are available at <https://github.com/mdshoaibuddinchanda/SchemaGuard/actions>; the exact pushed-commit run will be verified and linked in the delivery message.
