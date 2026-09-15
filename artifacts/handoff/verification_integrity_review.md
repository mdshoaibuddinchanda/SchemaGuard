# Verification Integrity Repair Handoff

## Status

PASS

The verification-integrity repair is complete and pushed. GitHub Actions is
green for the final commit. This handoff approves the repair only; it does not
authorize split generation or any later research workstream.

## Stage record

- completed_stage: verification_integrity_repair
- next_stage: independent_review_then_explicit_split_generation_authorization
- Required base commit: 3ace6087488f0e8e9635c70ce5da39d1f672f388
- Implementation commit: fe37b938840496adc9631e55a22cb0926b7bdf3a
- Handoff commit: ba975cbb6d52a08902ebb492daa591b7d11976b4
- Branch: main

## Remote verification

Workflow run: https://github.com/mdshoaibuddinchanda/SchemaGuard/actions/runs/34959791125

Both jobs succeeded:

| Job | Result |
| --- | --- |
| Core quality | Succeeded |
| Classical model validation | Succeeded |

The only remote annotation was GitHub's non-blocking Node.js 20 action warning
for actions/checkout@v4 and actions/setup-python@v5.

## Starting and final repository state

- Starting commit: 3ace6087488f0e8e9635c70ce5da39d1f672f388
- Starting branch: main
- Final branch: main
- Python executable: D:\Conda\P12\python.exe
- Python version: 3.12.14
- Conda environment: P12
- Final worktree: clean except for the user-owned, untracked root reference
  .docx; it was not opened, edited, moved, staged, or committed.

## Repair results

- Added jsonschema to the development dependency contract and regenerated
  uv.lock; the resolved development version is 4.26.0.
- Split GitHub Actions into core-quality and classical-model-validation jobs.
- Reworked repository validation so structural checks run in a clean checkout
  and local data/report checks are explicit evidence checks.
- Added the tracked data-foundation baseline and its generated strict schema.
- Added cross-field GPU report validation and direct contract tests.
- Strengthened ordered feature/target alignment and transactional cleanup
  fault tests.
- Renamed the misleading Windows dependency file to
  requirements/p12_windows_constraints.txt.
- Marked artifacts/handoff/repository_repair_review.md as SUPERSEDED.
- No datasets were acquired, no split was generated, and no model benchmark,
  transformation, prediction, SCNF, COSA, pilot, or main experiment was run.

## Repository hygiene

The naming audit and structural validator both pass. The effective
git check-ignore -v --no-index results were:

| Probe | Result |
| --- | --- |
| schemas/example.schema.json | Not ignored |
| tests/fixtures/example.csv | Not ignored |
| tests/fixtures/example.tsv | Not ignored |
| data/raw/example.csv | Ignored by data/raw/ |
| data/processed/example.parquet | Ignored by data/processed/ |
| results/example.csv | Ignored by results/ |
| artifacts/handoff/example.md | Not ignored |

No generated dataset, result, log, cache, or checkpoint file is tracked.

## Environment and model evidence

- OS: Windows 11
- CPU: Intel64 Family 6 Model 141; 4 physical / 8 logical CPUs
- Installed RAM: 32,494.8 MiB; available at capture: 10,112.5 MiB
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4,095.5 MiB total; 3,305.7 MiB free at capture
- NVIDIA driver: 616.64
- CUDA: visible; compute capability 8.6
- PyTorch: 2.6.0+cu124; CUDA build 12.4

| Model ID | Package | Observed version | CPU result |
| --- | --- | --- | --- |
| LR-1.9 | scikit-learn | 1.9.1 | PASS |
| CAT-1.2 | catboost | 1.2.10 | PASS |
| XGB-3.4 | xgboost | 3.4.1 | PASS |
| TPFN3-8.5 | tabpfn | 8.5.0 | PASS |
| TICL2-2.2 | tabicl | 2.2.0 | PASS |

Checkpoint metadata was validated locally and remains ignored:

| Model ID | Identifier | SHA-256 | License/access |
| --- | --- | --- | --- |
| TPFN3-8.5 | tabpfn-v3-classifier-v3_default.ckpt | d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988 | Package default; resolved |
| TICL2-2.2 | tabicl-classifier-v2-20260212.ckpt | bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0 | Package default; resolved |

Offline reuse evidence reports a cache hit, zero network attempts, unchanged
checkpoint hashes, and PASS for both CPU and CUDA probes.

## Preservation evidence

The data-foundation baseline remains unchanged:

- 748 rows; train 449, calibration 150, test 149
- 502 predictor groups
- 69 duplicated predictor groups
- 31 conflicting-target groups
- Zero predictor groups crossing partitions
- Split strategy: stratified_group_5fold_v1
- Before/after hash comparison: all unchanged

## Tests and clean-checkout evidence

| Check | Result |
| --- | --- |
| Local non-network unit tests | 87 passed |
| Local classical integration | 1 passed |
| Local Ruff | PASS |
| Local Mypy | PASS; 32 source files |
| Local evidence validator | PASS |
| Clean-clone non-network unit tests | 87 passed |
| Clean-clone classical integration | 1 passed |
| Clean-clone schema regeneration diff | PASS; no diff |
| Clean-clone naming audit | PASS |
| Clean-clone structural validator | PASS; local-only checks explicitly not applicable |
| GitHub Actions final run | PASS; both jobs succeeded |

The GPU contract tests cover missing model coverage, unaccounted profiles,
incomplete monitoring, worker exceptions/timeouts, unclassified process exits,
soft-limit breaches, and fallback classification. Fault tests cover truncated
outputs, invalid checksums, interrupted writes, stale/active locks, worker
failure, offline cache misses, API/version mismatches, invalid probabilities,
changed class order, configuration/commit/fixture identity changes, and
protected data-foundation mutation attempts.

## Tracked files changed

- .github/workflows/quality.yml
- artifacts/handoff/repository_repair_review.md
- artifacts/handoff/verification_integrity_review.md
- configs/baselines/data_foundation.json
- pyproject.toml
- requirements/p12_windows_constraints.txt
- schemas/data_foundation_baseline.schema.json
- schemas/gpu_capacity_report.schema.json
- schemas/model_compatibility.schema.json
- schemas/repository_validation.schema.json
- scripts/acquire_schemaorbit.py
- scripts/audit_data_foundation.py
- scripts/check_model_compatibility.py
- scripts/validate_local_evidence.py
- scripts/validate_repository_naming.py
- scripts/validate_repository_repair.py
- src/schemaguard/artifact_contracts.py
- src/schemaguard/compatibility/contracts.py
- src/schemaguard/compatibility/gpu_profiles.py
- src/schemaguard/compatibility/model_probes.py
- src/schemaguard/compatibility/runner.py
- src/schemaguard/data/schemaorbit.py
- tests/unit/test_artifact_schemas.py
- tests/unit/test_fault_injection.py
- tests/unit/test_gpu_report_contract.py
- tests/unit/test_repository_validator.py
- tests/unit/test_resource_monitor.py
- tests/unit/test_schemaorbit.py
- uv.lock

## Commands executed

- conda run -n P12 python -m ruff check .
- conda run -n P12 python -m mypy src/schemaguard
- conda run -n P12 python -m pytest -q tests/unit -m "not network and not gpu and not foundation_model and not evidence" -rA
- conda run -n P12 python -m pytest -q tests/integration/test_classical_model_probes.py -rA
- conda run -n P12 python scripts/generate_artifact_schemas.py
- conda run -n P12 python scripts/validate_repository_naming.py
- conda run -n P12 python scripts/validate_repository_repair.py --local-evidence
- conda run -n P12 python scripts/validate_local_evidence.py
- git commit -m "Repair CI and verification integrity"
- git commit -m "Keep core tests independent of model extras"
- git push origin main

The clean clone used the P12 interpreter with uv sync --extra dev for core
checks and uv sync --extra dev --extra models-cpu for the classical check.

## Deviations and warnings

- The first repair commit's clean-clone run exposed the optional-Torch test
  dependency defect; it was fixed in the final commit and the complete clean
  clone was rerun successfully.
- The existing P12 environment has an unrelated historical pip check warning
  for tableshift; it was not changed and is outside the tested dependency
  groups.
- GitHub reports a non-blocking Node.js action-runtime deprecation warning.

## Next permitted action

This repair is ready for independent review. Split generation remains
NOT_AUTHORIZED by this task. It may be considered only after independent
acceptance of this handoff and explicit authorization in a new request.
